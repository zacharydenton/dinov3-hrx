//! DINOv3 ViT inference at 224×224, specialized by model type.
//! Inputs are normalized RGB NCHW; outputs are tokens or compact descriptors.
mod checkpoint;
mod encoder;
pub use encoder::{
    Encoder, EncoderLayerWeights, EncoderOptions, EncoderWeights, LinearWeights, NormWeights,
    RotaryEmbedding,
};
pub mod hub;
mod pooling;
mod preprocess;
mod spec;
mod weights;
use anyhow::{Result, ensure};
use bytemuck::Zeroable;
use hrx::{
    execution::{Graph, MemoryPlacement},
    image::ImageOps,
    inference::{Inference, InferenceGraph, ModelContext, PreparedModel},
    loom::Specialization,
    model::{Command, Dispatch, KernelId, ModelDefinition, ModelFragment, ModelSession, Region},
    plan_cache::PlanCache,
    tensor::{DType, DeviceTensor, Layout, TensorDesc},
};
pub use spec::{EncoderSpec, ModelSpec, ViT7B16, ViTB16, ViTH16Plus, ViTL16, ViTS16, ViTS16Plus};
use std::marker::PhantomData;
use std::{collections::HashMap, path::Path};

pub const IMAGE_ELEMENTS: usize = 3 * 224 * 224;
pub const TOKENS: usize = 201;
/// Feature width of the default ViT-S+/16 model.
pub const HIDDEN: usize = 384;
/// Resident allocation limit. Larger input batches are processed in chunks.
#[derive(Clone, Copy, Debug)]
pub struct Options {
    pub device: i32,
    pub max_batch: usize,
}
impl Default for Options {
    fn default() -> Self {
        Self {
            device: 0,
            max_batch: 32,
        }
    }
}
/// Shared resident model with bounded private slots for each cached shape.
pub struct DINOv3Model<M: ModelSpec> {
    marker: PhantomData<M>,
    engine: ModelDefinition,
    encoder: Encoder<M>,
    plans: PlanCache<(usize, bool, bool), PreparedModel>,
    pooling: PlanCache<usize, PreparedModel>,
    summaries: PlanCache<(usize, bool), PreparedModel>,
    images: ImageOps,
    options: Options,
    weights: HashMap<String, Region>,
    a: Vec<Region>,
    kernels: Vec<KernelId>,
}
/// Default ViT-S+/16 model; preserves the original API.
pub type DINOv3 = DINOv3Model<ViTS16Plus>;
/// ViT-B/16 model with 768-feature outputs.
pub type DINOv3ViTB = DINOv3Model<ViTB16>;
/// ViT-S/16 with GELU (distinct from the default ViT-S+).
pub type DINOv3ViTS = DINOv3Model<ViTS16>;
/// ViT-L/16 with 1024-feature outputs.
pub type DINOv3ViTL = DINOv3Model<ViTL16>;
/// ViT-H+/16 with 1280-feature outputs.
pub type DINOv3ViTH = DINOv3Model<ViTH16Plus>;
/// ViT-7B/16 with 4096-feature outputs and 128-channel attention heads.
pub type DINOv3ViT7B = DINOv3Model<ViT7B16>;
impl<M: ModelSpec> DINOv3Model<M> {
    /// Feature width of this model architecture.
    pub const HIDDEN: usize = M::HIDDEN;

    /// Load the pinned pretrained model from the Hugging Face cache, fetching it
    /// if needed. Set `HF_HUB_OFFLINE=1` for cached weights only.
    /// Use [`Self::load`] to supply a local file instead.
    pub fn from_pretrained(options: Options) -> Result<Self> {
        ensure!(
            (1..=64).contains(&options.max_batch),
            "max_batch must be 1..=64"
        );
        Self::load(hub::weights_for::<M>(false)?, options)
    }

    /// Validate and pack the model, compile kernels, and allocate resident storage.
    /// Accepts a SafeTensors file, a shard index, or a checkpoint directory.
    pub fn load(path: impl AsRef<Path>, options: Options) -> Result<Self> {
        ensure!(
            (1..=64).contains(&options.max_batch),
            "max_batch must be 1..=64"
        );
        let packed = weights::load::<M>(path.as_ref())?;
        let context = ModelContext::new(hrx::execution::RuntimeOptions {
            gpu_index: options.device,
            ..Default::default()
        })?;
        Self::from_packed(packed, &context, options.max_batch)
    }
    /// Load into the caller's shared allocation, compiler and scheduling domain.
    pub fn load_in(
        path: impl AsRef<Path>,
        context: &ModelContext,
        max_batch: usize,
    ) -> Result<Self> {
        ensure!((1..=64).contains(&max_batch), "max_batch must be 1..=64");
        let packed = weights::load::<M>(path.as_ref())?;
        Self::from_packed(packed, context, max_batch)
    }
    fn from_packed(
        mut packed: HashMap<String, Vec<u8>>,
        context: &ModelContext,
        max_batch: usize,
    ) -> Result<Self> {
        let options = Options {
            device: context.runtime().gpu()?.index(),
            max_batch,
        };
        ensure!(
            context.runtime().gpu()?.target().as_str() == "gfx1151",
            "DINOv3 requires gfx1151"
        );
        let mut engine = ModelSession::in_context(context)?;
        let mut weights = HashMap::new();
        for name in ["patch_w", "patch_b", "prefix"] {
            weights.insert(
                name.to_owned(),
                engine.weight(&packed.remove(name).expect("validated checkpoint"))?,
            );
        }
        let encoder = Encoder::<M>::from_packed(
            packed,
            context,
            EncoderOptions {
                max_batch,
                prefix_tokens: 5,
                ..Default::default()
            },
        )?;
        let sizes = [
            max_batch * 196 * 768 * 4,
            max_batch * 196 * M::HIDDEN * 4,
            max_batch * TOKENS * M::HIDDEN * if M::RESIDUAL_F32 { 4 } else { 2 },
        ];
        let a = sizes
            .into_iter()
            .map(|n| engine.allocate(n))
            .collect::<std::result::Result<_, _>>()?;
        // Every source is embedded in this crate and its bindings are declared below.
        let kernels = unsafe { engine.compile(&embedding_specifications::<M>())? };
        let engine = engine.freeze(context)?;
        Ok(Self {
            marker: PhantomData,
            engine,
            encoder,
            plans: PlanCache::new(8, PreparedModel::is_idle)?,
            pooling: PlanCache::new(8, PreparedModel::is_idle)?,
            summaries: PlanCache::new(8, PreparedModel::is_idle)?,
            images: ImageOps::new(context, 8)?,
            options,
            weights,
            a,
            kernels,
        })
    }
    /// Shared context used by this model and its preprocessing operations.
    pub fn context(&self) -> &ModelContext {
        self.engine.context()
    }
    /// The resident encoder, usable with already embedded tokens in caller graphs.
    pub fn encoder(&self) -> &Encoder<M> {
        &self.encoder
    }

    /// Pool resident tokens to normalized CLS and masked patch-mean descriptors.
    /// Mask is U8 `[batch,196]`, nonzero selects a patch. Empty masks produce a
    /// zero mean descriptor. Only `[batch,2,M::HIDDEN]` descriptors need downloading.
    pub fn pool_descriptors(
        &self,
        tokens: &DeviceTensor,
        mask: &DeviceTensor,
    ) -> Result<Inference> {
        let shape = tokens.desc().shape();
        ensure!(
            shape.len() == 3
                && shape[1..] == [TOKENS, M::HIDDEN]
                && (1..=self.options.max_batch).contains(&shape[0]),
            "invalid token shape"
        );
        let plan = self.pooling.get_or_prepare(shape[0], || {
            pooling::fragment::<M>(self.context(), shape[0])?.prepare(3)
        })?;
        Ok(plan.submit(&[tokens.clone(), mask.clone()])?)
    }
    /// Describe normalized host images, keeping full tokens on the device.
    /// Inputs may exceed max_batch; `masks` supplies one byte per patch/image.
    pub fn descriptors(&self, pixels: &[f32], masks: &[u8]) -> Result<Vec<f32>> {
        ensure!(
            pixels.len().is_multiple_of(IMAGE_ELEMENTS) && pixels.iter().all(|x| x.is_finite()),
            "invalid image input"
        );
        let batch = pixels.len() / IMAGE_ELEMENTS;
        ensure!(masks.len() == batch * 196, "mask count mismatch");
        let mut output = vec![0f32; batch * 2 * M::HIDDEN];
        for ((input, mask), out) in pixels
            .chunks(self.options.max_batch * IMAGE_ELEMENTS)
            .zip(masks.chunks(self.options.max_batch * 196))
            .zip(output.chunks_mut(self.options.max_batch * 2 * M::HIDDEN))
        {
            let b = input.len() / IMAGE_ELEMENTS;
            self.prepare_pipeline(b, false, true)?
                .acquire_blocking()?
                .submit_host(&[bytemuck::cast_slice(input), mask])?
                .download()?
                .read_into(&mut [bytemuck::cast_slice_mut(out)])?;
        }
        Ok(output)
    }
    /// Normalize RGB, patchify, infer and pool on the GPU. Only the final
    /// `[batch,2,M::HIDDEN]` descriptors cross back to the host. Masks are U8 `[batch,196]`.
    pub fn describe_rgb(&self, rgb: &[u8], masks: &[u8]) -> Result<Vec<f32>> {
        ensure!(
            rgb.len().is_multiple_of(IMAGE_ELEMENTS),
            "expected complete 224×224 RGB images"
        );
        let batch = rgb.len() / IMAGE_ELEMENTS;
        ensure!(masks.len() == batch * 196, "mask count mismatch");
        let mut output = vec![0f32; batch * 2 * M::HIDDEN];
        for ((input, mask), out) in rgb
            .chunks(self.options.max_batch * IMAGE_ELEMENTS)
            .zip(masks.chunks(self.options.max_batch * 196))
            .zip(output.chunks_mut(self.options.max_batch * 2 * M::HIDDEN))
        {
            let b = input.len() / IMAGE_ELEMENTS;
            self.prepare_pipeline(b, true, true)?
                .acquire_blocking()?
                .submit_host(&[input, mask])?
                .download()?
                .read_into(&mut [bytemuck::cast_slice_mut(out)])?;
        }
        Ok(output)
    }
    /// Submit normalized device NCHW input, patchifying on the GPU. Output tokens
    /// remain resident and hold their inference slot until all consumers release it.
    pub fn submit(&self, pixels: &DeviceTensor) -> Result<Inference> {
        self.context().validate(pixels)?;
        let desc = pixels.desc();
        ensure!(
            desc.dtype() == DType::F32
                && desc.layout() == Layout::Nchw
                && desc.is_contiguous()
                && desc.shape().len() == 4
                && desc.shape()[1..] == [3, 224, 224],
            "expected normalized NCHW f32 input"
        );
        Ok(self
            .prepare_pipeline(desc.shape()[0], false, false)?
            .submit(std::slice::from_ref(pixels))?)
    }
    /// Infer an arbitrary batch of normalized NCHW RGB images.
    pub fn forward(&self, pixels: &[f32]) -> Result<Vec<f32>> {
        ensure!(
            pixels.len().is_multiple_of(IMAGE_ELEMENTS),
            "input must contain complete 3×224×224 images"
        );
        ensure!(pixels.iter().all(|x| x.is_finite()), "input must be finite");
        let batch = pixels.len() / IMAGE_ELEMENTS;
        let mut output = vec![0.; batch * TOKENS * M::HIDDEN];
        for (input, output) in pixels
            .chunks(self.options.max_batch * IMAGE_ELEMENTS)
            .zip(output.chunks_mut(self.options.max_batch * TOKENS * M::HIDDEN))
        {
            let b = input.len() / IMAGE_ELEMENTS;
            self.prepare_pipeline(b, false, false)?
                .acquire_blocking()?
                .submit_host(&[bytemuck::cast_slice(input)])?
                .download()?
                .read_into(&mut [bytemuck::cast_slice_mut(output)])?;
        }
        Ok(output)
    }
    /// Synchronized warm host latency including GPU patchification and transfers.
    pub fn benchmark(
        &self,
        pixels: &[f32],
        samples: usize,
    ) -> Result<hrx::benchmark::Distribution> {
        ensure!(
            !pixels.is_empty() && pixels.len() <= self.options.max_batch * IMAGE_ELEMENTS,
            "benchmark requires one nonempty batch"
        );
        ensure!(samples >= 10, "use at least ten timing samples");
        for _ in 0..10 {
            self.forward(pixels)?;
        }
        let mut times = Vec::with_capacity(samples);
        for _ in 0..samples {
            let start = std::time::Instant::now();
            self.forward(pixels)?;
            times.push(start.elapsed().as_secs_f64() * 1000.);
        }
        Ok(hrx::benchmark::Distribution::from_samples(times)?)
    }
    /// Run inference and return the CLS token from each image.
    pub fn cls(&self, pixels: &[f32]) -> Result<Vec<M::Row>> {
        self.raw_summary(pixels, false)
    }
    /// Run inference and average the 196 patch tokens for each image.
    pub fn patch_mean(&self, pixels: &[f32]) -> Result<Vec<M::Row>> {
        self.raw_summary(pixels, true)
    }
    fn raw_summary(&self, pixels: &[f32], mean: bool) -> Result<Vec<M::Row>> {
        ensure!(
            pixels.len().is_multiple_of(IMAGE_ELEMENTS),
            "input must contain complete 3×224×224 images"
        );
        ensure!(pixels.iter().all(|x| x.is_finite()), "input must be finite");
        let mut output = vec![M::Row::zeroed(); pixels.len() / IMAGE_ELEMENTS];
        for (input, output) in pixels
            .chunks(self.options.max_batch * IMAGE_ELEMENTS)
            .zip(output.chunks_mut(self.options.max_batch))
        {
            let batch = output.len();
            let plan = self.summaries.get_or_prepare((batch, mean), || {
                PreparedModel::prepare(self.context(), 3, |context| {
                    let input = context.allocate_with(
                        TensorDesc::new(DType::F32, vec![batch, 3, 224, 224])?
                            .with_layout(Layout::Nchw)?,
                        MemoryPlacement::HostVisible,
                    )?;
                    let mut graph = context.runtime().graph();
                    let tokens = self
                        .record(&mut graph, &input)
                        .map_err(|e| hrx::Error::Message(e.to_string()))?;
                    let output = pooling::raw_fragment::<M>(context, batch, mean)?
                        .record(&mut graph, &[tokens])?;
                    Ok(InferenceGraph {
                        inputs: vec![input],
                        outputs: output,
                        graph: graph.prepare()?,
                    })
                })
            })?;
            plan.acquire_blocking()?
                .submit_host(&[bytemuck::cast_slice(input)])?
                .download()?
                .read_into(&mut [bytemuck::cast_slice_mut(output)])?;
        }
        Ok(output)
    }
    /// Record normalized NCHW F32 pixels through patchification and transformer
    /// inference into a caller-owned graph. The returned token tensor is valid
    /// after that graph executes; no private inference pool or copies are used.
    pub fn record(&self, graph: &mut Graph, pixels: &DeviceTensor) -> Result<DeviceTensor> {
        self.context().validate(pixels)?;
        let desc = pixels.desc();
        ensure!(
            desc.dtype() == DType::F32
                && desc.layout() == Layout::Nchw
                && desc.is_contiguous()
                && desc.shape().len() == 4
                && desc.shape()[1..] == [3, 224, 224],
            "expected normalized NCHW f32 input"
        );
        let model = self.fragment(desc.shape()[0])?;
        let patches = self
            .images
            .patchify_fragment(desc, 16)?
            .record(graph, std::slice::from_ref(pixels))?;
        let embedded = model.record(graph, &patches)?.remove(0);
        self.encoder.record_owned(graph, embedded)
    }

    /// Record normalization (for U8 NHWC RGB), patchification, transformer and
    /// masked descriptor pooling in one graph. F32 NCHW inputs are already
    /// normalized. Only the final `[batch,2,M::HIDDEN]` descriptors need be downloaded.
    pub fn record_descriptors(
        &self,
        graph: &mut Graph,
        pixels: &DeviceTensor,
        masks: &DeviceTensor,
    ) -> Result<DeviceTensor> {
        self.context().validate(pixels)?;
        self.context().validate(masks)?;
        let batch = pixels.desc().shape().first().copied().unwrap_or(0);
        ensure!(
            masks.desc() == &TensorDesc::new(DType::U8, vec![batch, 196])?,
            "invalid descriptor masks"
        );
        let tokens = if pixels.desc().dtype() == DType::U8 {
            ensure!(
                pixels.desc()
                    == &TensorDesc::new(DType::U8, vec![batch, 224, 224, 3])?
                        .with_layout(Layout::Nhwc)?,
                "expected contiguous 224x224 RGB input"
            );
            let patches = preprocess::fragment(self.context(), batch)?
                .record(graph, std::slice::from_ref(pixels))?;
            let embedded = self.fragment(batch)?.record(graph, &patches)?.remove(0);
            self.encoder.record_owned(graph, embedded)?
        } else {
            self.record(graph, pixels)?
        };
        Ok(pooling::fragment::<M>(self.context(), batch)?
            .record(graph, &[tokens, masks.clone()])?
            .remove(0))
    }

    fn prepare_pipeline(
        &self,
        batch: usize,
        rgb: bool,
        descriptors: bool,
    ) -> Result<std::sync::Arc<PreparedModel>> {
        ensure!(
            (1..=self.options.max_batch).contains(&batch),
            "invalid batch size"
        );
        Ok(self.plans.get_or_prepare((batch, rgb, descriptors), || {
            PreparedModel::prepare(self.context(), 3, |context| {
                let pixels = context.allocate_with(
                    if rgb {
                        TensorDesc::new(DType::U8, vec![batch, 224, 224, 3])?
                            .with_layout(Layout::Nhwc)?
                    } else {
                        TensorDesc::new(DType::F32, vec![batch, 3, 224, 224])?
                            .with_layout(Layout::Nchw)?
                    },
                    MemoryPlacement::HostVisible,
                )?;
                let mut graph = context.runtime().graph();
                let (inputs, output) = if descriptors {
                    let mask = context.allocate_with(
                        TensorDesc::new(DType::U8, vec![batch, 196])?,
                        MemoryPlacement::HostVisible,
                    )?;
                    let output = self
                        .record_descriptors(&mut graph, &pixels, &mask)
                        .map_err(|error| hrx::Error::Message(error.to_string()))?;
                    (vec![pixels, mask], output)
                } else {
                    let output = self
                        .record(&mut graph, &pixels)
                        .map_err(|error| hrx::Error::Message(error.to_string()))?;
                    (vec![pixels], output)
                };
                Ok(InferenceGraph {
                    inputs,
                    outputs: vec![output],
                    graph: graph.prepare()?,
                })
            })
        })?)
    }

    fn fragment(&self, b: usize) -> Result<ModelFragment> {
        ensure!(
            (1..=self.options.max_batch).contains(&b),
            "invalid batch size"
        );
        let p = b * 196;
        let r = b * TOKENS;
        let image = self.a[0].slice(0, p * 768 * 4)?;
        let patched = self.a[1].slice(0, p * M::HIDDEN * 4)?;
        let embedded = self.a[2].slice(0, r * M::HIDDEN * if M::RESIDUAL_F32 { 4 } else { 2 })?;
        let commands = [
            Command::Dispatch(Dispatch::indices(
                self.kernels[0],
                [p as u32],
                [(M::HIDDEN / 64) as u32, p.div_ceil(64) as u32, 1],
                vec![
                    image.read(),
                    self.weights["patch_w"].read(),
                    self.weights["patch_b"].read(),
                    patched.write(),
                ],
            )),
            Command::Dispatch(Dispatch::indices(
                self.kernels[1],
                [r as u32],
                [r as u32, 1, 1],
                vec![
                    patched.read(),
                    self.weights["prefix"].read(),
                    embedded.write(),
                ],
            )),
        ];
        // Fixed 224x224 patch geometry; embedding kernels initialize every output.
        unsafe {
            Ok(self
                .engine
                .fragment(
                    &commands,
                    &[(image, TensorDesc::new(DType::F32, vec![b, 196, 768])?)],
                    &[(
                        embedded,
                        TensorDesc::new(Encoder::<M>::input_dtype(), vec![b, TOKENS, M::HIDDEN])?,
                    )],
                )?
                .reuse_private_scratch())
        }
    }
}
#[cfg(test)]
fn patchify(input: &[f32], out: &mut [f32]) {
    let mut i = 0;
    for image in input.as_chunks::<IMAGE_ELEMENTS>().0 {
        for y in 0..14 {
            for x in 0..14 {
                for c in 0..3 {
                    for dy in 0..16 {
                        let start = c * 224 * 224 + (y * 16 + dy) * 224 + x * 16;
                        out[i..i + 16].copy_from_slice(&image[start..start + 16]);
                        i += 16;
                    }
                }
            }
        }
    }
}
fn embedding_specifications<M: ModelSpec>() -> Vec<(&'static str, Specialization)> {
    let mut s = vec![];
    macro_rules! spec { ($file:literal,$ns:literal,[$($k:literal => $v:expr),*]) => {{
        let mut spec=Specialization::new(concat!("dinov3_",$ns));
        $(spec.set_config(concat!("dinov3.",$ns,".",$k),$v.to_string());)*
        s.push((include_str!(concat!("../kernels/",$file,".loom")),spec));
    }}; }
    spec!("matmul_bias_f16_wmma","matmul_bias_f16_wmma",["k_size"=>768,"n_size"=>M::HIDDEN]);
    if M::RESIDUAL_F32 {
        spec!("embed_scatter_residual_f32","embed_scatter_residual_f32",["hidden_size"=>M::HIDDEN,"tokens_per_image"=>201,"prefix"=>5]);
    } else {
        spec!("embed_scatter_f32","embed_scatter_f32",["hidden_size"=>M::HIDDEN,"tokens_per_image"=>201,"prefix"=>5]);
    }
    s
}
#[cfg(test)]
fn specifications<M: ModelSpec>() -> Vec<(&'static str, Specialization)> {
    let mut s = embedding_specifications::<M>();
    s.extend(encoder::specifications::<M>(EncoderOptions {
        prefix_tokens: 5,
        max_batch: 64,
        ..Default::default()
    }));
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn buffer_extents_preserve_patch_width_and_qkv_padding() {
        for batch in [1, 3, 64] {
            let small = encoder::buffer_sizes::<ViTS16Plus>(batch, TOKENS).unwrap();
            let base = encoder::buffer_sizes::<ViTB16>(batch, TOKENS).unwrap();
            for i in [0, 1, 2, 3, 4, 5, 6] {
                assert_eq!(base[i], 2 * small[i]);
            }
            assert_eq!(base[2] - batch * TOKENS * 2304 * 2, 16 * 2304 * 2);
            assert_eq!(base[6], 4 * TOKENS * 768 * 4);
        }
    }
    #[test]
    fn patchification_preserves_channel_and_spatial_order() {
        let input = (0..IMAGE_ELEMENTS).map(|i| i as f32).collect::<Vec<_>>();
        let mut output = vec![0.; input.len()];
        patchify(&input, &mut output);
        for gy in 0..14 {
            for gx in 0..14 {
                for c in 0..3 {
                    for dy in 0..16 {
                        for dx in 0..16 {
                            let dst = (((gy * 14 + gx) * 3 + c) * 16 + dy) * 16 + dx;
                            assert_eq!(
                                output[dst],
                                input[(c * 224 + gy * 16 + dy) * 224 + gx * 16 + dx]
                            );
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod kernel_tests;
