//! DINOv3 ViT-S+/16 at 224×224. Inputs are normalized RGB NCHW; outputs
//! are 201 tokens of 384 float32 features per image, or compact normalized descriptors.
pub mod hub;
mod pooling;
mod preprocess;
mod weights;
use anyhow::{Result, ensure};
use hrx::{
    execution::{Graph, MemoryPlacement},
    image::ImageOps,
    inference::{Inference, InferenceGraph, ModelContext, PreparedModel},
    loom::Specialization,
    model::{Command, Dispatch, KernelId, ModelDefinition, ModelFragment, ModelSession, Region},
    plan_cache::PlanCache,
    tensor::{DType, DeviceTensor, Layout, TensorDesc},
};
use std::{collections::HashMap, path::Path};

pub const IMAGE_ELEMENTS: usize = 3 * 224 * 224;
pub const TOKENS: usize = 201;
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
pub struct DINOv3 {
    engine: ModelDefinition,
    plans: PlanCache<(usize, bool, bool), PreparedModel>,
    pooling: PlanCache<usize, PreparedModel>,
    summaries: PlanCache<(usize, bool), PreparedModel>,
    images: ImageOps,
    options: Options,
    weights: HashMap<String, Region>,
    a: Vec<Region>,
    kernels: Vec<KernelId>,
}
impl DINOv3 {
    /// Load the pinned pretrained model from the Hugging Face cache, fetching it
    /// if needed. Set `HF_HUB_OFFLINE=1` for cached weights only.
    /// Use [`Self::load`] to supply a local file instead.
    pub fn from_pretrained(options: Options) -> Result<Self> {
        ensure!(
            (1..=64).contains(&options.max_batch),
            "max_batch must be 1..=64"
        );
        Self::load(hub::weights(false)?, options)
    }

    /// Validate and pack the model, compile kernels, and allocate resident storage.
    pub fn load(path: impl AsRef<Path>, options: Options) -> Result<Self> {
        let context = ModelContext::new(hrx::execution::RuntimeOptions {
            gpu_index: options.device,
            ..Default::default()
        })?;
        Self::load_in(path, &context, options.max_batch)
    }
    /// Load into the caller's shared allocation, compiler and scheduling domain.
    pub fn load_in(
        path: impl AsRef<Path>,
        context: &ModelContext,
        max_batch: usize,
    ) -> Result<Self> {
        ensure!((1..=64).contains(&max_batch), "max_batch must be 1..=64");
        let packed = weights::load(path.as_ref())?;
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
        for (name, bytes) in packed {
            weights.insert(name, engine.weight(&bytes)?);
        }
        let r = options.max_batch * 201;
        let p = options.max_batch * 196;
        // Residual x, h, QKV, attention and SwiGLU are f16.
        // Patch embeddings, final output, images and split-K partials are f32.
        let sizes = [
            r * 384 * 2,
            r * 384 * 2,
            (r + 16) * 1152 * 2,
            r * 384 * 2,
            r * 1536 * 2,
            r * 384 * 4,
            p * 384 * 4,
            p * 768 * 4,
            4 * 201 * 384 * 4,
        ];
        let a = sizes
            .into_iter()
            .map(|n| engine.allocate(n))
            .collect::<std::result::Result<_, _>>()?;
        // Every source is embedded in this crate and its bindings are declared below.
        let kernels = unsafe { engine.compile(&specifications())? };
        let engine = engine.freeze(context)?;
        Ok(Self {
            engine,
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
    /// Pool resident tokens to normalized CLS and masked patch-mean descriptors.
    /// Mask is U8 `[batch,196]`, nonzero selects a patch. Empty masks produce a
    /// zero mean descriptor. Only `[batch,2,384]` descriptors need downloading.
    pub fn pool_descriptors(
        &self,
        tokens: &DeviceTensor,
        mask: &DeviceTensor,
    ) -> Result<Inference> {
        let shape = tokens.desc().shape();
        ensure!(
            shape.len() == 3
                && shape[1..] == [TOKENS, HIDDEN]
                && (1..=self.options.max_batch).contains(&shape[0]),
            "invalid token shape"
        );
        let plan = self.pooling.get_or_prepare(shape[0], || {
            pooling::fragment(self.context(), shape[0])?.prepare(3)
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
        let mut output = vec![0f32; batch * 2 * HIDDEN];
        for ((input, mask), out) in pixels
            .chunks(self.options.max_batch * IMAGE_ELEMENTS)
            .zip(masks.chunks(self.options.max_batch * 196))
            .zip(output.chunks_mut(self.options.max_batch * 2 * HIDDEN))
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
    /// `[batch,2,384]` descriptors cross back to the host. Masks are U8 `[batch,196]`.
    pub fn describe_rgb(&self, rgb: &[u8], masks: &[u8]) -> Result<Vec<f32>> {
        ensure!(
            rgb.len().is_multiple_of(IMAGE_ELEMENTS),
            "expected complete 224×224 RGB images"
        );
        let batch = rgb.len() / IMAGE_ELEMENTS;
        ensure!(masks.len() == batch * 196, "mask count mismatch");
        let mut output = vec![0f32; batch * 2 * HIDDEN];
        for ((input, mask), out) in rgb
            .chunks(self.options.max_batch * IMAGE_ELEMENTS)
            .zip(masks.chunks(self.options.max_batch * 196))
            .zip(output.chunks_mut(self.options.max_batch * 2 * HIDDEN))
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
        let mut output = vec![0.; batch * TOKENS * HIDDEN];
        for (input, output) in pixels
            .chunks(self.options.max_batch * IMAGE_ELEMENTS)
            .zip(output.chunks_mut(self.options.max_batch * TOKENS * HIDDEN))
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
    pub fn cls(&self, pixels: &[f32]) -> Result<Vec<[f32; HIDDEN]>> {
        self.raw_summary(pixels, false)
    }
    /// Run inference and average the 196 patch tokens for each image.
    pub fn patch_mean(&self, pixels: &[f32]) -> Result<Vec<[f32; HIDDEN]>> {
        self.raw_summary(pixels, true)
    }
    fn raw_summary(&self, pixels: &[f32], mean: bool) -> Result<Vec<[f32; HIDDEN]>> {
        ensure!(
            pixels.len().is_multiple_of(IMAGE_ELEMENTS),
            "input must contain complete 3×224×224 images"
        );
        ensure!(pixels.iter().all(|x| x.is_finite()), "input must be finite");
        let mut output = vec![[0.; HIDDEN]; pixels.len() / IMAGE_ELEMENTS];
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
                    let output = pooling::raw_fragment(context, batch, mean)?
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
                .read_into(&mut [bytemuck::cast_slice_mut(output.as_flattened_mut())])?;
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
        Ok(model.record(graph, &patches)?.remove(0))
    }

    /// Record normalization (for U8 NHWC RGB), patchification, transformer and
    /// masked descriptor pooling in one graph. F32 NCHW inputs are already
    /// normalized. Only the final `[batch,2,384]` descriptors need be downloaded.
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
            self.fragment(batch)?.record(graph, &patches)?.remove(0)
        } else {
            self.record(graph, pixels)?
        };
        Ok(pooling::fragment(self.context(), batch)?
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
        let r = (b * 201) as u32;
        let p = (b * 196) as u32;
        let sizes = [
            b * 201 * 384 * 2,
            b * 201 * 384 * 2,
            (b * 201 + 16) * 1152 * 2,
            b * 201 * 384 * 2,
            b * 201 * 1536 * 2,
            b * 201 * 384 * 4,
            b * 196 * 384 * 4,
            b * 196 * 768 * 4,
            4 * 201 * 384 * 4,
        ];
        let shaped = self
            .a
            .iter()
            .zip(sizes)
            .map(|(&region, bytes)| region.slice(0, bytes))
            .collect::<hrx::Result<Vec<_>>>()?;
        let [x, h, q, attn, act, out, patched, image, partials]: [Region; 9] =
            shaped.try_into().unwrap();
        let w = |s: &str| self.weights[s];
        let mut commands = vec![Command::Fill {
            region: q.slice(r as usize * 1152 * 2, 16 * 1152 * 2)?,
            value: 0,
        }];
        let mut add = |kernel: usize,
                       scalar: u32,
                       grid: [u32; 3],
                       bindings: Vec<Region>,
                       output: Region,
                       reads_output: bool| {
            let bindings = bindings
                .into_iter()
                .map(|region| {
                    if region == output {
                        if reads_output {
                            region.read_write()
                        } else {
                            region.write()
                        }
                    } else {
                        region.read()
                    }
                })
                .collect();
            commands.push(Command::Dispatch(Dispatch::indices(
                self.kernels[kernel],
                [scalar],
                grid,
                bindings,
            )));
        };
        add(
            0,
            p,
            [6, p.div_ceil(64), 1],
            vec![image, w("patch_w"), w("patch_b"), patched],
            patched,
            false,
        );
        add(1, r, [r, 1, 1], vec![patched, w("prefix"), x], x, false);
        add(
            2,
            r,
            [r.div_ceil(8), 1, 1],
            vec![x, w("l0_norm1_w"), w("l0_norm1_b"), h],
            h,
            false,
        );
        for i in 0..12 {
            let s = format!("l{i}_");
            let wt = |suffix: &str| w(&(s.clone() + suffix));
            add(
                4,
                r,
                [18, r.div_ceil(64), 1],
                vec![h, wt("qkv_w"), wt("qkv_b"), q, w("rope_cos"), w("rope_sin")],
                q,
                false,
            );
            let k = q.slice(768, q.len() - 768)?;
            let v = q.slice(1536, q.len() - 1536)?;
            add(
                5,
                r,
                [(b * 13) as u32, 6, 1],
                vec![q, k, v, attn],
                attn,
                false,
            );
            add(
                6,
                r,
                [6, r.div_ceil(64), 1],
                vec![attn, wt("o_w"), wt("o_b"), x, wt("ls1")],
                x,
                true,
            );
            add(
                2,
                r,
                [r.div_ceil(8), 1, 1],
                vec![x, wt("norm2_w"), wt("norm2_b"), h],
                h,
                false,
            );
            add(
                8,
                r,
                [24, r.div_ceil(64), 1],
                vec![h, wt("gateup_w"), wt("gateup_b"), act],
                act,
                false,
            );
            if b == 1 {
                add(
                    9,
                    r,
                    [6, r.div_ceil(64), 4],
                    vec![act, wt("down_w"), wt("down_b"), partials],
                    partials,
                    false,
                );
                add(
                    10,
                    r,
                    [r, 1, 1],
                    vec![partials, wt("down_b"), x, wt("ls2")],
                    x,
                    true,
                );
            } else {
                add(
                    7,
                    r,
                    [6, r.div_ceil(64), 1],
                    vec![act, wt("down_w"), wt("down_b"), x, wt("ls2")],
                    x,
                    true,
                );
            }
            if i < 11 {
                add(
                    2,
                    r,
                    [r.div_ceil(8), 1, 1],
                    vec![
                        x,
                        w(&format!("l{}_norm1_w", i + 1)),
                        w(&format!("l{}_norm1_b", i + 1)),
                        h,
                    ],
                    h,
                    false,
                );
            }
        }
        add(
            3,
            r,
            [r.div_ceil(8), 1, 1],
            vec![x, w("norm_w"), w("norm_b"), out],
            out,
            false,
        );
        // The embedded kernels obey the scalar, region and access contracts above.
        unsafe {
            Ok(self.engine.fragment(
                &commands,
                &[(image, TensorDesc::new(DType::F32, vec![b, 196, 768])?)],
                &[(out, TensorDesc::new(DType::F32, vec![b, TOKENS, HIDDEN])?)],
            )?)
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
fn specifications() -> Vec<(&'static str, Specialization)> {
    let mut s = vec![];
    macro_rules! spec { ($file:literal,$ns:literal,[$($k:literal => $v:expr),*]) => {{
        let mut spec=Specialization::new(concat!("dinov3_",$ns));
        $(spec.set_config(concat!("dinov3.",$ns,".",$k),$v.to_string());)*
        s.push((include_str!(concat!("../kernels/",$file,".loom")),spec));
    }}; }
    spec!("matmul_bias_f16_wmma","matmul_bias_f16_wmma",["k_size"=>768,"n_size"=>384]);
    spec!("embed_scatter_f32","embed_scatter_f32",["hidden_size"=>384,"tokens_per_image"=>201,"prefix"=>5]);
    spec!("layernorm_rowwave_f16","layernorm_rowwave_f16",["hidden_size"=>384,"epsilon"=>"1e-5"]);
    spec!("layernorm_rowwave_f32out","layernorm_rowwave_f32out",["hidden_size"=>384,"epsilon"=>"1e-5"]);
    spec!("matmul_qkv_rope_f16_wmma","matmul_qkv_rope_f16_wmma",["k_size"=>384,"n_size"=>1152,"head_dim"=>64,"prefix"=>5,"tokens_per_image"=>201,"rope_channels"=>768]);
    spec!("attention_online_f16_wmma_cf16","attention_online_f16_wmma_cf16",["hidden_size"=>384,"qkv_stride"=>1152,"tokens_per_image"=>201,"scale"=>0.125,"max_images"=>64,"token_capacity"=>262144]);
    spec!("matmul_resid_f16_wmma","matmul_resid_f16_wmma",["k_size"=>384,"n_size"=>384]);
    spec!("matmul_resid_f16_wmma","matmul_resid_f16_wmma",["k_size"=>1536,"n_size"=>384]);
    spec!("matmul_swiglu_f16_wmma","matmul_swiglu_f16_wmma",["k_size"=>384,"n_size"=>1536]);
    spec!("matmul_splitk_f16_wmma","matmul_splitk_f16_wmma",["k_size"=>1536,"n_size"=>384,"splits"=>4]);
    spec!("splitk_reduce_f16","splitk_reduce_f16",["n_size"=>384,"splits"=>4]);
    s
}

#[cfg(test)]
mod tests {
    use super::*;
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
