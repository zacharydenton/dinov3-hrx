//! DINOv3 ViT-S+/16 at 224×224. Inputs are normalized RGB NCHW; outputs
//! are 201 tokens of 384 float32 features per image. A model owns a GPU stream.
pub mod hub;
mod weights;
use anyhow::{Result, ensure};
use hrx::loom::{
    Specialization,
    model::{Command, Dispatch, KernelId, ModelSession, Region},
};
use std::{collections::HashMap, path::Path};

pub use hrx::loom::model::{Distribution, ForwardTimings};

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
/// Resident model. Inference requires exclusive access; separate models can run independently.
pub struct DINOv3 {
    engine: ModelSession,
    options: Options,
    weights: HashMap<String, Region>,
    a: Vec<Region>,
    kernels: Vec<KernelId>,
    patched: Vec<f32>,
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
        ensure!(
            (1..=64).contains(&options.max_batch),
            "max_batch must be 1..=64"
        );
        let packed = weights::load(path.as_ref())?;
        let mut engine = ModelSession::open_for(options.device, "gfx1151")?;
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
        engine.reserve_readback(options.max_batch * TOKENS * HIDDEN * 4)?;
        Ok(Self {
            engine,
            options,
            weights,
            a,
            kernels,
            patched: vec![0.; options.max_batch * IMAGE_ELEMENTS],
        })
    }
    /// Infer an arbitrary batch of normalized NCHW RGB images.
    pub fn forward(&mut self, pixels: &[f32]) -> Result<Vec<f32>> {
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
            patchify(input, &mut self.patched[..input.len()]);
            self.prepare(b)?;
            self.engine.upload(
                self.a[7],
                bytemuck::cast_slice(&self.patched[..input.len()]),
            )?;
            self.engine.replay(b)?;
            self.engine
                .read(self.a[5], bytemuck::cast_slice_mut(output))?;
        }
        Ok(output)
    }
    /// Compare resident graph replay and direct dispatch after staging one batch.
    pub fn benchmark(&mut self, pixels: &[f32], samples: usize) -> Result<ForwardTimings> {
        ensure!(
            !pixels.is_empty() && pixels.len() <= self.options.max_batch * IMAGE_ELEMENTS,
            "benchmark requires one nonempty batch"
        );
        self.forward(pixels)?;
        Ok(self
            .engine
            .benchmark(pixels.len() / IMAGE_ELEMENTS, samples)?)
    }
    /// Run inference and return the CLS token from each image.
    pub fn cls(&mut self, pixels: &[f32]) -> Result<Vec<[f32; HIDDEN]>> {
        Ok(self
            .forward(pixels)?
            .chunks(TOKENS * HIDDEN)
            .map(|x| x[..HIDDEN].try_into().unwrap())
            .collect())
    }
    /// Run inference and average the 196 patch tokens for each image.
    pub fn patch_mean(&mut self, pixels: &[f32]) -> Result<Vec<[f32; HIDDEN]>> {
        Ok(self
            .forward(pixels)?
            .chunks(TOKENS * HIDDEN)
            .map(|x| {
                let mut out = [0.; HIDDEN];
                for row in x[5 * HIDDEN..].chunks(HIDDEN) {
                    for (a, b) in out.iter_mut().zip(row) {
                        *a += b / 196.;
                    }
                }
                out
            })
            .collect())
    }
    fn prepare(&mut self, b: usize) -> Result<()> {
        if self.engine.is_recorded(b) {
            return Ok(());
        }
        let r = (b * 201) as u32;
        let p = (b * 196) as u32;
        let [x, h, q, attn, act, out, patched, image, partials]: [Region; 9] =
            self.a.clone().try_into().unwrap();
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
        unsafe { self.engine.record(b, &commands)? };
        Ok(())
    }
}
fn patchify(input: &[f32], out: &mut [f32]) {
    let mut i = 0;
    for image in input.chunks_exact(IMAGE_ELEMENTS) {
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
        $(spec.config.insert(concat!("dinov3.",$ns,".",$k).into(),$v.to_string());)*
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
