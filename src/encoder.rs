//! A transformer encoder independent of image embedding and checkpoint metadata.
use crate::EncoderSpec;
use anyhow::{Result, ensure};
use half::f16;
use hrx::{
    execution::Graph,
    inference::ModelContext,
    loom::Specialization,
    model::{Command, Dispatch, KernelId, ModelDefinition, ModelFragment, ModelSession, Region},
    tensor::{DType, DeviceTensor, TensorDesc},
};
use std::{collections::HashMap, marker::PhantomData};

/// Sequence geometry and capacity, fixed when the encoder is constructed.
#[derive(Clone, Copy, Debug)]
pub struct EncoderOptions {
    /// Tokens in each independent, bidirectional attention sequence (1..=1024).
    pub tokens_per_sequence: usize,
    /// Leading tokens excluded from RoPE (0..=64, less than the sequence length).
    pub prefix_tokens: usize,
    /// Maximum batch (1..=1024, matching the attention kernel's sequence limit).
    pub max_batch: usize,
    /// Maximum activation workspace per graph, excluding weights and caller input.
    /// Defaults to 2 GiB. Includes residuals, QKV padding, FFN, output, and scratch.
    pub max_workspace_bytes: usize,
    /// Positive, finite LayerNorm epsilon.
    pub epsilon: f32,
}
impl Default for EncoderOptions {
    fn default() -> Self {
        Self {
            tokens_per_sequence: 201,
            prefix_tokens: 0,
            max_batch: 32,
            max_workspace_bytes: 2 * 1024 * 1024 * 1024,
            epsilon: 1e-5,
        }
    }
}

/// Affine LayerNorm parameters, each containing HIDDEN F32 values.
pub struct NormWeights {
    pub weight: Vec<f32>,
    pub bias: Vec<f32>,
}
/// Row-major `[output, input]` matrix and its F32 bias.
pub struct LinearWeights {
    pub weight: Vec<f16>,
    pub bias: Vec<f32>,
}
/// One pre-normalized attention/FFN block, including both LayerScale vectors.
pub struct EncoderLayerWeights {
    pub norm1: NormWeights,
    /// Concatenated Q, K, V rows: `[3*HIDDEN,HIDDEN]`, with a full packed bias.
    /// Q/V biases must be zero when `EncoderSpec::QV_BIAS` is false.
    pub qkv: LinearWeights,
    pub attention_output: LinearWeights,
    /// HIDDEN values; use ones to disable attention LayerScale.
    pub attention_scale: Vec<f32>,
    pub norm2: NormWeights,
    /// GELU: `[INTERMEDIATE,HIDDEN]`; SwiGLU: gate rows followed by up rows.
    pub mlp_up: LinearWeights,
    pub mlp_down: LinearWeights,
    /// HIDDEN values; use ones to disable FFN LayerScale.
    pub mlp_scale: Vec<f32>,
}
/// Split-half rotary tables, shared by every head and sequence in the batch.
/// Each table has `[tokens_per_sequence-prefix_tokens, HEAD_DIM]` F32 values.
pub struct RotaryEmbedding {
    pub cos: Vec<f32>,
    pub sin: Vec<f32>,
}
/// Host weights for an encoder, including its final LayerNorm.
pub struct EncoderWeights {
    pub layers: Vec<EncoderLayerWeights>,
    pub norm: NormWeights,
    /// `None` leaves Q/K positions unchanged (e.g. positions already added to input).
    pub rotary: Option<RotaryEmbedding>,
}

/// Resident encoder weights and code. Recording adds GPU operations to the caller's
/// graph; it does not submit inference or create a private inference pool.
///
/// Kernels support HIDDEN multiples of 128 in 128..=4096, head widths 64/128,
/// INTERMEDIATE multiples of 256 in 256..=8192, and 1..=256 layers on gfx1151.
/// Attention is bidirectional within each sequence; there is no causal/padding mask.
///
/// ```no_run
/// use dinov3_hrx::{Encoder, EncoderOptions, EncoderSpec, EncoderWeights};
/// use hrx::{execution::Graph, inference::ModelContext, tensor::DeviceTensor};
/// struct Custom;
/// impl EncoderSpec for Custom {
///     const HIDDEN: usize = 512;
///     const HEADS: usize = 8;
///     const LAYERS: usize = 6;
///     const INTERMEDIATE: usize = 2048;
///     const GATED: bool = false;
/// }
/// fn record(context: &ModelContext, graph: &mut Graph, embedded: &DeviceTensor,
///           weights: EncoderWeights) -> anyhow::Result<DeviceTensor> {
///     let encoder = Encoder::<Custom>::new(context, EncoderOptions {
///         tokens_per_sequence: 257, max_batch: 16, ..Default::default()
///     }, weights)?;
///     // `embedded` is contiguous F16 [batch,257,512]. The graph retains the
///     // encoder's weights/code; the returned F32 tensor can feed another node.
///     encoder.record(graph, embedded)
/// }
/// ```
pub struct Encoder<S: EncoderSpec> {
    marker: PhantomData<S>,
    engine: ModelDefinition,
    options: EncoderOptions,
    weights: HashMap<String, Region>,
    a: Vec<Region>,
    kernels: Vec<KernelId>,
}
impl<S: EncoderSpec> Encoder<S> {
    /// Validate and upload explicit weights into a shared HRX context.
    pub fn new(
        context: &ModelContext,
        options: EncoderOptions,
        weights: EncoderWeights,
    ) -> Result<Self> {
        validate::<S>(options)?;
        let packed = weights.pack::<S>(options)?;
        Self::from_packed(packed, context, options)
    }

    pub(crate) fn from_packed(
        packed: HashMap<String, Vec<u8>>,
        context: &ModelContext,
        options: EncoderOptions,
    ) -> Result<Self> {
        validate::<S>(options)?;
        ensure!(
            context.runtime().gpu()?.target().as_str() == "gfx1151",
            "encoder requires gfx1151"
        );
        let mut model = ModelSession::in_context(context)?;
        let mut weights = HashMap::new();
        for (name, bytes) in packed {
            let k = if name.ends_with("_down_w") {
                Some(S::INTERMEDIATE)
            } else if ["_qkv_w", "_o_w", "_up_w", "_gateup_w"]
                .iter()
                .any(|suffix| name.ends_with(suffix))
            {
                Some(S::HIDDEN)
            } else {
                None
            };
            let bytes = if let Some(k) = k {
                pad_weight_rows(bytes, k, weight_stride::<S>(k))
            } else {
                bytes
            };
            weights.insert(name, model.weight(&bytes)?);
        }
        let mut a: Vec<Region> = Vec::new();
        for (index, bytes) in buffer_sizes::<S>(options.max_batch, options.tokens_per_sequence)?
            .into_iter()
            .enumerate()
        {
            // QKV consumes normalized activations before attention writes its
            // output. Projection consumes that output before the next norm.
            // Both regions have the same shape and never share a dispatch.
            a.push(if index == 3 {
                a[1]
            } else {
                model.allocate(bytes)?
            });
        }
        // Embedded kernels, validated dimensions, and matching binding contracts below.
        let kernels = unsafe { model.compile(&specifications::<S>(options))? };
        Ok(Self {
            marker: PhantomData,
            engine: model.freeze(context)?,
            options,
            weights,
            a,
            kernels,
        })
    }

    pub fn context(&self) -> &ModelContext {
        self.engine.context()
    }

    /// Residual-stream input representation; matrix activations/weights are F16.
    pub fn input_dtype() -> DType {
        if S::RESIDUAL_F32 {
            DType::F32
        } else {
            DType::F16
        }
    }

    /// Record LayerNorm → QKV/RoPE → attention → residual → FFN for every layer,
    /// followed by final LayerNorm. Input is contiguous `[batch,tokens,HIDDEN]`
    /// in [`Self::input_dtype`]; output has the same shape in F32.
    ///
    /// Input is preserved with a graph-recorded device copy. Recorded graphs own
    /// their scratch and retain weights/code independently of this encoder. As in
    /// HRX fragment recording, existing input initialization is waited at preparation.
    pub fn record(&self, graph: &mut Graph, tokens: &DeviceTensor) -> Result<DeviceTensor> {
        let fragment = self.input_fragment(tokens)?;
        tokens.completion().wait()?;
        let residual = self.context().allocate(tokens.desc().clone())?;
        graph.copy(residual.binding().unwrap(), tokens.binding().unwrap())?;
        Ok(fragment.record(graph, &[residual])?.remove(0))
    }

    // DINO owns this fresh embedding output and has no other consumers. Reuse
    // it as the residual stream, preserving the existing copy-free image path.
    pub(crate) fn record_owned(
        &self,
        graph: &mut Graph,
        tokens: DeviceTensor,
    ) -> Result<DeviceTensor> {
        Ok(self
            .input_fragment(&tokens)?
            .record(graph, &[tokens])?
            .remove(0))
    }

    fn input_fragment(&self, tokens: &DeviceTensor) -> Result<ModelFragment> {
        self.context().validate(tokens)?;
        let shape = tokens.desc().shape();
        ensure!(
            shape.len() == 3
                && shape[1..] == [self.options.tokens_per_sequence, S::HIDDEN]
                && (1..=self.options.max_batch).contains(&shape[0]),
            "invalid encoder token shape"
        );
        ensure!(
            tokens.desc() == &TensorDesc::new(Self::input_dtype(), shape.to_vec())?,
            "expected contiguous encoder tokens in {:?}",
            Self::input_dtype()
        );
        self.fragment(shape[0])
    }
}

fn validate<S: EncoderSpec>(o: EncoderOptions) -> Result<()> {
    ensure!(
        (128..=4096).contains(&S::HIDDEN) && S::HIDDEN.is_multiple_of(128),
        "HIDDEN must be a multiple of 128 in 128..=4096"
    );
    ensure!(
        S::HEADS > 0 && S::HIDDEN.is_multiple_of(S::HEADS),
        "HEADS must divide HIDDEN"
    );
    ensure!(
        [64, 128].contains(&S::HEAD_DIM) && S::HEAD_DIM == S::HIDDEN / S::HEADS,
        "HEAD_DIM must be 64 or 128 and match HIDDEN/HEADS"
    );
    ensure!(
        (256..=8192).contains(&S::INTERMEDIATE) && S::INTERMEDIATE.is_multiple_of(256),
        "INTERMEDIATE must be a multiple of 256 in 256..=8192"
    );
    ensure!((1..=256).contains(&S::LAYERS), "LAYERS must be 1..=256");
    ensure!(
        (1..=1024).contains(&o.tokens_per_sequence),
        "tokens_per_sequence must be 1..=1024"
    );
    ensure!(
        o.prefix_tokens <= 64 && o.prefix_tokens < o.tokens_per_sequence,
        "invalid prefix_tokens"
    );
    ensure!(
        (1..=1024).contains(&o.max_batch),
        "max_batch must be 1..=1024"
    );
    let rows = o
        .max_batch
        .checked_mul(o.tokens_per_sequence)
        .ok_or_else(|| anyhow::anyhow!("encoder row count overflow"))?;
    let capacity = rows
        .checked_add(16)
        .and_then(|r| r.checked_next_multiple_of(16))
        .ok_or_else(|| anyhow::anyhow!("attention token capacity overflow"))?;
    ensure!(
        capacity <= 1_048_576,
        "attention token capacity {capacity} exceeds kernel limit 1048576 (including padding)"
    );
    let sizes = buffer_sizes::<S>(o.max_batch, o.tokens_per_sequence)?;
    let workspace = sizes
        .iter()
        .enumerate()
        .filter(|(index, _)| *index != 3) // Attention aliases normalized activations.
        .try_fold(0usize, |total, (_, &bytes)| total.checked_add(bytes))
        .ok_or_else(|| anyhow::anyhow!("encoder workspace byte count overflow"))?;
    ensure!(
        workspace <= o.max_workspace_bytes,
        "encoder workspace requires {workspace} bytes, exceeding max_workspace_bytes {}",
        o.max_workspace_bytes
    );
    // Loom's scalar indices are 32-bit. Keep each region inside their byte-address
    // range even when callers raise the aggregate workspace budget.
    ensure!(
        sizes.iter().all(|&bytes| bytes <= u32::MAX as usize),
        "encoder workspace region exceeds the 32-bit kernel address range"
    );
    ensure!(
        o.epsilon.is_finite() && o.epsilon > 0.,
        "epsilon must be finite and positive"
    );
    Ok(())
}

impl EncoderWeights {
    fn pack<S: EncoderSpec>(self, o: EncoderOptions) -> Result<HashMap<String, Vec<u8>>> {
        let mut p = HashMap::new();
        fn floats(
            p: &mut HashMap<String, Vec<u8>>,
            name: String,
            v: Vec<f32>,
            n: usize,
        ) -> Result<()> {
            ensure!(
                v.len() == n && v.iter().all(|v| v.is_finite()),
                "invalid {name}: expected {n} finite F32 values"
            );
            p.insert(name, bytemuck::cast_slice(&v).to_vec());
            Ok(())
        }
        fn norm(
            p: &mut HashMap<String, Vec<u8>>,
            name: &str,
            v: NormWeights,
            h: usize,
        ) -> Result<()> {
            floats(p, format!("{name}_w"), v.weight, h)?;
            floats(p, format!("{name}_b"), v.bias, h)
        }
        fn linear(
            p: &mut HashMap<String, Vec<u8>>,
            name: &str,
            v: LinearWeights,
            n: usize,
            k: usize,
        ) -> Result<()> {
            ensure!(
                v.weight.len() == n * k && v.weight.iter().all(|v| v.is_finite()),
                "invalid {name}: expected {} finite F16 weights",
                n * k
            );
            p.insert(
                format!("{name}_w"),
                bytemuck::cast_slice(&v.weight).to_vec(),
            );
            floats(p, format!("{name}_b"), v.bias, n)
        }
        ensure!(
            self.layers.len() == S::LAYERS,
            "expected {} encoder layers",
            S::LAYERS
        );
        for (i, l) in self.layers.into_iter().enumerate() {
            let h = S::HIDDEN;
            let f = S::INTERMEDIATE;
            ensure!(l.qkv.bias.len() == 3 * h, "invalid QKV bias length");
            ensure!(
                S::QV_BIAS
                    || l.qkv.bias[..h]
                        .iter()
                        .chain(&l.qkv.bias[2 * h..])
                        .all(|&x| x == 0.),
                "Q/V biases must be zero when EncoderSpec::QV_BIAS is false"
            );
            norm(&mut p, &format!("l{i}_norm1"), l.norm1, h)?;
            norm(&mut p, &format!("l{i}_norm2"), l.norm2, h)?;
            linear(&mut p, &format!("l{i}_qkv"), l.qkv, 3 * h, h)?;
            linear(&mut p, &format!("l{i}_o"), l.attention_output, h, h)?;
            linear(
                &mut p,
                &format!("l{i}_{}", if S::GATED { "gateup" } else { "up" }),
                l.mlp_up,
                f * if S::GATED { 2 } else { 1 },
                h,
            )?;
            linear(&mut p, &format!("l{i}_down"), l.mlp_down, h, f)?;
            floats(&mut p, format!("l{i}_ls1"), l.attention_scale, h)?;
            floats(&mut p, format!("l{i}_ls2"), l.mlp_scale, h)?;
        }
        norm(&mut p, "norm", self.norm, S::HIDDEN)?;
        let count = (o.tokens_per_sequence - o.prefix_tokens) * S::HEAD_DIM;
        let rotary = self.rotary.unwrap_or_else(|| RotaryEmbedding {
            cos: vec![1.; count],
            sin: vec![0.; count],
        });
        floats(&mut p, "rope_cos".into(), rotary.cos, count)?;
        floats(&mut p, "rope_sin".into(), rotary.sin, count)?;
        Ok(p)
    }
}

impl<S: EncoderSpec> Encoder<S> {
    fn fragment(&self, b: usize) -> Result<ModelFragment> {
        ensure!(
            (1..=self.options.max_batch).contains(&b),
            "invalid batch size"
        );
        let r = (b * self.options.tokens_per_sequence) as u32;
        let wide = S::RESIDUAL_F32 && S::GATED;
        let (tile_rows, tile_cols) = projection_tile::<S>();
        let projection_rows = r.div_ceil(tile_rows);
        let sizes = buffer_sizes::<S>(b, self.options.tokens_per_sequence)?;
        let shaped = self
            .a
            .iter()
            .zip(sizes)
            .map(|(&region, bytes)| region.slice(0, bytes))
            .collect::<hrx::Result<Vec<_>>>()?;
        let [x, h, q, attn, act, out, partials, projected]: [Region; 8] =
            shaped.try_into().unwrap();
        let w = |s: &str| self.weights[s];
        let mut commands = vec![Command::Fill {
            region: q.slice(
                r as usize * qkv_stride::<S>() * 2,
                16 * qkv_stride::<S>() * 2,
            )?,
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
            r,
            [r.div_ceil(8), 1, 1],
            vec![x, w("l0_norm1_w"), w("l0_norm1_b"), h],
            h,
            false,
        );
        for i in 0..S::LAYERS {
            let s = format!("l{i}_");
            let wt = |suffix: &str| w(&(s.clone() + suffix));
            if !separate_qkv::<S>() {
                add(
                    2,
                    r,
                    [(3 * S::HIDDEN / 64) as u32, r.div_ceil(64), 1],
                    vec![h, wt("qkv_w"), wt("qkv_b"), q, w("rope_cos"), w("rope_sin")],
                    q,
                    false,
                );
            } else {
                // The wide kernel's fifth binding is LayerScale. Projection
                // specializations ignore it, so reuse their valid bias region.
                add(
                    2,
                    r,
                    [(3 * S::HIDDEN / tile_cols) as u32, projection_rows, 1],
                    if wide {
                        vec![h, wt("qkv_w"), wt("qkv_b"), projected, wt("qkv_b")]
                    } else {
                        vec![h, wt("qkv_w"), wt("qkv_b"), projected]
                    },
                    projected,
                    false,
                );
                add(
                    9,
                    r,
                    [(r as usize * 3 * S::HIDDEN).div_ceil(256) as u32, 1, 1],
                    vec![projected, w("rope_cos"), w("rope_sin"), q],
                    q,
                    false,
                );
            }
            let k = q.slice(S::HIDDEN * 2, q.len() - S::HIDDEN * 2)?;
            let v = q.slice(S::HIDDEN * 4, q.len() - S::HIDDEN * 4)?;
            add(
                3,
                r,
                [
                    (b * self.options.tokens_per_sequence.div_ceil(16)) as u32,
                    S::HEADS as u32,
                    1,
                ],
                vec![q, k, v, attn],
                attn,
                false,
            );
            add(
                4,
                r,
                [(S::HIDDEN / tile_cols) as u32, projection_rows, 1],
                vec![attn, wt("o_w"), wt("o_b"), x, wt("ls1")],
                x,
                true,
            );
            add(
                0,
                r,
                [r.div_ceil(8), 1, 1],
                vec![x, wt("norm2_w"), wt("norm2_b"), h],
                h,
                false,
            );
            if wide && S::GATED {
                add(
                    6,
                    r,
                    [(2 * S::INTERMEDIATE / tile_cols) as u32, projection_rows, 1],
                    vec![h, wt("gateup_w"), wt("gateup_b"), projected, wt("gateup_b")],
                    projected,
                    false,
                );
                add(
                    10,
                    r,
                    [(r as usize * S::INTERMEDIATE).div_ceil(1024) as u32, 1, 1],
                    vec![projected, act],
                    act,
                    false,
                );
            } else if wide {
                add(
                    6,
                    r,
                    [(S::INTERMEDIATE / tile_cols) as u32, projection_rows, 1],
                    vec![h, wt("up_w"), wt("up_b"), act, wt("up_b")],
                    act,
                    false,
                );
            } else {
                add(
                    6,
                    r,
                    [(S::INTERMEDIATE / 64) as u32, r.div_ceil(64), 1],
                    vec![
                        h,
                        wt(if S::GATED { "gateup_w" } else { "up_w" }),
                        wt(if S::GATED { "gateup_b" } else { "up_b" }),
                        act,
                    ],
                    act,
                    false,
                );
            }
            if b == 1 {
                add(
                    7,
                    r,
                    [(S::HIDDEN / tile_cols) as u32, projection_rows, 4],
                    if wide {
                        vec![act, wt("down_w"), wt("down_b"), partials, wt("ls2")]
                    } else {
                        vec![act, wt("down_w"), wt("down_b"), partials]
                    },
                    partials,
                    false,
                );
                add(
                    8,
                    r,
                    [r, 1, 1],
                    vec![partials, wt("down_b"), x, wt("ls2")],
                    x,
                    true,
                );
            } else {
                add(
                    5,
                    r,
                    [(S::HIDDEN / tile_cols) as u32, projection_rows, 1],
                    vec![act, wt("down_w"), wt("down_b"), x, wt("ls2")],
                    x,
                    true,
                );
            }
            if i + 1 < S::LAYERS {
                add(
                    0,
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
            1,
            r,
            [r.div_ceil(8), 1, 1],
            vec![x, w("norm_w"), w("norm_b"), out],
            out,
            false,
        );
        // The embedded kernels obey the scalar, region and access contracts above.
        // Every private activation is written before use; QKV padding is filled
        // above. Residual and split-K paths need no initial scratch contents.
        unsafe {
            Ok(self
                .engine
                .fragment(
                    &commands,
                    &[(
                        x,
                        TensorDesc::new(
                            Self::input_dtype(),
                            vec![b, self.options.tokens_per_sequence, S::HIDDEN],
                        )?,
                    )],
                    &[(
                        out,
                        TensorDesc::new(
                            DType::F32,
                            vec![b, self.options.tokens_per_sequence, S::HIDDEN],
                        )?,
                    )],
                )?
                .reuse_private_scratch())
        }
    }
}

pub(crate) fn buffer_sizes<M: EncoderSpec>(batch: usize, tokens: usize) -> Result<[usize; 8]> {
    fn product(values: &[usize]) -> Result<usize> {
        values
            .iter()
            .try_fold(1usize, |n, &v| n.checked_mul(v))
            .ok_or_else(|| anyhow::anyhow!("encoder workspace byte count overflow"))
    }
    let rows = product(&[batch, tokens])?;
    let padded = rows
        .checked_add(16)
        .ok_or_else(|| anyhow::anyhow!("encoder padding overflow"))?;
    let projected = if separate_qkv::<M>() {
        let channels = product(&[3, M::HIDDEN])?.max(if M::RESIDUAL_F32 && M::GATED {
            product(&[2, M::INTERMEDIATE])?
        } else {
            0
        });
        product(&[rows, channels, 4])?
    } else {
        4
    };
    Ok([
        product(&[rows, M::HIDDEN, if M::RESIDUAL_F32 { 4 } else { 2 }])?,
        product(&[rows, M::HIDDEN, 2])?,
        product(&[padded, qkv_stride::<M>(), 2])?,
        product(&[rows, M::HIDDEN, 2])?,
        product(&[rows, M::INTERMEDIATE, 2])?,
        product(&[rows, M::HIDDEN, 4])?,
        product(&[4, tokens, M::HIDDEN, 4])?,
        projected,
    ])
}
// Padding power-of-two rows avoids the memory-access conflicts measured on gfx1151.
pub(crate) fn qkv_stride<M: EncoderSpec>() -> usize {
    3 * M::HIDDEN
        + if M::HIDDEN >= 1024 && M::HIDDEN.is_power_of_two() {
            64
        } else {
            0
        }
}
fn weight_stride<M: EncoderSpec>(k: usize) -> usize {
    k + if M::RESIDUAL_F32 && k >= 1024 && k.is_power_of_two() {
        64
    } else {
        0
    }
}
fn pad_weight_rows(bytes: Vec<u8>, k: usize, stride: usize) -> Vec<u8> {
    if stride == k {
        return bytes;
    }
    let mut padded = vec![0; bytes.len() / (k * 2) * stride * 2];
    for (src, dst) in bytes
        .chunks_exact(k * 2)
        .zip(padded.chunks_exact_mut(stride * 2))
    {
        dst[..src.len()].copy_from_slice(src);
    }
    padded
}
fn projection_tile<M: EncoderSpec>() -> (u32, usize) {
    if M::RESIDUAL_F32 && M::GATED {
        (128, 128)
    } else {
        (64, 64)
    }
}
fn separate_qkv<M: EncoderSpec>() -> bool {
    (M::RESIDUAL_F32 && M::GATED) || M::HEAD_DIM == 128 || M::HIDDEN > 8192 / 3
}

pub(crate) fn specifications<M: EncoderSpec>(
    options: EncoderOptions,
) -> Vec<(&'static str, Specialization)> {
    let mut s = vec![];
    let wide = M::RESIDUAL_F32 && M::GATED;
    macro_rules! spec { ($file:literal,$ns:literal,[$($k:literal => $v:expr),*]) => {{
        let mut spec=Specialization::new(concat!("dinov3_",$ns));
        $(spec.set_config(concat!("dinov3.",$ns,".",$k),$v.to_string());)*
        s.push((include_str!(concat!("../kernels/",$file,".loom")),spec));
    }}; }
    // H+ keeps the released schedule: register prefetch regressed batch-16
    // throughput for its 1280/5120 dimensions in paired measurements.
    macro_rules! wide_spec { ([$($k:literal => $v:expr),*]) => {{
        if M::HIDDEN == 1280 && M::INTERMEDIATE == 5120 {
            spec!("matmul_wide_wmma","matmul_wide_wmma",["tile_rows"=>128,$($k=>$v),*]);
        } else {
            spec!("matmul_prefetch_wmma","matmul_wide_wmma",[$($k=>$v),*]);
        }
    }}; }
    if !M::RESIDUAL_F32 && M::HIDDEN <= 384 {
        spec!("layernorm_rowwave_f16","layernorm_rowwave_f16",["hidden_size"=>M::HIDDEN,"epsilon"=>options.epsilon]);
        spec!("layernorm_rowwave_f32out","layernorm_rowwave_f32out",["hidden_size"=>M::HIDDEN,"epsilon"=>options.epsilon]);
    } else if !M::RESIDUAL_F32 && M::HIDDEN == 768 {
        spec!("layernorm_rowwave_768_f16","layernorm_rowwave_768_f16",["hidden_size"=>M::HIDDEN,"epsilon"=>options.epsilon]);
        spec!("layernorm_rowwave_768_f32out","layernorm_rowwave_768_f32out",["hidden_size"=>M::HIDDEN,"epsilon"=>options.epsilon]);
    } else if M::RESIDUAL_F32 && M::HIDDEN >= 1024 {
        spec!("layernorm_rowwave_wide_f16","layernorm_rowwave_wide_f16",["hidden_size"=>M::HIDDEN,"epsilon"=>options.epsilon]);
        spec!("layernorm_rowwave_wide_f32out","layernorm_rowwave_wide_f32out",["hidden_size"=>M::HIDDEN,"epsilon"=>options.epsilon]);
    } else {
        spec!("layernorm_encoder","layernorm_encoder",["hidden_size"=>M::HIDDEN,"epsilon"=>options.epsilon,"input_f32"=>usize::from(M::RESIDUAL_F32),"output_f32"=>0]);
        spec!("layernorm_encoder","layernorm_encoder",["hidden_size"=>M::HIDDEN,"epsilon"=>options.epsilon,"input_f32"=>usize::from(M::RESIDUAL_F32),"output_f32"=>1]);
    }
    if wide {
        wide_spec!(["weight_stride"=>weight_stride::<M>(M::HIDDEN),"max_rows"=>options.max_batch*options.tokens_per_sequence,"k_size"=>M::HIDDEN,"n_size"=>3*M::HIDDEN,"splits"=>1,"epilogue"=>0]);
    } else if separate_qkv::<M>() {
        spec!("matmul_qkv_f32out","matmul_qkv_f32out",["weight_stride"=>weight_stride::<M>(M::HIDDEN),"k_size"=>M::HIDDEN,"n_size"=>3*M::HIDDEN]);
    } else {
        spec!("matmul_qkv_rope_f16_wmma","matmul_qkv_rope_f16_wmma",["weight_stride"=>weight_stride::<M>(M::HIDDEN),"k_size"=>M::HIDDEN,"n_size"=>3*M::HIDDEN,"head_dim"=>64,"prefix"=>options.prefix_tokens,"tokens_per_image"=>options.tokens_per_sequence,"rope_channels"=>2*M::HIDDEN,"output_stride"=>qkv_stride::<M>()]);
    }
    if M::HEAD_DIM == 64 {
        spec!("attention_online_f16_wmma_cf16","attention_online_f16_wmma_cf16",["hidden_size"=>M::HIDDEN,"qkv_stride"=>qkv_stride::<M>(),"tokens_per_image"=>options.tokens_per_sequence,"scale"=>0.125,"max_images"=>options.max_batch,"token_capacity"=>(options.max_batch*options.tokens_per_sequence+16).next_multiple_of(16)]);
    } else {
        spec!("attention_online_f16_wmma_h128","attention_online_f16_wmma_h128",["hidden_size"=>M::HIDDEN,"qkv_stride"=>qkv_stride::<M>(),"tokens_per_image"=>options.tokens_per_sequence,"scale"=>1.0_f64/128.0_f64.sqrt(),"max_images"=>options.max_batch,"token_capacity"=>(options.max_batch*options.tokens_per_sequence+16).next_multiple_of(16)]);
    }
    if wide {
        wide_spec!(["weight_stride"=>weight_stride::<M>(M::HIDDEN),"max_rows"=>options.max_batch*options.tokens_per_sequence,"k_size"=>M::HIDDEN,"n_size"=>M::HIDDEN,"splits"=>1,"epilogue"=>2]);
        wide_spec!(["weight_stride"=>weight_stride::<M>(M::INTERMEDIATE),"max_rows"=>options.max_batch*options.tokens_per_sequence,"k_size"=>M::INTERMEDIATE,"n_size"=>M::HIDDEN,"splits"=>1,"epilogue"=>2]);
    } else if M::RESIDUAL_F32 {
        spec!("matmul_resid_f32_wmma","matmul_resid_f32_wmma",["weight_stride"=>weight_stride::<M>(M::HIDDEN),"k_size"=>M::HIDDEN,"n_size"=>M::HIDDEN]);
        spec!("matmul_resid_f32_wmma","matmul_resid_f32_wmma",["weight_stride"=>weight_stride::<M>(M::INTERMEDIATE),"k_size"=>M::INTERMEDIATE,"n_size"=>M::HIDDEN]);
    } else {
        spec!("matmul_resid_f16_wmma","matmul_resid_f16_wmma",["k_size"=>M::HIDDEN,"n_size"=>M::HIDDEN]);
        spec!("matmul_resid_f16_wmma","matmul_resid_f16_wmma",["k_size"=>M::INTERMEDIATE,"n_size"=>M::HIDDEN]);
    }
    if wide {
        wide_spec!(["weight_stride"=>weight_stride::<M>(M::HIDDEN),"max_rows"=>options.max_batch*options.tokens_per_sequence,"k_size"=>M::HIDDEN,"n_size"=>if M::GATED {2*M::INTERMEDIATE} else {M::INTERMEDIATE},"splits"=>1,"epilogue"=>if M::GATED {0} else {1}]);
    } else if M::GATED {
        spec!("matmul_swiglu_f16_wmma","matmul_swiglu_f16_wmma",["k_size"=>M::HIDDEN,"n_size"=>M::INTERMEDIATE]);
    } else {
        spec!("matmul_gelu_f16_wmma","matmul_gelu_f16_wmma",["weight_stride"=>weight_stride::<M>(M::HIDDEN),"k_size"=>M::HIDDEN,"n_size"=>M::INTERMEDIATE]);
    }
    if wide {
        wide_spec!(["weight_stride"=>weight_stride::<M>(M::INTERMEDIATE),"max_rows"=>options.tokens_per_sequence,"k_size"=>M::INTERMEDIATE,"n_size"=>M::HIDDEN,"splits"=>4,"epilogue"=>0]);
    } else {
        spec!("matmul_splitk_f16_wmma","matmul_splitk_f16_wmma",["weight_stride"=>weight_stride::<M>(M::INTERMEDIATE),"k_size"=>M::INTERMEDIATE,"n_size"=>M::HIDDEN,"splits"=>4]);
    }
    if M::RESIDUAL_F32 {
        spec!("splitk_reduce_f32","splitk_reduce_f32",["n_size"=>M::HIDDEN,"splits"=>4]);
    } else {
        spec!("splitk_reduce_f16","splitk_reduce_f16",["n_size"=>M::HIDDEN,"splits"=>4]);
    }
    if separate_qkv::<M>() {
        spec!("rope_f32_to_f16","rope_f32_to_f16",["hidden_size"=>M::HIDDEN,"max_rows"=>options.max_batch*options.tokens_per_sequence,"head_dim"=>M::HEAD_DIM,"tokens_per_image"=>options.tokens_per_sequence,"prefix"=>options.prefix_tokens,"output_stride"=>qkv_stride::<M>()]);
        if wide {
            spec!("swiglu_f32_to_f16","swiglu_f32_to_f16",["width"=>M::INTERMEDIATE,"max_rows"=>options.max_batch*options.tokens_per_sequence]);
        }
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    struct Spec<const H: usize, const HEADS: usize, const F: usize, const L: usize>;
    impl<const H: usize, const HEADS: usize, const F: usize, const L: usize> EncoderSpec
        for Spec<H, HEADS, F, L>
    {
        const HIDDEN: usize = H;
        const HEADS: usize = HEADS;
        const INTERMEDIATE: usize = F;
        const LAYERS: usize = L;
        const GATED: bool = false;
    }
    #[test]
    fn rejects_unsupported_architectures_and_sequence_extents_before_gpu_work() {
        type Valid = Spec<512, 8, 2048, 6>;
        let o = EncoderOptions::default();
        assert!(validate::<Valid>(o).is_ok());
        assert!(validate::<Spec<127, 2, 256, 1>>(o).is_err());
        assert!(validate::<Spec<128, 0, 256, 1>>(o).is_err());
        assert!(validate::<Spec<128, 3, 256, 1>>(o).is_err());
        assert!(validate::<Spec<128, 4, 256, 1>>(o).is_err());
        assert!(validate::<Spec<128, 2, 128, 1>>(o).is_err());
        assert!(validate::<Spec<128, 2, 256, 0>>(o).is_err());
        for options in [
            EncoderOptions {
                tokens_per_sequence: 0,
                ..o
            },
            EncoderOptions {
                tokens_per_sequence: 1025,
                ..o
            },
            EncoderOptions {
                tokens_per_sequence: 16,
                prefix_tokens: 16,
                ..o
            },
            EncoderOptions {
                prefix_tokens: 65,
                ..o
            },
            EncoderOptions { max_batch: 0, ..o },
            EncoderOptions {
                max_batch: usize::MAX,
                ..o
            },
            EncoderOptions {
                tokens_per_sequence: 1024,
                max_batch: 1024,
                ..o
            },
            EncoderOptions {
                epsilon: f32::NAN,
                ..o
            },
            EncoderOptions {
                epsilon: f32::INFINITY,
                ..o
            },
            EncoderOptions { epsilon: 0., ..o },
        ] {
            assert!(validate::<Valid>(options).is_err(), "{options:?}");
        }
    }
    #[test]
    fn workspace_budget_admits_taste_and_accounts_for_width() {
        type Taste = Spec<384, 6, 1536, 1>;
        let o = EncoderOptions {
            max_batch: 1024,
            tokens_per_sequence: 32,
            max_workspace_bytes: 512 * 1024 * 1024,
            ..Default::default()
        };
        validate::<Taste>(o).unwrap();
        let bytes = buffer_sizes::<Taste>(1024, 32)
            .unwrap()
            .into_iter()
            .enumerate()
            .filter_map(|(index, bytes)| (index != 3).then_some(bytes))
            .sum::<usize>();
        assert!(bytes < o.max_workspace_bytes);
        let exact = EncoderOptions {
            max_workspace_bytes: bytes,
            ..o
        };
        validate::<Taste>(exact).unwrap();
        let error = validate::<Taste>(EncoderOptions {
            max_workspace_bytes: bytes - 1,
            ..o
        })
        .unwrap_err()
        .to_string();
        assert!(error.contains("workspace requires"));
        assert!(
            validate::<crate::ViT7B16>(o)
                .unwrap_err()
                .to_string()
                .contains("workspace requires")
        );
        assert!(buffer_sizes::<Taste>(usize::MAX, 32).is_err());
        // DINO's existing maximum batch remains within the default byte budget.
        validate::<crate::ViT7B16>(EncoderOptions {
            max_batch: 64,
            prefix_tokens: 5,
            ..Default::default()
        })
        .unwrap();
    }
}
