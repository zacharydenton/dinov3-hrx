use anyhow::Result;
use dinov3_hrx::{
    Encoder, EncoderLayerWeights, EncoderOptions, EncoderSpec, EncoderWeights, LinearWeights,
    NormWeights, RotaryEmbedding,
};
use half::f16;
use hrx::{
    inference::ModelContext,
    tensor::{DType, TensorDesc},
};

// These are downstream implementations: no ModelSpec, checkpoint, Row, or sealed trait.
struct Plain;
impl EncoderSpec for Plain {
    const HIDDEN: usize = 128;
    const HEADS: usize = 2;
    const LAYERS: usize = 2;
    const INTERMEDIATE: usize = 256;
    const GATED: bool = false;
}
struct Taste;
impl EncoderSpec for Taste {
    const HIDDEN: usize = 384;
    const HEADS: usize = 6;
    const LAYERS: usize = 4;
    const INTERMEDIATE: usize = 1536;
    const GATED: bool = true;
}
struct Gated;
impl EncoderSpec for Gated {
    const HIDDEN: usize = 512;
    const HEADS: usize = 8;
    const LAYERS: usize = 2;
    const INTERMEDIATE: usize = 768;
    const GATED: bool = true;
}
struct WideHead;
impl EncoderSpec for WideHead {
    const HIDDEN: usize = 128;
    const HEADS: usize = 1;
    const LAYERS: usize = 2;
    const INTERMEDIATE: usize = 256;
    const GATED: bool = true;
    const RESIDUAL_F32: bool = true;
    const QV_BIAS: bool = false;
}
struct PlainWideHead;
impl EncoderSpec for PlainWideHead {
    const HIDDEN: usize = 256;
    const HEADS: usize = 2;
    const LAYERS: usize = 2;
    const INTERMEDIATE: usize = 512;
    const GATED: bool = false;
    const RESIDUAL_F32: bool = true;
}

fn values(n: usize, seed: usize, scale: f32) -> Vec<f32> {
    (0..n)
        .map(|i| (((i * 37 + i / 19 + seed * 71) % 113) as f32 - 56.) * scale / 56.)
        .collect()
}
fn weights<S: EncoderSpec>(o: EncoderOptions, rotary: bool) -> EncoderWeights {
    let h = S::HIDDEN;
    let f = S::INTERMEDIATE;
    let norm = |seed| NormWeights {
        weight: values(h, seed, 0.1).into_iter().map(|v| 1. + v).collect(),
        bias: values(h, seed + 1, 0.02),
    };
    let linear = |n, k, seed| LinearWeights {
        weight: values(n * k, seed, 0.12)
            .into_iter()
            .map(f16::from_f32)
            .collect(),
        bias: values(n, seed + 1, 0.02),
    };
    EncoderWeights {
        layers: (0..S::LAYERS)
            .map(|i| {
                let mut qkv = linear(3 * h, h, i * 11);
                qkv.bias[h..2 * h].fill(0.);
                if !S::QV_BIAS {
                    qkv.bias.fill(0.);
                }
                EncoderLayerWeights {
                    norm1: norm(i * 11 + 1),
                    qkv,
                    attention_output: linear(h, h, i * 11 + 2),
                    attention_scale: vec![0.4; h],
                    norm2: norm(i * 11 + 3),
                    mlp_up: linear(f * if S::GATED { 2 } else { 1 }, h, i * 11 + 4),
                    mlp_down: linear(h, f, i * 11 + 5),
                    mlp_scale: vec![0.3; h],
                }
            })
            .collect(),
        norm: norm(19),
        rotary: rotary.then(|| {
            let angles: Vec<_> = (0..(o.tokens_per_sequence - o.prefix_tokens) * S::HEAD_DIM)
                .map(|i| {
                    let p = i / S::HEAD_DIM;
                    let c = i % (S::HEAD_DIM / 2);
                    (p as f64 + 0.3) / (1. + c as f64)
                })
                .collect();
            RotaryEmbedding {
                cos: angles.iter().map(|x| x.cos() as f32).collect(),
                sin: angles.iter().map(|x| x.sin() as f32).collect(),
            }
        }),
    }
}
fn norm(x: &[f64], w: &NormWeights, eps: f64) -> Vec<f64> {
    let h = w.weight.len();
    let mut y = x.to_vec();
    for (x, y) in x.chunks(h).zip(y.chunks_mut(h)) {
        let mean = x.iter().sum::<f64>() / h as f64;
        let var = x.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / h as f64;
        for c in 0..h {
            y[c] = (x[c] - mean) / (var + eps).sqrt() * w.weight[c] as f64 + w.bias[c] as f64;
        }
    }
    y
}
fn linear(x: &[f64], w: &LinearWeights) -> Vec<f64> {
    let n = w.bias.len();
    let k = w.weight.len() / n;
    let m = x.len() / k;
    let matrix: Vec<_> = w.weight.iter().map(|v| v.to_f64()).collect();
    let mut y = vec![0.; m * n];
    // Independent F64 GEMM with dimensions and strides bounded by these vectors.
    unsafe {
        matrixmultiply::dgemm(
            m,
            k,
            n,
            1.,
            x.as_ptr(),
            k as isize,
            1,
            matrix.as_ptr(),
            1,
            k as isize,
            0.,
            y.as_mut_ptr(),
            n as isize,
            1,
        );
    }
    for row in y.chunks_mut(n) {
        for (v, b) in row.iter_mut().zip(&w.bias) {
            *v += *b as f64;
        }
    }
    y
}
fn reference<S: EncoderSpec>(input: &[f64], w: &EncoderWeights, o: EncoderOptions) -> Vec<f64> {
    let h = S::HIDDEN;
    let d = S::HEAD_DIM;
    let t = o.tokens_per_sequence;
    let rows = input.len() / h;
    let mut x = input.to_vec();
    for l in &w.layers {
        let mut qkv = linear(&norm(&x, &l.norm1, o.epsilon as f64), &l.qkv);
        if let Some(rot) = &w.rotary {
            for r in 0..rows {
                if r % t < o.prefix_tokens {
                    continue;
                }
                let old = qkv[r * 3 * h..(r + 1) * 3 * h].to_vec();
                for c in 0..2 * h {
                    let half = c % d < d / 2;
                    let partner = if half { c + d / 2 } else { c - d / 2 };
                    let p = (r % t - o.prefix_tokens) * d + c % d;
                    qkv[r * 3 * h + c] = old[c] * rot.cos[p] as f64
                        + old[partner] * rot.sin[p] as f64 * if half { -1. } else { 1. };
                }
            }
        }
        let mut attn = vec![0.; rows * h];
        for r in 0..rows {
            for head in 0..S::HEADS {
                let base = r / t * t;
                let c = head * d;
                let mut scores: Vec<_> = (0..t)
                    .map(|j| {
                        (0..d)
                            .map(|k| qkv[r * 3 * h + c + k] * qkv[(base + j) * 3 * h + h + c + k])
                            .sum::<f64>()
                            / (d as f64).sqrt()
                    })
                    .collect();
                let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                for s in &mut scores {
                    *s = (*s - max).exp();
                }
                let sum = scores.iter().sum::<f64>();
                for k in 0..d {
                    attn[r * h + c + k] = (0..t)
                        .map(|j| scores[j] / sum * qkv[(base + j) * 3 * h + 2 * h + c + k])
                        .sum();
                }
            }
        }
        for (i, v) in linear(&attn, &l.attention_output).into_iter().enumerate() {
            x[i] += v * l.attention_scale[i % h] as f64;
        }
        let up = linear(&norm(&x, &l.norm2, o.epsilon as f64), &l.mlp_up);
        let f = S::INTERMEDIATE;
        let act: Vec<_> = if S::GATED {
            up.chunks(2 * f)
                .flat_map(|r| (0..f).map(move |i| r[i] / (1. + (-r[i]).exp()) * r[f + i]))
                .collect()
        } else {
            up.iter()
                .map(|&v| v * 0.5 * (1. + libm::erf(v / std::f64::consts::SQRT_2)))
                .collect()
        };
        for (i, v) in linear(&act, &l.mlp_down).into_iter().enumerate() {
            x[i] += v * l.mlp_scale[i % h] as f64;
        }
    }
    norm(&x, &w.norm, o.epsilon as f64)
}
fn check<S: EncoderSpec>(
    context: &ModelContext,
    rotary: bool,
    compose: bool,
    tokens: usize,
    prefix: usize,
) -> Result<()> {
    let o = EncoderOptions {
        tokens_per_sequence: tokens,
        prefix_tokens: prefix,
        max_batch: 2,
        epsilon: 2e-5,
        ..Default::default()
    };
    let w = weights::<S>(o, rotary);
    let mut inputs = Vec::new();
    let mut expected = Vec::new();
    for batch in [1, 2] {
        let mut x = values(batch * tokens * S::HIDDEN, 11 + batch, 1.);
        if !S::RESIDUAL_F32 {
            for v in &mut x {
                *v = f16::from_f32(*v).to_f32();
            }
        }
        if S::RESIDUAL_F32 {
            x[17] = 100_000.;
        }
        let x64: Vec<_> = x.iter().map(|&v| v as f64).collect();
        let y = reference::<S>(&x64, &w, o);
        expected.push(if compose {
            reference::<S>(&y, &w, o)
        } else {
            y
        });
        inputs.push(x);
    }
    let encoder = Encoder::<S>::new(context, o, w)?;
    let mut prepared = Vec::new();
    for (i, x) in inputs.iter().enumerate() {
        let raw = if S::RESIDUAL_F32 {
            bytemuck::cast_slice(x).to_vec()
        } else {
            bytemuck::cast_slice(&x.iter().map(|&v| f16::from_f32(v)).collect::<Vec<_>>()).to_vec()
        };
        let input = context.upload(
            TensorDesc::new(Encoder::<S>::input_dtype(), vec![i + 1, tokens, S::HIDDEN])?,
            &raw,
        )?;
        let mut graph = context.runtime().graph();
        let first = encoder.record(&mut graph, &input)?;
        let output = if compose {
            encoder.record(&mut graph, &first)?
        } else {
            first
        };
        // Wrong shape/dtype must fail without mutating graph inputs.
        let wrong = context.allocate(TensorDesc::new(DType::F32, vec![1, 34, S::HIDDEN])?)?;
        assert!(encoder.record(&mut graph, &wrong).is_err());
        prepared.push((graph.prepare()?, input, output, raw));
    }
    drop(encoder); // Graphs retain code/weights and use independent scratch.
    for (i, (graph, input, output, raw)) in prepared.into_iter().enumerate().rev() {
        for _ in 0..2 {
            graph.submit()?.wait()?;
            let bytes = context.download(&output)?.wait()?;
            assert_eq!(
                context.download(&input)?.wait()?,
                raw,
                "record must preserve caller input"
            );
            let actual: Vec<f64> = bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()) as f64)
                .collect();
            let want = &expected[i];
            assert!(actual.iter().all(|x| x.is_finite()));
            let cosine = actual.iter().zip(want).map(|(a, b)| a * b).sum::<f64>()
                / (actual.iter().map(|x| x * x).sum::<f64>()
                    * want.iter().map(|x| x * x).sum::<f64>())
                .sqrt();
            let max = actual
                .iter()
                .zip(want)
                .map(|(a, b)| (a - b).abs())
                .fold(0., f64::max);
            assert!(
                cosine > 0.9999 && max < 0.025,
                "{}: cosine={cosine}, max={max}",
                std::any::type_name::<S>()
            );
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires gfx1151 and Loom; no checkpoints required"]
fn downstream_encoder_specs_match_reference_and_compose_in_caller_graphs() -> Result<()> {
    let context = ModelContext::new(Default::default())?;
    let o = EncoderOptions::default();
    let mut bad = weights::<WideHead>(o, true);
    bad.layers[0].qkv.bias[0] = 1.;
    assert!(Encoder::<WideHead>::new(&context, o, bad).is_err());
    let mut bad = weights::<Plain>(o, true);
    bad.layers[0].mlp_up.weight[0] = f16::INFINITY;
    assert!(Encoder::<Plain>::new(&context, o, bad).is_err());
    let mut bad = weights::<Plain>(o, true);
    bad.rotary.as_mut().unwrap().cos.pop();
    assert!(Encoder::<Plain>::new(&context, o, bad).is_err());
    let mut bad = weights::<Plain>(o, false);
    bad.layers[0].norm1.weight.pop();
    assert!(Encoder::<Plain>::new(&context, o, bad).is_err());
    check::<Plain>(&context, false, false, 33, 3)?;
    check::<Plain>(&context, false, false, 1, 0)?;
    check::<Gated>(&context, true, false, 33, 3)?;
    check::<WideHead>(&context, true, true, 33, 3)?;
    check::<WideHead>(&context, true, false, 17, 0)?;
    check::<PlainWideHead>(&context, false, false, 33, 3)?;
    check::<PlainWideHead>(&context, true, false, 257, 64)
}

#[test]
fn architecture_constants_do_not_require_model_metadata() {
    fn architecture<S: EncoderSpec>() -> (usize, usize) {
        (S::HIDDEN, S::LAYERS)
    }
    assert_eq!(architecture::<Plain>(), (128, 2));
    assert_eq!(<dinov3_hrx::ViT7B16 as EncoderSpec>::HEAD_DIM, 128);
}

#[test]
#[ignore = "requires gfx1151 and Loom; no checkpoints required"]
fn taste_shape_records_1024_sequences_of_32_tokens() -> Result<()> {
    let context = ModelContext::new(Default::default())?;
    let options = EncoderOptions {
        max_batch: 1024,
        tokens_per_sequence: 32,
        max_workspace_bytes: 512 * 1024 * 1024,
        ..Default::default()
    };
    let mut weights = weights::<Taste>(options, false);
    for layer in &mut weights.layers {
        // Taste's ordinary biased QKV projection and unscaled residual branches.
        layer.qkv.bias[Taste::HIDDEN..2 * Taste::HIDDEN].fill(0.02);
        layer.attention_scale.fill(1.);
        layer.mlp_scale.fill(1.);
    }
    let patterns: Vec<Vec<f16>> = (0..3)
        .map(|seed| {
            values(32 * Taste::HIDDEN, seed + 7, 1.)
                .into_iter()
                .map(f16::from_f32)
                .collect()
        })
        .collect();
    let expected: Vec<_> = patterns
        .iter()
        .map(|input| {
            reference::<Taste>(
                &input.iter().map(|v| v.to_f64()).collect::<Vec<_>>(),
                &weights,
                options,
            )
        })
        .collect();
    let input: Vec<f16> = (0..1024)
        .flat_map(|b| patterns[b % 3].iter().copied())
        .collect();
    let tokens = context.upload(
        TensorDesc::new(DType::F16, vec![1024, 32, Taste::HIDDEN])?,
        bytemuck::cast_slice(&input),
    )?;
    let encoder = Encoder::<Taste>::new(&context, options, weights)?;
    let mut graph = context.runtime().graph();
    let output = encoder.record(&mut graph, &tokens)?;
    let graph = graph.prepare()?;
    for _ in 0..2 {
        graph.submit()?.wait()?;
        let bytes = context.download(&output)?.wait()?;
        assert_eq!(bytes.len(), 1024 * 32 * Taste::HIDDEN * 4);
        for (b, sequence) in bytes.chunks_exact(32 * Taste::HIDDEN * 4).enumerate() {
            let mut dot = 0.;
            let mut actual_norm = 0.;
            let mut reference_norm = 0.;
            for (element, &want) in sequence.chunks_exact(4).zip(&expected[b % 3]) {
                let got = f32::from_le_bytes(element.try_into().unwrap()) as f64;
                assert!(got.is_finite());
                dot += got * want;
                actual_norm += got * got;
                reference_norm += want * want;
            }
            assert!(
                dot / (actual_norm * reference_norm).sqrt() > 0.9999,
                "sequence {b}"
            );
        }
    }
    assert_eq!(
        context.download(&tokens)?.wait()?,
        bytemuck::cast_slice::<f16, u8>(&input)
    );
    Ok(())
}
