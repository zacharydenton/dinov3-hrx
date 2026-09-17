//! Numerical tests for the new kernel paths, independent of model checkpoints.
use super::*;
use half::f16;

fn run_kernel<M: ModelSpec>(
    index: usize,
    input: &[f16],
    shape: Vec<usize>,
    parameters: &[Vec<u8>],
    output: TensorDesc,
    grid: [u32; 3],
) -> Result<Vec<f32>> {
    run_kernel_bytes::<M>(
        index,
        bytemuck::cast_slice(input),
        DType::F16,
        shape,
        parameters,
        output,
        grid,
    )
}
fn run_kernel_bytes<M: ModelSpec>(
    index: usize,
    input: &[u8],
    input_dtype: DType,
    shape: Vec<usize>,
    parameters: &[Vec<u8>],
    output: TensorDesc,
    grid: [u32; 3],
) -> Result<Vec<f32>> {
    let context = ModelContext::new(Default::default())?;
    let mut model = ModelSession::in_context(&context)?;
    let input_desc = TensorDesc::new(input_dtype, shape.clone())?;
    let x = model.allocate(input_desc.bytes())?;
    let out = model.allocate(output.bytes())?;
    let specs = specifications::<M>();
    let kernel = unsafe { model.compile(&[specs[index].clone()])? }[0];
    let mut bindings = vec![x.read()];
    for parameter in parameters {
        bindings.push(model.weight(parameter)?.read());
    }
    bindings.push(out.write());
    let commands = [Command::Dispatch(Dispatch::indices(
        kernel,
        [shape[0] as u32],
        grid,
        bindings,
    ))];
    let fragment = unsafe {
        model
            .freeze(&context)?
            .fragment(&commands, &[(x, input_desc)], &[(out, output.clone())])?
    };
    let bytes = fragment
        .prepare(1)?
        .submit_host(&[input])?
        .download()?
        .wait()?
        .remove(0);
    Ok(if output.dtype() == DType::F16 {
        bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|v| f16::from_le_bytes(*v).to_f32())
            .collect()
    } else {
        bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|v| f32::from_le_bytes(*v))
            .collect()
    })
}

#[test]
#[ignore = "requires gfx1151 and Loom"]
fn wide_layernorm_keeps_all_channels_and_partial_row_groups() -> Result<()> {
    let rows = 9;
    let input: Vec<_> = (0..rows * 768)
        .map(|i| f16::from_f32(((i * 19 % 127) as f32 - 63.) / 17. + (i % 768 / 384) as f32 * 2.))
        .collect();
    let gamma: Vec<_> = (0..768).map(|i| 0.5 + i as f32 / 768.).collect();
    let beta: Vec<_> = (0..768).map(|i| (i as f32 - 384.) / 768.).collect();
    for (kernel, dtype) in [(2, DType::F16), (3, DType::F32)] {
        let actual = run_kernel::<ViTB16>(
            kernel,
            &input,
            vec![rows, 768],
            &[
                bytemuck::cast_slice(&gamma).to_vec(),
                bytemuck::cast_slice(&beta).to_vec(),
            ],
            TensorDesc::new(dtype, vec![rows, 768])?,
            [rows.div_ceil(8) as u32, 1, 1],
        )?;
        for (row, got) in input.chunks(768).zip(actual.chunks(768)) {
            let mean = row.iter().map(|x| x.to_f64()).sum::<f64>() / 768.;
            let var = row.iter().map(|x| (x.to_f64() - mean).powi(2)).sum::<f64>() / 768.;
            for c in 0..768 {
                let want = ((row[c].to_f64() - mean) / (var + 1e-5).sqrt() * gamma[c] as f64
                    + beta[c] as f64) as f32;
                let tolerance = if dtype == DType::F16 {
                    0.001 * want.abs().max(1.)
                } else {
                    2e-5
                };
                assert!(
                    (got[c] - want).abs() < tolerance,
                    "{dtype:?} channel {c}: {} != {want}",
                    got[c]
                );
            }
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires gfx1151 and Loom"]
fn fused_gelu_matches_erf_reference_with_partial_tiles() -> Result<()> {
    let rows = 65;
    let input: Vec<_> = (0..rows * 768)
        .map(|i| f16::from_f32((i % 33) as f32 / 16. - 1.))
        .collect();
    let mut weights = vec![f16::ZERO; 3072 * 768];
    for c in 0..3072 {
        weights[c * 768 + c % 768] = f16::from_f32(0.5);
    }
    let bias: Vec<_> = (0..3072).map(|c| (c as f32 - 1536.) / 128.).collect();
    let actual = run_kernel::<ViTB16>(
        8,
        &input,
        vec![rows, 768],
        &[
            bytemuck::cast_slice(&weights).to_vec(),
            bytemuck::cast_slice(&bias).to_vec(),
        ],
        TensorDesc::new(DType::F16, vec![rows, 3072])?,
        [48, rows.div_ceil(64) as u32, 1],
    )?;
    for (i, got) in actual.iter().enumerate() {
        let c = i % 3072;
        let x = input[i / 3072 * 768 + c % 768].to_f64() * 0.5 + bias[c] as f64;
        let want = (0.5 * x * (1. + libm::erf(x / std::f64::consts::SQRT_2))) as f32;
        assert!(
            (got - want).abs() < 0.001 * want.abs().max(1.),
            "element {i}: {got} != {want}"
        );
    }
    Ok(())
}

#[test]
#[ignore = "requires gfx1151 and Loom"]
fn all_family_kernel_specializations_compile() -> Result<()> {
    fn compile<M: ModelSpec>(context: &ModelContext) -> Result<()> {
        let mut model = ModelSession::in_context(context)?;
        unsafe {
            model.compile(&specifications::<M>())?;
        }
        Ok(())
    }
    let context = ModelContext::new(Default::default())?;
    compile::<ViTS16>(&context)?;
    compile::<ViTL16>(&context)?;
    compile::<ViTH16Plus>(&context)?;
    compile::<ViT7B16>(&context)
}

fn large_norm<M: ModelSpec>() -> Result<()> {
    let rows = 9;
    let h = M::HIDDEN;
    let input: Vec<_> = (0..rows * h)
        .map(|i| ((i * 13 % 251) as f32 - 125.) * 2000.)
        .collect();
    let gamma: Vec<_> = (0..h).map(|i| 0.75 + i as f32 / h as f32).collect();
    let beta = vec![0.125f32; h];
    for (kernel, dtype) in [(2, DType::F16), (3, DType::F32)] {
        let got = run_kernel_bytes::<M>(
            kernel,
            bytemuck::cast_slice(&input),
            DType::F32,
            vec![rows, h],
            &[
                bytemuck::cast_slice(&gamma).to_vec(),
                bytemuck::cast_slice(&beta).to_vec(),
            ],
            TensorDesc::new(dtype, vec![rows, h])?,
            [rows.div_ceil(8) as u32, 1, 1],
        )?;
        for (row, actual) in input.chunks(h).zip(got.chunks(h)) {
            let mean = row.iter().map(|x| *x as f64).sum::<f64>() / h as f64;
            let var = row.iter().map(|x| (*x as f64 - mean).powi(2)).sum::<f64>() / h as f64;
            for c in 0..h {
                let want = ((row[c] as f64 - mean) / (var + 1e-5).sqrt() * gamma[c] as f64
                    + beta[c] as f64) as f32;
                let tolerance = if dtype == DType::F16 {
                    0.001 * want.abs().max(1.)
                } else {
                    2e-5
                };
                assert!(
                    (actual[c] - want).abs() < tolerance,
                    "{} channel {c}: {} != {want}",
                    M::NAME,
                    actual[c]
                );
            }
        }
    }
    Ok(())
}
#[test]
#[ignore = "requires gfx1151 and Loom"]
fn large_layernorm_widths_match_reference() -> Result<()> {
    large_norm::<ViTL16>()?;
    large_norm::<ViTH16Plus>()?;
    large_norm::<ViT7B16>()
}

#[test]
#[ignore = "requires gfx1151 and Loom"]
fn attention_128_matches_reference_across_heads_and_image_tails() -> Result<()> {
    let h = 4096;
    let rows = 2 * TOKENS;
    let stride = 3 * h;
    let mut qkv = vec![f16::ZERO; (rows + 16) * stride];
    for t in 0..rows {
        for c in 0..h {
            qkv[t * stride + c] = f16::from_f32(((t * 7 + c * 11) % 31) as f32 / 31. - 0.5);
            qkv[t * stride + h + c] = f16::from_f32(((t * 13 + c * 3) % 37) as f32 / 37. - 0.5);
            qkv[t * stride + 2 * h + c] = f16::from_f32(((t * 17 + c * 5) % 41) as f32 / 21. - 1.);
        }
    }
    let context = ModelContext::new(Default::default())?;
    let mut model = ModelSession::in_context(&context)?;
    let input_desc = TensorDesc::new(DType::F16, vec![rows + 16, stride])?;
    let output_desc = TensorDesc::new(DType::F16, vec![rows, h])?;
    let q = model.allocate(input_desc.bytes())?;
    let k = q.slice(h * 2, q.len() - h * 2)?;
    let v = q.slice(h * 4, q.len() - h * 4)?;
    let out = model.allocate(output_desc.bytes())?;
    let kernel = unsafe { model.compile(&[specifications::<ViT7B16>()[5].clone()])? }[0];
    let commands = [Command::Dispatch(Dispatch::indices(
        kernel,
        [rows as u32],
        [26, 32, 1],
        vec![q.read(), k.read(), v.read(), out.write()],
    ))];
    let fragment = unsafe {
        model
            .freeze(&context)?
            .fragment(&commands, &[(q, input_desc)], &[(out, output_desc)])?
    };
    let bytes = fragment
        .prepare(1)?
        .submit_host(&[bytemuck::cast_slice(&qkv)])?
        .download()?
        .wait()?
        .remove(0);
    let actual: Vec<_> = bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|v| f16::from_le_bytes(*v).to_f64())
        .collect();
    assert!(actual.iter().all(|x| x.is_finite()));
    for t in [0, 15, 16, 195, 200, 201, 216, 401] {
        for head in 0..32 {
            let base = head * 128;
            let origin = t / TOKENS * TOKENS;
            let mut scores: Vec<_> = (origin..origin + TOKENS)
                .map(|key| {
                    (0..128)
                        .map(|c| {
                            qkv[t * stride + base + c].to_f64()
                                * qkv[key * stride + h + base + c].to_f64()
                        })
                        .sum::<f64>()
                        / 128f64.sqrt()
                })
                .collect();
            let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            for score in &mut scores {
                *score = (*score - max).exp();
            }
            let sum = scores.iter().sum::<f64>();
            for c in 0..128 {
                let expected = scores
                    .iter()
                    .enumerate()
                    .map(|(key, &s)| s * qkv[(origin + key) * stride + 2 * h + base + c].to_f64())
                    .sum::<f64>()
                    / sum;
                let got = actual[t * h + base + c];
                assert!(
                    (got - expected).abs() < 0.0003,
                    "row {t} head {head} channel {c}: {got} != {expected}"
                );
            }
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires gfx1151 and Loom"]
fn rope_128_preserves_prefix_and_value_channels() -> Result<()> {
    let rows = 2 * TOKENS;
    let stride = 3 * 4096;
    let input: Vec<f32> = (0..rows * stride)
        .map(|i| ((i * 7 % 127) as f32 - 63.) / 17.)
        .collect();
    let mut cos = Vec::new();
    let mut sin = Vec::new();
    for p in 0..196 {
        for c in 0..128 {
            let pos = if c % 64 < 32 { p / 14 } else { p % 14 };
            let angle = 2. * std::f64::consts::PI * (2. * (pos as f64 + 0.5) / 14. - 1.)
                / 100f64.powf((c % 32) as f64 / 32.);
            cos.push(angle.cos() as f32);
            sin.push(angle.sin() as f32);
        }
    }
    let context = ModelContext::new(Default::default())?;
    let mut model = ModelSession::in_context(&context)?;
    let input_desc = TensorDesc::new(DType::F32, vec![rows, stride])?;
    let output_desc = TensorDesc::new(DType::F16, vec![rows, stride])?;
    let x = model.allocate(input_desc.bytes())?;
    let out = model.allocate(output_desc.bytes())?;
    let cos_region = model.weight(bytemuck::cast_slice(&cos))?;
    let sin_region = model.weight(bytemuck::cast_slice(&sin))?;
    let kernel = unsafe { model.compile(&[specifications::<ViT7B16>()[11].clone()])? }[0];
    let commands = [Command::Dispatch(Dispatch::indices(
        kernel,
        [rows as u32],
        [(rows * stride).div_ceil(256) as u32, 1, 1],
        vec![x.read(), cos_region.read(), sin_region.read(), out.write()],
    ))];
    let fragment = unsafe {
        model
            .freeze(&context)?
            .fragment(&commands, &[(x, input_desc)], &[(out, output_desc)])?
    };
    let bytes = fragment
        .prepare(1)?
        .submit_host(&[bytemuck::cast_slice(&input)])?
        .download()?
        .wait()?
        .remove(0);
    for (i, bytes) in bytes.as_chunks::<2>().0.iter().enumerate() {
        let row = i / stride;
        let c = i % stride;
        let local = row % TOKENS;
        let expected = if local >= 5 && c < 8192 {
            let channel = c % 128;
            let other = if channel < 64 {
                -input[i + 64]
            } else {
                input[i - 64]
            };
            let offset = (local - 5) * 128 + channel;
            other.mul_add(sin[offset], input[i] * cos[offset])
        } else {
            input[i]
        };
        let actual = f16::from_le_bytes(*bytes).to_f32();
        assert_eq!(
            actual,
            f16::from_f32(expected).to_f32(),
            "row {row} channel {c}"
        );
    }
    Ok(())
}
