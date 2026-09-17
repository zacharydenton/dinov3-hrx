//! Numerical tests for the new kernel paths, independent of model checkpoints.
use super::*;
use half::f16;

fn run_kernel(
    index: usize,
    input: &[f16],
    shape: Vec<usize>,
    parameters: &[Vec<u8>],
    output: TensorDesc,
    grid: [u32; 3],
) -> Result<Vec<f32>> {
    let context = ModelContext::new(Default::default())?;
    let mut model = ModelSession::in_context(&context)?;
    let input_desc = TensorDesc::new(DType::F16, shape.clone())?;
    let x = model.allocate(input_desc.bytes())?;
    let out = model.allocate(output.bytes())?;
    let specs = specifications::<ViTB16>();
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
        .submit_host(&[bytemuck::cast_slice(input)])?
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
        let actual = run_kernel(
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
    let actual = run_kernel(
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
