use super::*;

pub(crate) fn fragment(context: &ModelContext, batch: usize) -> hrx::Result<ModelFragment> {
    let mut model = ModelSession::in_context(context)?;
    let input_desc = TensorDesc::new(DType::F32, vec![batch, TOKENS, HIDDEN])?;
    let mask_desc = TensorDesc::new(DType::U8, vec![batch, 196])?;
    let output_desc = TensorDesc::new(DType::F32, vec![batch, 2, HIDDEN])?;
    let tokens = model.allocate(input_desc.bytes())?;
    let masks = model.allocate(mask_desc.bytes())?;
    let pooled = model.allocate(output_desc.bytes())?;
    let output = model.allocate_shared(output_desc.bytes())?;
    // Both kernels are embedded here; all accesses are bounded by rows*384.
    let kernels = unsafe {
        model.compile(&[
            (
                include_str!("../kernels/descriptor_pool.loom"),
                Specialization::new("dinov3_pool"),
            ),
            (
                include_str!("../kernels/descriptor_l2.loom"),
                Specialization::new("dinov3_descriptor_l2"),
            ),
        ])?
    };
    let rows = (batch * 2) as u32;
    let grid = [rows, 1, 1];
    let commands = [
        Command::Dispatch(Dispatch::indices(
            kernels[0],
            [rows],
            grid,
            vec![tokens.read(), masks.read(), pooled.write()],
        )),
        Command::Dispatch(Dispatch::indices(
            kernels[1],
            [rows],
            [(batch * 2 * HIDDEN).div_ceil(256) as u32, 1, 1],
            vec![pooled.read(), output.write()],
        )),
    ];
    unsafe {
        model.freeze(context)?.fragment(
            &commands,
            &[(tokens, input_desc), (masks, mask_desc)],
            &[(output, output_desc)],
        )
    }
}

/// Raw summaries preserve the pre-existing public CLS and patch-mean values.
pub(crate) fn raw_fragment(
    context: &ModelContext,
    batch: usize,
    mean: bool,
) -> hrx::Result<ModelFragment> {
    let input_desc = TensorDesc::new(DType::F32, vec![batch, TOKENS, HIDDEN])?;
    let output_desc = TensorDesc::new(DType::F32, vec![batch, HIDDEN])?;
    let mut source = include_str!("../kernels/raw_descriptor.loom").to_owned();
    for (key, value) in [
        ("GRID", output_desc.elements().div_ceil(256).to_string()),
        ("COUNT", output_desc.elements().to_string()),
        ("LAST", (output_desc.elements() - 1).to_string()),
        ("INPUT", input_desc.elements().to_string()),
        ("INPUT_LAST", (input_desc.elements() - 1).to_string()),
        ("MEAN", mean.to_string()),
    ] {
        source = source.replace(&format!("@{key}@"), &value);
    }
    let mut model = ModelSession::in_context(context)?;
    let input = model.allocate(input_desc.bytes())?;
    let output = model.allocate_shared(output_desc.bytes())?;
    // Every access is bounded by the descriptors above.
    let kernel = unsafe { model.compile(&[(&source, Specialization::new("raw_pool"))])? }[0];
    unsafe {
        model.freeze(context)?.fragment(
            &[Command::Dispatch(Dispatch::indices(
                kernel,
                [0],
                [output_desc.elements().div_ceil(256) as u32, 1, 1],
                vec![input.read(), output.write()],
            ))],
            &[(input, input_desc)],
            &[(output, output_desc)],
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[ignore = "requires GPU and Loom compiler"]
    fn masked_descriptor_pool_matches_host_including_empty_mask() -> Result<()> {
        let context = ModelContext::new(Default::default())?;
        let plan = fragment(&context, 2)?.prepare(3)?;
        let values = (0..2 * TOKENS * HIDDEN)
            .map(|i| ((i % 71) as f32 - 30.) / 37.)
            .collect::<Vec<_>>();
        let masks = (0..392)
            .map(|i| u8::from(i < 196 && i % 3 != 0))
            .collect::<Vec<_>>();
        let output = plan
            .submit_host(&[bytemuck::cast_slice(&values), &masks])?
            .download()?
            .wait()?
            .remove(0);
        let actual = output
            .as_chunks::<4>()
            .0
            .iter()
            .map(|v| f32::from_le_bytes(*v))
            .collect::<Vec<_>>();
        let mut expected = Vec::new();
        for b in 0..2 {
            let mut cls = values[b * TOKENS * HIDDEN..b * TOKENS * HIDDEN + HIDDEN].to_vec();
            let mut mean = vec![0f32; HIDDEN];
            let mut kept = 0;
            for p in 0..196 {
                if masks[b * 196 + p] != 0 {
                    kept += 1;
                    for h in 0..HIDDEN {
                        mean[h] += values[(b * TOKENS + p + 5) * HIDDEN + h];
                    }
                }
            }
            if kept > 0 {
                for v in &mut mean {
                    *v /= kept as f32;
                }
            }
            for row in [&mut cls, &mut mean] {
                let sum = row.iter().map(|v| (*v as f64).powi(2)).sum::<f64>();
                let length = if sum > 0. { sum.sqrt() as f32 } else { 1. };
                for value in row.iter_mut() {
                    *value /= length;
                }
                expected.extend_from_slice(row);
            }
        }
        for (a, e) in actual.iter().zip(expected) {
            assert!((a - e).abs() < 2e-7, "{a} vs {e}");
        }
        Ok(())
    }
}
