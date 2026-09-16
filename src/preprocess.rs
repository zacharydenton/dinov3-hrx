//! RGB normalization and patch layout in one pass.
use super::*;

pub(crate) fn fragment(context: &ModelContext, batch: usize) -> hrx::Result<ModelFragment> {
    let input_desc =
        TensorDesc::new(DType::U8, vec![batch, 224, 224, 3])?.with_layout(Layout::Nhwc)?;
    let output_desc = TensorDesc::new(DType::F32, vec![batch, 196, 768])?;
    let count = input_desc.elements();
    let source = include_str!("../kernels/rgb_patches.loom")
        .replace("@COUNT@", &count.to_string())
        .replace("@LAST@", &(count - 1).to_string())
        .replace("@GRID@", &count.div_ceil(256).to_string());
    let mut model = ModelSession::in_context(context)?;
    let input = model.allocate(input_desc.bytes())?;
    let output = model.allocate(output_desc.bytes())?;
    // The source fixes channel/patch geometry and bounds all accesses to count.
    let kernel = unsafe { model.compile(&[(&source, Specialization::new("rgb_patches"))])? }[0];
    unsafe {
        model.freeze(context)?.fragment(
            &[Command::Dispatch(Dispatch::indices(
                kernel,
                [0],
                [count.div_ceil(256) as u32, 1, 1],
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
    #[ignore = "requires gfx1151 and Loom compiler"]
    fn fused_rgb_patches_match_shared_operations_exactly() -> Result<()> {
        let context = ModelContext::new(Default::default())?;
        let images = ImageOps::new(&context, 8)?;
        for batch in [1, 3] {
            let desc =
                TensorDesc::new(DType::U8, vec![batch, 224, 224, 3])?.with_layout(Layout::Nhwc)?;
            let rgb = (0..desc.elements())
                .map(|i| ((i * 37 + i / 751) % 256) as u8)
                .collect::<Vec<_>>();
            let pixels = context.upload(desc, &rgb)?;
            let normalized =
                images.normalize_rgb(&pixels, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])?;
            let patches = images.patchify(&normalized.outputs()[0], 16)?;
            let expected = patches.download()?.wait()?.remove(0);
            let actual = fragment(&context, batch)?
                .prepare(1)?
                .submit(&[pixels])?
                .download()?
                .wait()?
                .remove(0);
            assert_eq!(actual, expected, "batch {batch}");
        }
        Ok(())
    }
}
