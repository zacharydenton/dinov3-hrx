use crate::ModelSpec;
use crate::checkpoint::Checkpoint;
use anyhow::{Result, ensure};
use half::{bf16, f16};
use hrx::artifacts::safetensors::DType;
use std::{collections::HashMap, path::Path};

fn pack_f16(name: &str, values: &[f32]) -> Result<Vec<u8>> {
    let mut bytes = Vec::with_capacity(values.len() * 2);
    for (index, &value) in values.iter().enumerate() {
        let narrowed = f16::from_f32(value);
        ensure!(
            narrowed.is_finite(),
            "tensor {name}[{index}] cannot be represented as finite float16: {value}"
        );
        bytes.extend_from_slice(&narrowed.to_le_bytes());
    }
    Ok(bytes)
}

pub(crate) fn load<M: ModelSpec>(path: &Path) -> Result<HashMap<String, Vec<u8>>> {
    let mut tensors = Checkpoint::open(path)?;
    ensure!(
        M::GATED || !tensors.contains("layer.0.mlp.gate_proj.weight")?,
        "checkpoint uses a gated MLP, but {} expects GELU",
        M::NAME
    );
    let mut get = |name: &str, shape: &[usize]| -> Result<Vec<f32>> {
        let t = tensors.get(name)?;
        ensure!(
            t.shape == shape,
            "{name}: expected {shape:?}, found {:?}",
            t.shape
        );
        let v: Vec<f32> = match t.dtype {
            DType::F32 => t
                .bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|x| f32::from_le_bytes(*x))
                .collect(),
            DType::F16 => t
                .bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|x| f16::from_le_bytes(*x).to_f32())
                .collect(),
            DType::BF16 => t
                .bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|x| bf16::from_le_bytes(*x).to_f32())
                .collect(),
            x => anyhow::bail!("unsupported dtype {x:?} for {name}"),
        };
        ensure!(v.iter().all(|x| x.is_finite()), "non-finite tensor {name}");
        Ok(v)
    };
    let mut out = HashMap::new();
    let mut emit = |name: String, values: Vec<f32>, half: bool| -> Result<()> {
        let bytes = if half {
            pack_f16(&name, &values)?
        } else {
            values.iter().flat_map(|x| x.to_le_bytes()).collect()
        };
        out.insert(name, bytes);
        Ok(())
    };
    emit(
        "patch_w".into(),
        get(
            "embeddings.patch_embeddings.weight",
            &[M::HIDDEN, 3, 16, 16],
        )?,
        true,
    )?;
    emit(
        "patch_b".into(),
        get("embeddings.patch_embeddings.bias", &[M::HIDDEN])?,
        false,
    )?;
    let mut prefix = get("embeddings.cls_token", &[1, 1, M::HIDDEN])?;
    prefix.extend(get("embeddings.register_tokens", &[1, 4, M::HIDDEN])?);
    emit("prefix".into(), prefix, false)?;
    for i in 0..M::LAYERS {
        let p = format!("layer.{i}.");
        for (short, long) in [
            ("norm1_w", "norm1.weight"),
            ("norm1_b", "norm1.bias"),
            ("norm2_w", "norm2.weight"),
            ("norm2_b", "norm2.bias"),
            ("ls1", "layer_scale1.lambda1"),
            ("ls2", "layer_scale2.lambda1"),
        ] {
            emit(
                format!("l{i}_{short}"),
                get(&(p.clone() + long), &[M::HIDDEN])?,
                false,
            )?;
        }
        let mut qw = vec![];
        let mut qb = vec![];
        for q in ["q", "k", "v"] {
            qw.extend(get(
                &format!("{p}attention.{q}_proj.weight"),
                &[M::HIDDEN, M::HIDDEN],
            )?);
            qb.extend(if q == "k" || !M::QV_BIAS {
                vec![0.; M::HIDDEN]
            } else {
                get(&format!("{p}attention.{q}_proj.bias"), &[M::HIDDEN])?
            });
        }
        emit(format!("l{i}_qkv_w"), qw, true)?;
        emit(format!("l{i}_qkv_b"), qb, false)?;
        if M::GATED {
            let mut gw = get(
                &(p.clone() + "mlp.gate_proj.weight"),
                &[M::INTERMEDIATE, M::HIDDEN],
            )?;
            gw.extend(get(
                &(p.clone() + "mlp.up_proj.weight"),
                &[M::INTERMEDIATE, M::HIDDEN],
            )?);
            let mut gb = get(&(p.clone() + "mlp.gate_proj.bias"), &[M::INTERMEDIATE])?;
            gb.extend(get(&(p.clone() + "mlp.up_proj.bias"), &[M::INTERMEDIATE])?);
            emit(format!("l{i}_gateup_w"), gw, true)?;
            emit(format!("l{i}_gateup_b"), gb, false)?;
        } else {
            emit(
                format!("l{i}_up_w"),
                get(
                    &(p.clone() + "mlp.up_proj.weight"),
                    &[M::INTERMEDIATE, M::HIDDEN],
                )?,
                true,
            )?;
            emit(
                format!("l{i}_up_b"),
                get(&(p.clone() + "mlp.up_proj.bias"), &[M::INTERMEDIATE])?,
                false,
            )?;
        }
        for (short, long, k) in [
            ("o", "attention.o_proj", M::HIDDEN),
            ("down", "mlp.down_proj", M::INTERMEDIATE),
        ] {
            emit(
                format!("l{i}_{short}_w"),
                get(&format!("{p}{long}.weight"), &[M::HIDDEN, k])?,
                true,
            )?;
            emit(
                format!("l{i}_{short}_b"),
                get(&format!("{p}{long}.bias"), &[M::HIDDEN])?,
                false,
            )?;
        }
    }
    emit("norm_w".into(), get("norm.weight", &[M::HIDDEN])?, false)?;
    emit("norm_b".into(), get("norm.bias", &[M::HIDDEN])?, false)?;
    let mut cos = vec![];
    let mut sin = vec![];
    for y in 0..14 {
        for x in 0..14 {
            for _ in 0..2 {
                for pos in [y, x] {
                    for j in 0..M::HEAD_DIM / 4 {
                        let coord = 2. * (pos as f64 + 0.5) / 14. - 1.;
                        let a = 2. * std::f64::consts::PI * coord
                            / 100f64.powf(j as f64 / (M::HEAD_DIM / 4) as f64);
                        cos.push(a.cos() as f32);
                        sin.push(a.sin() as f32);
                    }
                }
            }
        }
    }
    emit("rope_cos".into(), cos, false)?;
    emit("rope_sin".into(), sin, false)?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Sparse zero-filled fixtures keep CPU tests independent of model downloads.
    // Distinct first values check packed projection order and dtype conversion.
    fn fixture(path: &Path, h: usize, gated: bool, dtype: &str) -> Result<()> {
        use std::io::{Seek, SeekFrom, Write};
        let f = h * 4;
        let mut shapes = vec![
            (
                "embeddings.patch_embeddings.weight".to_owned(),
                vec![h, 3, 16, 16],
            ),
            ("embeddings.patch_embeddings.bias".to_owned(), vec![h]),
            ("embeddings.cls_token".to_owned(), vec![1, 1, h]),
            ("embeddings.register_tokens".to_owned(), vec![1, 4, h]),
            ("norm.weight".to_owned(), vec![h]),
            ("norm.bias".to_owned(), vec![h]),
        ];
        for i in 0..12 {
            for name in [
                "norm1.weight",
                "norm1.bias",
                "norm2.weight",
                "norm2.bias",
                "layer_scale1.lambda1",
                "layer_scale2.lambda1",
            ] {
                shapes.push((format!("layer.{i}.{name}"), vec![h]));
            }
            for (name, n, k, bias) in [
                ("attention.q_proj", h, h, true),
                ("attention.k_proj", h, h, false),
                ("attention.v_proj", h, h, true),
                ("attention.o_proj", h, h, true),
                ("mlp.up_proj", f, h, true),
                ("mlp.down_proj", h, f, true),
            ] {
                shapes.push((format!("layer.{i}.{name}.weight"), vec![n, k]));
                if bias {
                    shapes.push((format!("layer.{i}.{name}.bias"), vec![n]));
                }
            }
            if gated {
                shapes.push((format!("layer.{i}.mlp.gate_proj.weight"), vec![f, h]));
                shapes.push((format!("layer.{i}.mlp.gate_proj.bias"), vec![f]));
            }
        }
        let mut header = serde_json::Map::new();
        let mut offset = 0;
        let bytes_per_element = if dtype == "F32" { 4 } else { 2 };
        let mut markers = Vec::new();
        for (name, shape) in shapes {
            let end = offset + shape.iter().product::<usize>() * bytes_per_element;
            let marker: f32 = if name.contains("q_proj") {
                1.
            } else if name.contains("k_proj") {
                2.
            } else if name.contains("v_proj") {
                3.
            } else if name.contains("gate_proj") {
                4.
            } else {
                5.
            };
            markers.push((offset, marker));
            header.insert(
                name,
                serde_json::json!({"dtype":dtype,"shape":shape,"data_offsets":[offset,end]}),
            );
            offset = end;
        }
        let mut header = serde_json::to_vec(&header)?;
        header.resize(header.len().next_multiple_of(8), b' ');
        let mut file = std::fs::File::create(path)?;
        file.write_all(&(header.len() as u64).to_le_bytes())?;
        file.write_all(&header)?;
        let start = 8 + header.len();
        file.set_len((start + offset) as u64)?;
        for (offset, marker) in markers {
            file.seek(SeekFrom::Start((start + offset) as u64))?;
            match dtype {
                "F32" => file.write_all(&marker.to_le_bytes())?,
                "F16" => file.write_all(&f16::from_f32(marker).to_le_bytes())?,
                _ => file.write_all(&bf16::from_f32(marker).to_le_bytes())?,
            }
        }
        Ok(())
    }

    fn check_packing<M: ModelSpec>() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("fixture.safetensors");
        for dtype in ["F32", "F16", "BF16"] {
            fixture(&path, M::HIDDEN, M::GATED, dtype)?;
            let packed = load::<M>(&path)?;
            let read_half = |name: &str, i: usize| {
                f16::from_le_bytes(packed[name][i * 2..i * 2 + 2].try_into().unwrap()).to_f32()
            };
            let read_float = |name: &str, i: usize| {
                f32::from_le_bytes(packed[name][i * 4..i * 4 + 4].try_into().unwrap())
            };
            for i in [0, 11] {
                let w = format!("l{i}_qkv_w");
                assert_eq!(packed[&w].len(), 3 * M::HIDDEN * M::HIDDEN * 2);
                for (projection, value) in [1., 2., 3.].into_iter().enumerate() {
                    assert_eq!(read_half(&w, projection * M::HIDDEN * M::HIDDEN), value);
                }
                let bias = format!("l{i}_qkv_b");
                assert_eq!(read_float(&bias, 0), 1.);
                assert_eq!(read_float(&bias, M::HIDDEN), 0.);
                assert_eq!(read_float(&bias, 2 * M::HIDDEN), 3.);
                let mlp = format!("l{i}_{}_w", if M::GATED { "gateup" } else { "up" });
                let count = M::INTERMEDIATE * M::HIDDEN;
                assert_eq!(packed[&mlp].len(), count * 2 * if M::GATED { 2 } else { 1 });
                assert_eq!(read_half(&mlp, 0), if M::GATED { 4. } else { 5. });
                if M::GATED {
                    assert_eq!(read_half(&mlp, count), 5.);
                }
                assert_eq!(packed[&format!("l{i}_down_w")].len(), count * 2);
            }
            assert_eq!(packed["prefix"].len(), 5 * M::HIDDEN * 4);
            assert_eq!(packed["rope_cos"].len(), 196 * 64 * 4);
        }
        Ok(())
    }

    #[test]
    fn architecture_packing_preserves_shapes_dtypes_and_projection_order() -> Result<()> {
        check_packing::<crate::ViTS16Plus>()?;
        check_packing::<crate::ViTB16>()
    }

    #[test]
    fn packing_preserves_finite_values_at_float16_limits() -> Result<()> {
        // Values just above MAX can still round to a finite float16 value.
        let bytes = pack_f16(
            "weights",
            &[0., -0., 1., -1., 65504., -65504., 65519., -65519.],
        )?;
        let decoded: Vec<_> = bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|bytes| f16::from_le_bytes(*bytes).to_f32())
            .collect();
        assert_eq!(
            decoded,
            [0., -0., 1., -1., 65504., -65504., 65504., -65504.]
        );
        assert!(decoded[1].is_sign_negative());
        Ok(())
    }
}
