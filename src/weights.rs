use anyhow::{Context, Result, ensure};
use half::{bf16, f16};
use safetensors::{Dtype, SafeTensors};
use std::{collections::HashMap, path::Path};
pub(crate) fn load(path: &Path) -> Result<HashMap<String, Vec<u8>>> {
    let file = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    let tensors = SafeTensors::deserialize(&file)?;
    let get = |name: &str, shape: &[usize]| -> Result<Vec<f32>> {
        let t = tensors
            .tensor(name)
            .with_context(|| format!("missing tensor {name}"))?;
        ensure!(
            t.shape() == shape,
            "{name}: expected {shape:?}, found {:?}",
            t.shape()
        );
        let v: Vec<f32> = match t.dtype() {
            Dtype::F32 => t
                .data()
                .chunks_exact(4)
                .map(|x| f32::from_le_bytes(x.try_into().unwrap()))
                .collect(),
            Dtype::F16 => t
                .data()
                .chunks_exact(2)
                .map(|x| f16::from_le_bytes(x.try_into().unwrap()).to_f32())
                .collect(),
            Dtype::BF16 => t
                .data()
                .chunks_exact(2)
                .map(|x| bf16::from_le_bytes(x.try_into().unwrap()).to_f32())
                .collect(),
            x => anyhow::bail!("unsupported dtype {x:?} for {name}"),
        };
        ensure!(v.iter().all(|x| x.is_finite()), "non-finite tensor {name}");
        Ok(v)
    };
    let mut out = HashMap::new();
    let mut emit = |name: String, values: Vec<f32>, half: bool| {
        let bytes = if half {
            values
                .iter()
                .flat_map(|x| f16::from_f32(*x).to_le_bytes())
                .collect()
        } else {
            values.iter().flat_map(|x| x.to_le_bytes()).collect()
        };
        out.insert(name, bytes);
    };
    emit(
        "patch_w".into(),
        get("embeddings.patch_embeddings.weight", &[384, 3, 16, 16])?,
        true,
    );
    emit(
        "patch_b".into(),
        get("embeddings.patch_embeddings.bias", &[384])?,
        false,
    );
    let mut prefix = get("embeddings.cls_token", &[1, 1, 384])?;
    prefix.extend(get("embeddings.register_tokens", &[1, 4, 384])?);
    emit("prefix".into(), prefix, false);
    for i in 0..12 {
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
                get(&(p.clone() + long), &[384])?,
                false,
            );
        }
        let mut qw = vec![];
        let mut qb = vec![];
        for q in ["q", "k", "v"] {
            qw.extend(get(&format!("{p}attention.{q}_proj.weight"), &[384, 384])?);
            qb.extend(if q == "k" {
                vec![0.; 384]
            } else {
                get(&format!("{p}attention.{q}_proj.bias"), &[384])?
            });
        }
        emit(format!("l{i}_qkv_w"), qw, true);
        emit(format!("l{i}_qkv_b"), qb, false);
        let mut gw = get(&(p.clone() + "mlp.gate_proj.weight"), &[1536, 384])?;
        gw.extend(get(&(p.clone() + "mlp.up_proj.weight"), &[1536, 384])?);
        let mut gb = get(&(p.clone() + "mlp.gate_proj.bias"), &[1536])?;
        gb.extend(get(&(p.clone() + "mlp.up_proj.bias"), &[1536])?);
        emit(format!("l{i}_gateup_w"), gw, true);
        emit(format!("l{i}_gateup_b"), gb, false);
        for (short, long, k) in [
            ("o", "attention.o_proj", 384),
            ("down", "mlp.down_proj", 1536),
        ] {
            emit(
                format!("l{i}_{short}_w"),
                get(&format!("{p}{long}.weight"), &[384, k])?,
                true,
            );
            emit(
                format!("l{i}_{short}_b"),
                get(&format!("{p}{long}.bias"), &[384])?,
                false,
            );
        }
    }
    emit("norm_w".into(), get("norm.weight", &[384])?, false);
    emit("norm_b".into(), get("norm.bias", &[384])?, false);
    let mut cos = vec![];
    let mut sin = vec![];
    for y in 0..14 {
        for x in 0..14 {
            for _ in 0..2 {
                for pos in [y, x] {
                    for j in 0..16 {
                        let coord = 2. * (pos as f64 + 0.5) / 14. - 1.;
                        let a = 2. * std::f64::consts::PI * coord / 100f64.powf(j as f64 / 16.);
                        cos.push(a.cos() as f32);
                        sin.push(a.sin() as f32);
                    }
                }
            }
        }
    }
    emit("rope_cos".into(), cos, false);
    emit("rope_sin".into(), sin, false);
    Ok(out)
}
