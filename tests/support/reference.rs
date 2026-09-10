use anyhow::{Result, ensure};
use std::{collections::HashMap, path::Path};
const T: usize = 201;
const H: usize = 384;
fn linear(x: &[f64], w: &[f64], bias: Option<&[f64]>, k: usize, n: usize) -> Vec<f64> {
    let m = x.len() / k;
    assert_eq!(w.len(), n * k);
    let mut out = vec![0.; m * n];
    // All matrix extents and row/column strides are bounded by the slices above.
    unsafe {
        matrixmultiply::dgemm(
            m,
            k,
            n,
            1.,
            x.as_ptr(),
            k as isize,
            1,
            w.as_ptr(),
            1,
            k as isize,
            0.,
            out.as_mut_ptr(),
            n as isize,
            1,
        );
    }
    if let Some(b) = bias {
        assert_eq!(b.len(), n);
        for row in out.chunks_mut(n) {
            for (v, b) in row.iter_mut().zip(b) {
                *v += b;
            }
        }
    }
    out
}
fn norm(x: &[f64], w: &[f64], b: &[f64]) -> Vec<f64> {
    let mut out = x.to_vec();
    for (src, dst) in x.chunks(H).zip(out.chunks_mut(H)) {
        let mean = src.iter().sum::<f64>() / H as f64;
        let variance = src.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / H as f64;
        for c in 0..H {
            dst[c] = (src[c] - mean) / (variance + 1e-5).sqrt() * w[c] + b[c];
        }
    }
    out
}
pub fn forward(path: &Path, image: &[f32]) -> Result<Vec<f64>> {
    let file = std::fs::read(path)?;
    let model = safetensors::SafeTensors::deserialize(&file)?;
    let mut weights = HashMap::new();
    for (name, t) in model.tensors() {
        ensure!(
            t.dtype() == safetensors::Dtype::F32,
            "reference expects original float32 model"
        );
        weights.insert(
            name,
            t.data()
                .chunks_exact(4)
                .map(|v| f32::from_le_bytes(v.try_into().unwrap()) as f64)
                .collect::<Vec<_>>(),
        );
    }
    let w = |s: &str| weights[s].as_slice();
    let mut patches = vec![];
    for y in 0..14 {
        for x in 0..14 {
            for c in 0..3 {
                for dy in 0..16 {
                    for dx in 0..16 {
                        patches
                            .push(image[c * 224 * 224 + (y * 16 + dy) * 224 + x * 16 + dx] as f64);
                    }
                }
            }
        }
    }
    let mut x = w("embeddings.cls_token").to_vec();
    x.extend(w("embeddings.register_tokens"));
    x.extend(linear(
        &patches,
        w("embeddings.patch_embeddings.weight"),
        Some(w("embeddings.patch_embeddings.bias")),
        768,
        H,
    ));
    for layer in 0..12 {
        let p = format!("layer.{layer}.");
        let get = |suffix: &str| w(&(p.clone() + suffix));
        let h = norm(&x, get("norm1.weight"), get("norm1.bias"));
        let mut q = linear(
            &h,
            get("attention.q_proj.weight"),
            Some(get("attention.q_proj.bias")),
            H,
            H,
        );
        let mut k = linear(&h, get("attention.k_proj.weight"), None, H, H);
        let v = linear(
            &h,
            get("attention.v_proj.weight"),
            Some(get("attention.v_proj.bias")),
            H,
            H,
        );
        for values in [&mut q, &mut k] {
            for token in 5..T {
                let patch = token - 5;
                for head in 0..6 {
                    let base = token * H + head * 64;
                    let original = values[base..base + 64].to_vec();
                    for c in 0..64 {
                        let pos = if c % 32 < 16 { patch / 14 } else { patch % 14 };
                        let angle =
                            2. * std::f64::consts::PI * (2. * (pos as f64 + 0.5) / 14. - 1.)
                                / 100f64.powf((c % 16) as f64 / 16.);
                        let rotated = if c < 32 {
                            -original[c + 32]
                        } else {
                            original[c - 32]
                        };
                        values[base + c] = original[c] * angle.cos() + rotated * angle.sin();
                    }
                }
            }
        }
        let mut context = vec![0.; T * H];
        let mut scores = vec![0.; T];
        for head in 0..6 {
            for row in 0..T {
                for col in 0..T {
                    scores[col] = (0..64)
                        .map(|c| q[row * H + head * 64 + c] * k[col * H + head * 64 + c])
                        .sum::<f64>()
                        * 0.125;
                }
                let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                for s in &mut scores {
                    *s = (*s - max).exp();
                }
                let sum = scores.iter().sum::<f64>();
                for col in 0..T {
                    let a = scores[col] / sum;
                    for c in 0..64 {
                        context[row * H + head * 64 + c] += a * v[col * H + head * 64 + c];
                    }
                }
            }
        }
        let o = linear(
            &context,
            get("attention.o_proj.weight"),
            Some(get("attention.o_proj.bias")),
            H,
            H,
        );
        for i in 0..x.len() {
            x[i] += o[i] * get("layer_scale1.lambda1")[i % H];
        }
        let h = norm(&x, get("norm2.weight"), get("norm2.bias"));
        let mut gate = linear(
            &h,
            get("mlp.gate_proj.weight"),
            Some(get("mlp.gate_proj.bias")),
            H,
            1536,
        );
        let up = linear(
            &h,
            get("mlp.up_proj.weight"),
            Some(get("mlp.up_proj.bias")),
            H,
            1536,
        );
        for (g, u) in gate.iter_mut().zip(up) {
            *g = *g / (1. + (-*g).exp()) * u;
        }
        let down = linear(
            &gate,
            get("mlp.down_proj.weight"),
            Some(get("mlp.down_proj.bias")),
            1536,
            H,
        );
        for i in 0..x.len() {
            x[i] += down[i] * get("layer_scale2.lambda1")[i % H];
        }
    }
    Ok(norm(&x, w("norm.weight"), w("norm.bias")))
}
