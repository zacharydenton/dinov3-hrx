use anyhow::{Result, ensure};
use std::{
    collections::HashMap,
    path::{Path, PathBuf},
};
const T: usize = 201;

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
    let h = w.len();
    let mut out = x.to_vec();
    for (src, dst) in x.chunks(h).zip(out.chunks_mut(h)) {
        let mean = src.iter().sum::<f64>() / h as f64;
        let variance = src.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / h as f64;
        for c in 0..h {
            dst[c] = (src[c] - mean) / (variance + 1e-5).sqrt() * w[c] + b[c];
        }
    }
    out
}
// The reference buffers one checkpoint shard and one layer of F64 weights.
// SafeTensors parsing is independent of the production loader.
struct Reader {
    files: HashMap<String, PathBuf>,
    current: PathBuf,
    bytes: Vec<u8>,
}
impl Reader {
    fn open(path: &Path) -> Result<Self> {
        let mut reader = Self {
            files: HashMap::new(),
            current: PathBuf::new(),
            bytes: vec![],
        };
        if path.extension().is_some_and(|s| s == "json") {
            let index: serde_json::Value = serde_json::from_slice(&std::fs::read(path)?)?;
            for (name, file) in index["weight_map"].as_object().unwrap() {
                reader.files.insert(
                    name.clone(),
                    path.parent().unwrap().join(file.as_str().unwrap()),
                );
            }
        } else {
            reader.current = path.to_owned();
            reader.bytes = std::fs::read(path)?;
            for name in safetensors::SafeTensors::deserialize(&reader.bytes)?.names() {
                reader.files.insert(name.to_string(), path.to_owned());
            }
        }
        Ok(reader)
    }
    fn prefix(&mut self, prefix: &str) -> Result<HashMap<String, Vec<f64>>> {
        let mut result = HashMap::new();
        let mut names: Vec<_> = self
            .files
            .keys()
            .filter(|s| s.starts_with(prefix))
            .cloned()
            .collect();
        names.sort();
        for name in names {
            let path = &self.files[&name];
            if &self.current != path {
                self.bytes.clear();
                self.bytes.shrink_to_fit();
                self.bytes = std::fs::read(path)?;
                self.current = path.clone();
            }
            let model = safetensors::SafeTensors::deserialize(&self.bytes)?;
            let t = model.tensor(&name)?;
            ensure!(
                t.dtype() == safetensors::Dtype::F32,
                "reference expects original float32 model"
            );
            result.insert(
                name,
                t.data()
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|v| f32::from_le_bytes(*v) as f64)
                    .collect(),
            );
        }
        Ok(result)
    }
}
pub fn forward(path: &Path, image: &[f32], variant: &str) -> Result<Vec<f64>> {
    // Deliberately separate from the production ModelSpec constants.
    let (h, layers, heads, intermediate, gated, qv_bias) = match variant {
        "vits16" => (384, 12, 6, 1536, false, true),
        "vits16plus" => (384, 12, 6, 1536, true, true),
        "vitb16" => (768, 12, 12, 3072, false, true),
        "vitl16" => (1024, 24, 16, 4096, false, true),
        "vith16plus" => (1280, 32, 20, 5120, true, true),
        "vit7b16" => (4096, 40, 32, 8192, true, false),
        _ => anyhow::bail!("unknown reference architecture"),
    };
    let d = h / heads;
    let mut reader = Reader::open(path)?;
    let embeddings = reader.prefix("embeddings.")?;
    let w = |s: &str| embeddings[s].as_slice();
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
        h,
    ));
    for layer in 0..layers {
        let p = format!("layer.{layer}.");
        let weights = reader.prefix(&p)?;
        let get = |suffix: &str| weights[&(p.clone() + suffix)].as_slice();
        let normalized = norm(&x, get("norm1.weight"), get("norm1.bias"));
        let mut q = linear(
            &normalized,
            get("attention.q_proj.weight"),
            if qv_bias {
                Some(get("attention.q_proj.bias"))
            } else {
                None
            },
            h,
            h,
        );
        let mut k = linear(&normalized, get("attention.k_proj.weight"), None, h, h);
        let v = linear(
            &normalized,
            get("attention.v_proj.weight"),
            if qv_bias {
                Some(get("attention.v_proj.bias"))
            } else {
                None
            },
            h,
            h,
        );
        for values in [&mut q, &mut k] {
            for token in 5..T {
                let patch = token - 5;
                for head in 0..heads {
                    let base = token * h + head * d;
                    let original = values[base..base + d].to_vec();
                    for c in 0..d {
                        let pos = if c % (d / 2) < d / 4 {
                            patch / 14
                        } else {
                            patch % 14
                        };
                        let angle =
                            2. * std::f64::consts::PI * (2. * (pos as f64 + 0.5) / 14. - 1.)
                                / 100f64.powf((c % (d / 4)) as f64 / (d / 4) as f64);
                        let rotated = if c < d / 2 {
                            -original[c + d / 2]
                        } else {
                            original[c - d / 2]
                        };
                        values[base + c] = original[c] * angle.cos() + rotated * angle.sin();
                    }
                }
            }
        }
        let mut context = vec![0.; T * h];
        let mut scores = vec![0.; T];
        for head in 0..heads {
            for row in 0..T {
                for col in 0..T {
                    scores[col] = (0..d)
                        .map(|c| q[row * h + head * d + c] * k[col * h + head * d + c])
                        .sum::<f64>()
                        / (d as f64).sqrt();
                }
                let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                for s in &mut scores {
                    *s = (*s - max).exp();
                }
                let sum = scores.iter().sum::<f64>();
                for col in 0..T {
                    let a = scores[col] / sum;
                    for c in 0..d {
                        context[row * h + head * d + c] += a * v[col * h + head * d + c];
                    }
                }
            }
        }
        let o = linear(
            &context,
            get("attention.o_proj.weight"),
            Some(get("attention.o_proj.bias")),
            h,
            h,
        );
        for i in 0..x.len() {
            x[i] += o[i] * get("layer_scale1.lambda1")[i % h];
        }
        let normalized = norm(&x, get("norm2.weight"), get("norm2.bias"));
        let gate = if gated {
            let mut gate = linear(
                &normalized,
                get("mlp.gate_proj.weight"),
                Some(get("mlp.gate_proj.bias")),
                h,
                intermediate,
            );
            let up = linear(
                &normalized,
                get("mlp.up_proj.weight"),
                Some(get("mlp.up_proj.bias")),
                h,
                intermediate,
            );
            for (g, u) in gate.iter_mut().zip(up) {
                *g = *g / (1. + (-*g).exp()) * u;
            }
            gate
        } else {
            linear(
                &normalized,
                get("mlp.up_proj.weight"),
                Some(get("mlp.up_proj.bias")),
                h,
                intermediate,
            )
            .into_iter()
            .map(|v| 0.5 * v * (1. + libm::erf(v / std::f64::consts::SQRT_2)))
            .collect()
        };
        let down = linear(
            &gate,
            get("mlp.down_proj.weight"),
            Some(get("mlp.down_proj.bias")),
            intermediate,
            h,
        );
        for i in 0..x.len() {
            x[i] += down[i] * get("layer_scale2.lambda1")[i % h];
        }
    }
    let weights = reader.prefix("norm.")?;
    Ok(norm(&x, &weights["norm.weight"], &weights["norm.bias"]))
}
