//! SafeTensors checkpoints with at most one owned shard buffered at a time.
use anyhow::{Context, Result, ensure};
use hrx::artifacts::safetensors::{FileView, Tensor};
use serde::Deserialize;
use std::{
    collections::{BTreeMap, BTreeSet},
    path::{Path, PathBuf},
};

#[derive(Deserialize)]
pub(crate) struct Index {
    pub weight_map: BTreeMap<String, String>,
}
impl Index {
    pub fn read(path: &Path) -> Result<Self> {
        let index: Self = serde_json::from_reader(std::fs::File::open(path)?)?;
        ensure!(!index.weight_map.is_empty(), "empty checkpoint index");
        for file in index.weight_map.values() {
            ensure!(
                Path::new(file).file_name().and_then(|n| n.to_str()) == Some(file.as_str())
                    && !file.contains(['/', '\\'])
                    && file.ends_with(".safetensors"),
                "invalid checkpoint shard filename: {file}"
            );
        }
        Ok(index)
    }
    pub fn files(&self) -> BTreeSet<&str> {
        self.weight_map.values().map(String::as_str).collect()
    }
}

pub(crate) struct Checkpoint {
    path: PathBuf,
    index: Option<Index>,
    current: Option<(PathBuf, FileView)>,
}
impl Checkpoint {
    pub fn open(path: &Path) -> Result<Self> {
        let path = if path.is_dir() {
            let index = path.join("model.safetensors.index.json");
            if index.is_file() {
                index
            } else {
                path.join("model.safetensors")
            }
        } else {
            path.to_owned()
        };
        let index = if path.extension().is_some_and(|ext| ext == "json") {
            Some(Index::read(&path)?)
        } else {
            None
        };
        Ok(Self {
            path,
            index,
            current: None,
        })
    }
    pub fn contains(&mut self, name: &str) -> Result<bool> {
        if let Some(index) = &self.index {
            return Ok(index.weight_map.contains_key(name));
        }
        if self.current.is_none() {
            self.current = Some((self.path.clone(), FileView::read(&self.path)?));
        }
        Ok(self.current.as_ref().unwrap().1.contains(name))
    }
    pub fn get(&mut self, name: &str) -> Result<Tensor<'_>> {
        let file = match &self.index {
            Some(index) => self.path.parent().unwrap_or(Path::new(".")).join(
                index
                    .weight_map
                    .get(name)
                    .with_context(|| format!("missing tensor {name} in checkpoint index"))?,
            ),
            None => self.path.clone(),
        };
        if self.current.as_ref().is_none_or(|(path, _)| path != &file) {
            // Release the previous shard before allocating the next one.
            self.current = None;
            let view = FileView::read(&file)?;
            self.current = Some((file, view));
        }
        Ok(self.current.as_ref().unwrap().1.get(name)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use safetensors::{Dtype, tensor::TensorView};
    #[test]
    fn sharded_checkpoint_switches_files_and_reports_missing_tensors() -> Result<()> {
        let dir = tempfile::tempdir()?;
        for (name, value) in [("a", 1f32), ("b", 2f32)] {
            let bytes = value.to_le_bytes();
            let tensor = TensorView::new(Dtype::F32, vec![1], &bytes)?;
            std::fs::write(
                dir.path().join(format!("{name}.safetensors")),
                safetensors::serialize([(name, tensor)], None)?,
            )?;
        }
        let path = dir.path().join("model.safetensors.index.json");
        std::fs::write(
            &path,
            r#"{"weight_map":{"a":"a.safetensors","b":"b.safetensors","absent":"b.safetensors"}}"#,
        )?;
        let mut checkpoint = Checkpoint::open(dir.path())?;
        for (name, want) in [("a", 1f32), ("b", 2f32), ("a", 1f32)] {
            let tensor = checkpoint.get(name)?;
            assert_eq!(tensor.shape, [1]);
            assert_eq!(tensor.bytes, want.to_le_bytes());
        }
        assert!(checkpoint.get("missing").is_err());
        assert!(checkpoint.get("absent").is_err());
        std::fs::remove_file(dir.path().join("b.safetensors"))?;
        assert!(Checkpoint::open(&path)?.get("b").is_err());
        Ok(())
    }
    #[test]
    fn rejects_empty_or_escaping_shard_paths() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("index.json");
        for file in [
            "../other.safetensors",
            "/tmp/other.safetensors",
            "nested/other.safetensors",
            "file.txt",
        ] {
            std::fs::write(
                &path,
                serde_json::to_vec(&serde_json::json!({"weight_map":{"a":file}}))?,
            )?;
            assert!(Index::read(&path).is_err());
        }
        std::fs::write(&path, r#"{"weight_map":{}}"#)?;
        assert!(Index::read(&path).is_err());
        Ok(())
    }
}
