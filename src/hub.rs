//! Pinned model weights in the shared Hugging Face cache.

use crate::{ModelSpec, ViTS16Plus};
use anyhow::Result;
use hrx::artifacts::hf::{HubFile, Repository, Resolver};
use std::path::PathBuf;

/// Hugging Face model repository.
pub const REPO: (&str, &str) = ("facebook", ViTS16Plus::REPO);
/// Revision whose weights are covered by this crate's numerical tests.
pub const REVISION: &str = ViTS16Plus::REVISION;
/// Original model file within the repository.
pub const FILE: &str = "model.safetensors";

/// Resolve the pinned weights, downloading only on a cache miss.
pub fn weights(offline: bool) -> Result<PathBuf> {
    weights_for::<ViTS16Plus>(offline)
}

/// Resolve the pinned checkpoint for a compile-time model architecture.
/// Returns a SafeTensors file, or an index after fetching every required shard.
pub fn weights_for<M: ModelSpec>(offline: bool) -> Result<PathBuf> {
    let resolver =
        Resolver::new(Repository::new("facebook", M::REPO).at(M::REVISION)).offline(offline);
    if M::SHARDED {
        let path = resolver.resolve(&HubFile::new("model.safetensors.index.json"))?;
        let index = crate::checkpoint::Index::read(&path)?;
        for file in index.files() {
            resolver.resolve(&HubFile::new(file))?;
        }
        Ok(path)
    } else {
        Ok(resolver.resolve(&HubFile::new(FILE))?)
    }
}
