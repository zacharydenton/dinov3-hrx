//! Pinned model weights in the shared Hugging Face cache.

use anyhow::Result;
use hrx::artifacts::hf::{HubFile, Repository, Resolver};
use std::path::PathBuf;

/// Hugging Face model repository.
pub const REPO: (&str, &str) = ("facebook", "dinov3-vits16plus-pretrain-lvd1689m");
/// Revision whose weights are covered by this crate's numerical tests.
pub const REVISION: &str = "c93d816fc9e567563bc068f01475bec89cc634a6";
/// Original model file within the repository.
pub const FILE: &str = "model.safetensors";

/// Resolve the pinned weights, downloading only on a cache miss.
pub fn weights(offline: bool) -> Result<PathBuf> {
    Ok(Resolver::new(Repository::new(REPO.0, REPO.1).at(REVISION))
        .offline(offline)
        .resolve(&HubFile::new(FILE))?)
}
