//! Compile-time architecture descriptions for the supported checkpoints.
mod sealed {
    pub trait Sealed {}
}

/// A supported architecture. Sealed so tensor and kernel contracts stay consistent.
pub trait ModelSpec: sealed::Sealed + Send + Sync + 'static {
    const NAME: &'static str;
    const HIDDEN: usize;
    const HEADS: usize;
    const LAYERS: usize = 12;
    const INTERMEDIATE: usize;
    const GATED: bool;
    const REPO: &'static str;
    const REVISION: &'static str;
    /// One unnormalized CLS or patch-mean row, with a statically known width.
    type Row: bytemuck::Pod + bytemuck::Zeroable + AsRef<[f32]> + Send + Sync;
}

/// DINOv3 ViT-S+/16 with SwiGLU feed-forward layers.
pub enum ViTS16Plus {}
/// DINOv3 ViT-B/16 with GELU feed-forward layers.
pub enum ViTB16 {}
impl sealed::Sealed for ViTS16Plus {}
impl sealed::Sealed for ViTB16 {}
impl ModelSpec for ViTS16Plus {
    const NAME: &'static str = "vits16plus";
    const HIDDEN: usize = 384;
    const HEADS: usize = 6;
    const INTERMEDIATE: usize = 1536;
    const GATED: bool = true;
    const REPO: &'static str = "dinov3-vits16plus-pretrain-lvd1689m";
    const REVISION: &'static str = "c93d816fc9e567563bc068f01475bec89cc634a6";
    type Row = [f32; 384];
}
impl ModelSpec for ViTB16 {
    const NAME: &'static str = "vitb16";
    const HIDDEN: usize = 768;
    const HEADS: usize = 12;
    const INTERMEDIATE: usize = 3072;
    const GATED: bool = false;
    const REPO: &'static str = "dinov3-vitb16-pretrain-lvd1689m";
    const REVISION: &'static str = "5931719e67bbdb9737e363e781fb0c67687896bc";
    type Row = [f32; 768];
}
