//! Compile-time architecture descriptions for the supported checkpoints.
mod sealed {
    pub trait Sealed {}
}

/// A supported architecture. Sealed so tensor and kernel contracts stay consistent.
pub trait ModelSpec: sealed::Sealed + Send + Sync + 'static {
    const NAME: &'static str;
    const HIDDEN: usize;
    const HEADS: usize;
    const HEAD_DIM: usize = Self::HIDDEN / Self::HEADS;
    /// Large models contain register-token residuals outside the finite F16 range.
    const RESIDUAL_F32: bool = Self::HIDDEN >= 1024;
    const QV_BIAS: bool = true;
    const SHARDED: bool = false;
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

/// DINOv3 vits16 architecture with 384-feature outputs.
pub enum ViTS16 {}
impl sealed::Sealed for ViTS16 {}
impl ModelSpec for ViTS16 {
    const NAME: &'static str = "vits16";
    const HIDDEN: usize = 384;
    const HEADS: usize = 6;
    const LAYERS: usize = 12;
    const INTERMEDIATE: usize = 1536;
    const GATED: bool = false;
    const QV_BIAS: bool = true;
    const SHARDED: bool = false;
    const REPO: &'static str = "dinov3-vits16-pretrain-lvd1689m";
    const REVISION: &'static str = "114c1379950215c8b35dfcd4e90a5c251dde0d32";
    type Row = [f32; 384];
}

/// DINOv3 vitl16 architecture with 1024-feature outputs.
pub enum ViTL16 {}
impl sealed::Sealed for ViTL16 {}
impl ModelSpec for ViTL16 {
    const NAME: &'static str = "vitl16";
    const HIDDEN: usize = 1024;
    const HEADS: usize = 16;
    const LAYERS: usize = 24;
    const INTERMEDIATE: usize = 4096;
    const GATED: bool = false;
    const QV_BIAS: bool = true;
    const SHARDED: bool = false;
    const REPO: &'static str = "dinov3-vitl16-pretrain-lvd1689m";
    const REVISION: &'static str = "ea8dc2863c51be0a264bab82070e3e8836b02d51";
    type Row = [f32; 1024];
}

/// DINOv3 vith16plus architecture with 1280-feature outputs.
pub enum ViTH16Plus {}
impl sealed::Sealed for ViTH16Plus {}
impl ModelSpec for ViTH16Plus {
    const NAME: &'static str = "vith16plus";
    const HIDDEN: usize = 1280;
    const HEADS: usize = 20;
    const LAYERS: usize = 32;
    const INTERMEDIATE: usize = 5120;
    const GATED: bool = true;
    const QV_BIAS: bool = true;
    const SHARDED: bool = false;
    const REPO: &'static str = "dinov3-vith16plus-pretrain-lvd1689m";
    const REVISION: &'static str = "c807c9eeea853df70aec4069e6f56b28ddc82acc";
    type Row = [f32; 1280];
}

/// DINOv3 vit7b16 architecture with 4096-feature outputs.
pub enum ViT7B16 {}
impl sealed::Sealed for ViT7B16 {}
impl ModelSpec for ViT7B16 {
    const NAME: &'static str = "vit7b16";
    const HIDDEN: usize = 4096;
    const HEADS: usize = 32;
    const LAYERS: usize = 40;
    const INTERMEDIATE: usize = 8192;
    const GATED: bool = true;
    const QV_BIAS: bool = false;
    const SHARDED: bool = true;
    const REPO: &'static str = "dinov3-vit7b16-pretrain-lvd1689m";
    const REVISION: &'static str = "b80367753773648a6793235ab9c65cdbb029506f";
    type Row = [f32; 4096];
}
