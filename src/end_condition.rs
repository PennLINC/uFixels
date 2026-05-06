/// Why a vertex's inward walk terminated. Encoded as `f32` in the output GIFTI.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EndCondition {
    NoCompatibleFixel = 0,
    HitSurface = 1,
    MaxDepth = 2,
    LeftMask = 3,
}

impl EndCondition {
    pub fn as_f32(self) -> f32 {
        self as i32 as f32
    }
}
