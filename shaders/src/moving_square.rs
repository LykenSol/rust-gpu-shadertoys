//! Ported to Rust from <https://www.shadertoy.com/view/llXSzX>

use glam::{vec3, Mat2, Vec2, Vec3, Vec3Swizzles, Vec4};
use shared::*;

// Note: This cfg is incorrect on its surface, it really should be "are we compiling with std", but
// we tie #[no_std] above to the same condition, so it's fine.
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

pub struct Inputs {
    pub resolution: Vec3,
    pub time: f32,
}

fn rect(uv: Vec2, pos: Vec2, r: f32) -> Vec4 {
    let re_c: Vec2 = (uv - pos).abs();
    let dif1: Vec2 = re_c - Vec2::splat(r / 2.);
    let dif2: Vec2 = (re_c - Vec2::splat(r / 2.)).clamp(Vec2::zero(), Vec2::one());
    let d1: f32 = (dif1.x + dif1.y).clamp(0.0, 1.0);
    let _d2: f32 = (dif2.x + dif2.y).clamp(0.0, 1.0);

    Vec4::splat(d1)
}

impl Inputs {
    pub fn main_image(&self, frag_color: &mut Vec4, frag_coord: Vec2) {
        let mut uv: Vec2 = frag_coord;
        let t: f32 = self.time.sin();

        let c: Vec2 = self.resolution.xy() * 0.5; // + sin(iTime) * 50.;

        uv = Mat2::from_cols_array(&[t.cos(), -t.sin(), t.sin(), t.cos()]) * (uv - c) + c;

        *frag_color = rect(uv, c, (self.time * 10.).sin() * 50. + 50.);
        *frag_color *= vec3(0.5, 0.2, 1.).extend(1.);
        *frag_color += rect(uv, c, self.time.sin() * 50. + 50.);
        *frag_color *= vec3(0.5, 0.8, 1.).extend(1.);
    }
}
