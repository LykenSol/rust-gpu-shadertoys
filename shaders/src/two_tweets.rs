//! Ported to Rust from <https://www.shadertoy.com/view/MsfGzM>
//!
//! Original comment:
//! ```glsl
//! // Created by inigo quilez - iq/2013
//! // License Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License.
//! ```

use spirv_std::glam::{Vec2, Vec3, Vec4};

// Note: This cfg is incorrect on its surface, it really should be "are we compiling with std", but
// we tie #[no_std] above to the same condition, so it's fine.
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

pub struct Inputs {
    pub resolution: Vec3,
    pub time: f32,
}

impl Inputs {
    fn f(&self, mut p: Vec3) -> f32 {
        p.z += self.time;
        (Vec3::splat(0.05 * (9. * p.y * p.x).cos()) + Vec3::new(p.x.cos(), p.y.cos(), p.z.cos())
            - Vec3::splat(0.1 * (9. * (p.z + 0.3 * p.x - p.y)).cos()))
        .length()
            - 1.
    }

    pub fn main_image(&self, c: &mut Vec4, p: Vec2) {
        let d: Vec3 = Vec3::splat(0.5) - p.extend(1.0) / self.resolution.x;
        let mut o: Vec3 = d;
        let mut i = 0;
        while i < 128 {
            o += self.f(o) * d;
            i += 1;
        }
        *c = ((self.f(o - d) * Vec3::new(0.0, 1.0, 2.0)
            + Vec3::splat(self.f(o - Vec3::splat(0.6))) * Vec3::new(2.0, 1.0, 0.0))
        .abs()
            * (1. - 0.1 * o.z))
            .extend(1.0);
    }
}
