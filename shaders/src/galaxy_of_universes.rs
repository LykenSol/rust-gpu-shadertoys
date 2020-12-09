//! Ported to Rust from <https://www.shadertoy.com/view/MdXSzS>
//!
//! Original comment:
//! ```glsl
//! // https://www.shadertoy.com/view/MdXSzS
//! // The Big Bang - just a small explosion somewhere in a massive Galaxy of Universes.
//! // Outside of this there's a massive galaxy of 'Galaxy of Universes'... etc etc. :D
//!
//! // To fake a perspective it takes advantage of the screen being wider than it is tall.
//! ```

use shared::*;
use spirv_std::glam::{vec3, Mat2, Vec2, Vec3, Vec3Swizzles, Vec4};

// Note: This cfg is incorrect on its surface, it really should be "are we compiling with std", but
// we tie #[no_std] above to the same condition, so it's fine.
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

pub struct Inputs {
    pub resolution: Vec3,
    pub time: f32,
}

impl Inputs {
    pub fn main_image(&self, frag_color: &mut Vec4, frag_coord: Vec2) {
        let uv: Vec2 = (frag_coord / self.resolution.xy()) - Vec2::splat(0.5);
        let t: f32 = self.time * 0.1
            + ((0.25 + 0.05 * (self.time * 0.1).sin()) / (uv.length() + 0.07)) * 2.2;
        let si: f32 = t.sin();
        let co: f32 = t.cos();
        let ma: Mat2 = Mat2::from_cols_array(&[co, si, -si, co]);

        let mut v1: f32 = 0.0;
        let mut v2: f32 = 0.0;
        let mut v3: f32 = 0.0;

        let mut s: f32 = 0.0;

        let mut i = 0;
        while i < 90 {
            let mut p: Vec3 = s * uv.extend(0.0);
            p = (ma.transpose() * p.xy()).extend(p.z);
            p += vec3(0.22, 0.3, s - 1.5 - (self.time * 0.13).sin() * 0.1);
            {
                let mut i = 0;
                while i < 8 {
                    p = p.abs() / p.dot(p) - Vec3::splat(0.659);
                    i += 1;
                }
            }
            v1 += p.dot(p) * 0.0015 * (1.8 + ((uv * 13.0).length() + 0.5 - self.time * 0.2).sin());
            v2 += p.dot(p) * 0.0013 * (1.5 + ((uv * 14.5).length() + 1.2 - self.time * 0.3).sin());
            v3 += (p.xy() * 10.0).length() * 0.0003;
            s += 0.035;
            i += 1;
        }

        let len: f32 = uv.length();
        v1 *= smoothstep(0.7, 0.0, len);
        v2 *= smoothstep(0.5, 0.0, len);
        v3 *= smoothstep(0.9, 0.0, len);

        let col: Vec3 = vec3(
            v3 * (1.5 + (self.time * 0.2).sin() * 0.4),
            (v1 + v3) * 0.3,
            v2,
        ) + Vec3::splat(smoothstep(0.2, 0.0, len) * 0.85)
            + Vec3::splat(smoothstep(0.0, 0.6, v3) * 0.3);

        *frag_color = col
            .abs()
            .powf_vec(Vec3::splat(1.2))
            .min(Vec3::splat(1.0))
            .extend(1.0);
    }
}
