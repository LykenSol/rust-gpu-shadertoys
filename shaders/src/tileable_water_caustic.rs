//! Ported to Rust from <https://www.shadertoy.com/view/MdlXz8>
//!
//! Original comment:
//! ```glsl
//! // Found this on GLSL sandbox. I really liked it, changed a few things and made it tileable.
//! // :)
//! // by David Hoskins.
//!
//!
//! // Water turbulence effect by joltz0r 2013-07-04, improved 2013-07-07
//! ```

use glam::{vec2, vec3, Vec2, Vec3, Vec3Swizzles, Vec4};
use shared::*;

// Note: This cfg is incorrect on its surface, it really should be "are we compiling with std", but
// we tie #[no_std] above to the same condition, so it's fine.
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

pub struct Inputs {
    pub resolution: Vec3,
    pub time: f32,
}

// Redefine below to see the tiling...
const SHOW_TILING: bool = false;

const TAU: f32 = 6.28318530718;
const MAX_ITER: usize = 5;

impl Inputs {
    pub fn main_image(&self, frag_color: &mut Vec4, frag_coord: Vec2) {
        let time: f32 = self.time * 0.5 + 23.0;
        // uv should be the 0-1 uv of texture...
        let mut uv: Vec2 = frag_coord / self.resolution.xy();

        let p: Vec2 = if SHOW_TILING {
            (uv * TAU * 2.0).rem_euclid(TAU) - Vec2::splat(250.0)
        } else {
            (uv * TAU).rem_euclid(TAU) - Vec2::splat(250.0)
        };
        let mut i: Vec2 = p;
        let mut c: f32 = 1.0;
        let inten: f32 = 0.005;

        let mut n = 0;
        while n < MAX_ITER {
            let t: f32 = time * (1.1 - (3.5 / (n + 1) as f32));
            i = p + vec2(
                (t - i.x).cos() + (t + i.y).sin(),
                (t - i.y).sin() + (t + i.x).cos(),
            );
            c += 1.0
                / vec2(
                    p.x / ((i.x + t).sin() / inten),
                    p.y / ((i.y + t).cos() / inten),
                )
                .length();
            n += 1;
        }
        c /= MAX_ITER as f32;
        c = 1.17 - c.powf(1.4);
        let mut colour: Vec3 = Vec3::splat(c.abs().powf(8.0));
        colour = (colour + vec3(0.0, 0.35, 0.5)).clamp(Vec3::zero(), Vec3::one());

        if SHOW_TILING {
            let pixel: Vec2 = 2.0 / self.resolution.xy();
            uv *= 2.0;

            let f: f32 = (self.time * 0.5).rem_euclid(2.0).floor(); // Flash value.
            let first: Vec2 = pixel.step(uv) * f; // Rule out first screen pixels and flash.
            uv = uv.gl_fract().step(pixel); // Add one line of pixels per tile.
            colour = mix(
                colour,
                vec3(1.0, 1.0, 0.0),
                (uv.x + uv.y) * first.x * first.y,
            ); // Yellow line
        }

        *frag_color = colour.extend(1.0);
    }
}
