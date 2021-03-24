//! Ported to Rust from <https://www.shadertoy.com/view/4df3Rn>
//!
//! Original comment:
//! ```glsl
//! // Created by inigo quilez - iq/2013
//! // License Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License.
//!
//! // See here for more information on smooth iteration count:
//! //
//! // http://iquilezles.org/www/articles/mset_smooth/mset_smooth.htm
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

// increase this if you have a very fast GPU
const AA: usize = 2;

impl Inputs {
    fn mandelbrot(&self, c: Vec2) -> f32 {
        if true {
            let c2: f32 = c.dot(c);
            // skip computation inside M1 - http://iquilezles.org/www/articles/mset_1bulb/mset1bulb.htm
            if 256.0 * c2 * c2 - 96.0 * c2 + 32.0 * c.x - 3.0 < 0.0 {
                return 0.0;
            }
            // skip computation inside M2 - http://iquilezles.org/www/articles/mset_2bulb/mset2bulb.htm
            if (16.0 * (c2 + 2.0 * c.x + 1.0) - 1.0) < 0.0 {
                return 0.0;
            }
        }
        const B: f32 = 256.0;
        let mut l: f32 = 0.0;
        let mut z: Vec2 = Vec2::zero();
        let mut i = 0;
        while i < 512 {
            z = vec2(z.x * z.x - z.y * z.y, 2.0 * z.x * z.y) + c;
            if z.dot(z) > (B * B) {
                break;
            }
            l += 1.0;
            i += 1;
        }

        if l > 511.0 {
            return 0.0;
        }
        // ------------------------------------------------------
        // smooth interation count
        //float sl = l - log(log(length(z))/log(B))/log(2.0);
        //
        // equivalent optimized smooth interation count
        let sl: f32 = l - z.dot(z).log2().log2() + 4.0;

        let al: f32 = smoothstep(-0.1, 0.0, (self.time * 0.5 * 6.2831).sin());
        mix(l, sl, al)
    }

    pub fn main_image(&self, frag_color: &mut Vec4, frag_coord: Vec2) {
        let mut col: Vec3 = Vec3::zero();

        let mut m = 0;
        while m < AA {
            let mut n = 0;
            while n < AA {
                let p: Vec2 = (-self.resolution.xy()
                    + Vec2::splat(2.0) * (frag_coord + vec2(m as f32, n as f32) / AA as f32))
                    / self.resolution.y;
                let w: f32 = (AA * m + n) as f32;
                let time: f32 = self.time + 0.5 * (1.0 / 24.0) * w / (AA * AA) as f32;

                let mut zoo: f32 = 0.62 + 0.38 * (0.07 * time).cos();
                let coa: f32 = (0.15 * (1.0 - zoo) * time).cos();
                let sia = (0.15 * (1.0 - zoo) * time).sin();
                zoo = zoo.powf(8.0);
                let xy: Vec2 = vec2(p.x * coa - p.y * sia, p.x * sia + p.y * coa);
                let c: Vec2 = vec2(-0.745, 0.186) + xy * zoo;

                let l: f32 = self.mandelbrot(c);
                col += Vec3::splat(0.5)
                    + Vec3::splat(0.5) * (Vec3::splat(3.0 + l * 0.15) + vec3(0.0, 0.6, 1.0)).cos();
                n += 1;
            }
            m += 1;
        }
        col /= (AA * AA) as f32;
        *frag_color = col.extend(1.0);
    }
}
