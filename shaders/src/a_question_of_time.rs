//! Ported to Rust from <https://www.shadertoy.com/view/lljfRD>
//!
//! Original comment:
//! ```glsl
//! // Author: Rigel rui@gil.com
//! // licence: https://creativecommons.org/licenses/by/4.0/
//! // link: https://www.shadertoy.com/view/lljfRD
//!
//!
//! /*
//! This was a study on circles, inspired by this artwork
//! http://www.dailymail.co.uk/news/article-1236380/Worlds-largest-artwork-etched-desert-sand.html
//!
//! and implemented with the help of this article
//! http://www.ams.org/samplings/feature-column/fcarc-kissing
//!
//! The structure is called an apollonian packing (or gasket)
//! https://en.m.wikipedia.org/wiki/Apollonian_gasket
//!
//! There is a lot of apollonians in shadertoy, but not many quite like the image above.
//! This one by klems is really cool. He uses a technique called a soddy circle.
//! https://www.shadertoy.com/view/4s2czK
//!
//! This shader uses another technique called a Descartes Configuration.
//! The only thing that makes this technique interesting is that it can be generalized to higher dimensions.
//! */
//! ```

use glam::{vec2, vec3, Mat2, Vec2, Vec2Swizzles, Vec3, Vec3Swizzles, Vec4, Vec4Swizzles};
use shared::*;

// Note: This cfg is incorrect on its surface, it really should be "are we compiling with std", but
// we tie #[no_std] above to the same condition, so it's fine.
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

pub struct Inputs {
    pub resolution: Vec3,
    pub time: f32,
    pub mouse: Vec4,
}

// a few utility functions
// a signed distance function for a rectangle
fn sdf_rect(uv: Vec2, s: Vec2) -> f32 {
    let auv: Vec2 = uv.abs();
    (auv.x - s.x).max(auv.y - s.y)
}
// a signed distance function for a circle
fn sdf_circle(uv: Vec2, c: Vec2, r: f32) -> f32 {
    (uv - c).length() - r
}
// fills an sdf in 2d
fn fill(d: f32, s: f32, i: f32) -> f32 {
    (smoothstep(0.0, s, d) - i).abs()
}
// makes a stroke of an sdf at the zero boundary
fn stroke(d: f32, w: f32, s: f32, i: f32) -> f32 {
    (smoothstep(0.0, s, d.abs() - (w * 0.5)) - i).abs()
}
// a simple palette
fn pal(d: f32) -> Vec3 {
    0.5 * ((6.283 * d * vec3(2.0, 2.0, 1.0) + vec3(0.0, 1.4, 0.0)).cos() + Vec3::one())
}
// 2d rotation matrix
fn uvr_rotate(a: f32) -> Mat2 {
    Mat2::from_cols_array(&[a.cos(), a.sin(), -a.sin(), a.cos()])
}
// circle inversion
fn inversion(uv: Vec2, r: f32) -> Vec2 {
    (r * r * uv) / Vec2::splat(uv.dot(uv))
}
// seeded random number
fn hash(s: Vec2) -> f32 {
    ((s.dot(vec2(12.9898, 78.2333))).sin() * 43758.5453123).gl_fract()
}

// this is an algorithm to construct an apollonian packing with a descartes configuration
// remaps the plane to a circle at the origin and a specific radius. vec3(x,y,radius)
fn apollonian(uv: Vec2) -> Vec3 {
    // the algorithm is recursive and must start with a initial descartes configuration
    // each vec3 represents a circle with the form vec3(centerx, centery, 1./radius)
    // the signed inverse radius is also called the bend (refer to the article above)
    let mut dec: [Vec3; 4] = [Vec3::zero(), Vec3::zero(), Vec3::zero(), Vec3::zero()];
    // a DEC is a configuration of 4 circles tangent to each other
    // the easiest way to build the initial one it to construct a symetric Steiner Chain.
    // http://mathworld.wolfram.com/SteinerChain.html
    let a: f32 = 6.283 / 3.;
    let ra: f32 = 1.0 + (a * 0.5).sin();
    let rb: f32 = 1.0 - (a * 0.5).sin();
    dec[0] = vec3(0.0, 0.0, -1.0 / ra);
    let radius: f32 = 0.5 * (ra - rb);
    let bend: f32 = 1.0 / radius;
    let mut i = 1;
    while i < 4 {
        dec[i] = vec3((i as f32 * a).cos(), (i as f32 * a).sin(), bend);
        // if the point is in one of the starting circles we have already found our solution
        if (uv - dec[i].xy()).length() < radius {
            return (uv - dec[i].xy()).extend(radius);
        }
        i += 1;
    }

    // Now that we have a starting DEC we are going to try to
    // find the solution for the current point
    let mut i = 0;
    while i < 7 {
        // find the circle that is further away from the point uv, using euclidean distance
        let mut fi: usize = 0;
        let mut d: f32 = uv.distance(dec[0].xy()) - (1.0 / dec[0].z).abs();
        // for some reason, the euclidean distance doesn't work for the circle with negative bend
        // can anyone with proper math skills, explain me why?
        d *= if dec[0].z < 0.0 { -0.5 } else { 1.0 }; // just scale it to make it work...
        let mut j = 1;
        while j < 4 {
            let mut fd: f32 = uv.distance(dec[j].xy()) - (1. / dec[j].z).abs();
            fd *= if dec[j].z < 0.0 { -0.5 } else { 1.0 };
            if fd > d {
                fi = j;
                d = fd;
            }
            j += 1;
        }
        // put the cicle found in the last slot, to generate a solution
        // in the "direction" of the point
        let c: Vec3 = dec[3];
        dec[3] = dec[fi];
        dec[fi] = c;
        // generate a new solution
        let bend: f32 = (2.0 * (dec[0].z + dec[1].z + dec[2].z)) - dec[3].z;
        let center: Vec2 = (2.0
            * (dec[0].z * dec[0].xy() + dec[1].z * dec[1].xy() + dec[2].z * dec[2].xy())
            - dec[3].z * dec[3].xy())
            / bend;

        let solution: Vec3 = center.extend(bend);
        // is the solution radius is to small, quit
        if (1. / bend).abs() < 0.01 {
            break;
        }
        // if the solution contains the point return the circle
        if (uv - solution.xy()).length() < 1. / bend {
            return (uv - solution.xy()).extend(1. / bend);
        }
        // else update the descartes configuration,
        dec[3] = solution;
        // and repeat...
        i += 1;
    }
    // if nothing is found we return by default the inner circle of the Steiner chain
    uv.extend(rb)
}

impl Inputs {
    fn scene(&self, mut uv: Vec2, ms: Vec4) -> Vec3 {
        let mut ci: Vec2 = Vec2::zero();

        // drag your mouse to apply circle inversion
        if ms.y != -2.0 && ms.w > -2.0 {
            uv = inversion(uv, (60.0.deg_to_radians()).cos());
            ci = ms.xy();
        }

        // remap uv to appolonian packing
        let uv_apo: Vec3 = apollonian(uv - ci);

        let d: f32 = 6.2830 / 360.0;
        let a: f32 = uv_apo.y.atan2(uv_apo.x);
        let r: f32 = uv_apo.xy().length();

        let circle: f32 = sdf_circle(uv, uv - uv_apo.xy(), uv_apo.z);

        // background
        let mut c: Vec3 = uv.length() * pal(0.7) * 0.2;

        // drawing the clocks
        if uv_apo.z > 0.3 {
            c = mix(c, pal(0.75 - r * 0.1) * 0.8, fill(circle + 0.02, 0.01, 1.0)); // clock
            c = mix(
                c,
                pal(0.4 + r * 0.1),
                stroke(circle + (uv_apo.z * 0.03), uv_apo.z * 0.01, 0.005, 1.),
            ); // dial
            let h: f32 = stroke(
                (a + d * 15.0).rem_euclid(d * 30.) - d * 15.0,
                0.02,
                0.01,
                1.0,
            );
            c = mix(
                c,
                pal(0.4 + r * 0.1),
                h * stroke(circle + (uv_apo.z * 0.16), uv_apo.z * 0.25, 0.005, 1.0),
            ); // hours

            let m: f32 = stroke(
                (a + d * 15.0).rem_euclid(d * 6.0) - d * 3.0,
                0.005,
                0.01,
                1.0,
            );
            c = mix(
                c,
                pal(0.45 + r * 0.1),
                (1.0 - h) * m * stroke(circle + (uv_apo.z * 0.15), uv_apo.z * 0.1, 0.005, 1.0),
            ); // minutes,

            // needles rotation
            let uvrh: Vec2 = uvr_rotate(
                (hash(Vec2::splat(uv_apo.z)) * d * 180.0).cos().gl_sign()
                    * d
                    * self.time
                    * (1.0 / uv_apo.z * 10.0)
                    - d * 90.0,
            )
            .transpose()
                * uv_apo.xy();
            let uvrm: Vec2 = uvr_rotate(
                (hash(Vec2::splat(uv_apo.z) * 4.0) * d * 180.0)
                    .cos()
                    .gl_sign()
                    * d
                    * self.time
                    * (1.0 / uv_apo.z * 120.0)
                    - d * 90.0,
            )
            .transpose()
                * uv_apo.xy();
            // draw needles
            c = mix(
                c,
                pal(0.85),
                stroke(
                    sdf_rect(
                        uvrh + vec2(uv_apo.z - (uv_apo.z * 0.8), 0.0),
                        uv_apo.z * vec2(0.4, 0.03),
                    ),
                    uv_apo.z * 0.01,
                    0.005,
                    1.0,
                ),
            );
            c = mix(
                c,
                pal(0.9),
                fill(
                    sdf_rect(
                        uvrm + vec2(uv_apo.z - (uv_apo.z * 0.65), 0.0),
                        uv_apo.z * vec2(0.5, 0.002),
                    ),
                    0.005,
                    1.0,
                ),
            );
            c = mix(
                c,
                pal(0.5 + r * 10.0),
                fill(circle + uv_apo.z - 0.02, 0.005, 1.0),
            ); // center
               // drawing the gears
        } else if uv_apo.z > 0.05 {
            let uvrg: Vec2 = uvr_rotate(
                (hash(Vec2::splat(uv_apo.z + 2.0)) * d * 180.0)
                    .cos()
                    .gl_sign()
                    * d
                    * self.time
                    * (1.0 / uv_apo.z * 20.0),
            )
            .transpose()
                * uv_apo.xy();
            let g: f32 = stroke(
                (uvrg.y.atan2(uvrg.x) + d * 22.5).rem_euclid(d * 45.) - d * 22.5,
                0.3,
                0.05,
                1.0,
            );
            let size: Vec2 = uv_apo.z * vec2(0.45, 0.08);
            c = mix(
                c,
                pal(0.55 - r * 0.6),
                fill(circle + g * (uv_apo.z * 0.2) + 0.01, 0.001, 1.)
                    * fill(circle + (uv_apo.z * 0.6), 0.005, 0.0),
            );
            c = mix(
                c,
                pal(0.55 - r * 0.6),
                fill(
                    sdf_rect(uvrg, size).min(sdf_rect(uvrg, size.yx())),
                    0.005,
                    1.,
                ),
            );
        // drawing the screws
        } else {
            let size: Vec2 = uv_apo.z * vec2(0.5, 0.1);
            c = mix(
                c,
                pal(0.85 - (uv_apo.z * 2.0)),
                fill(circle + 0.01, 0.007, 1.0),
            );
            c = mix(
                c,
                pal(0.8 - (uv_apo.z * 3.)),
                fill(
                    sdf_rect(uv_apo.xy(), size).min(sdf_rect(uv_apo.xy(), size.yx())),
                    0.002,
                    1.0,
                ),
            );
        }
        c
    }

    pub fn main_image(&mut self, frag_color: &mut Vec4, frag_coord: Vec2) {
        let uv: Vec2 = (frag_coord - self.resolution.xy() * 0.5) / self.resolution.y;
        let ms: Vec4 = (self.mouse - self.resolution.xyxy() * 0.5) / self.resolution.y;
        *frag_color = self.scene(uv * 4., ms * 4.).extend(1.0);
    }
}
