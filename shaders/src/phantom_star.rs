//! Ported to Rust from <https://www.shadertoy.com/view/ttKGDt>

use glam::{vec3, Mat2, Vec2, Vec3, Vec3Swizzles, Vec4};
use shared::*;

use core::f32::consts::PI;

// Note: This cfg is incorrect on its surface, it really should be "are we compiling with std", but
// we tie #[no_std] above to the same condition, so it's fine.
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

pub struct Inputs {
    pub resolution: Vec3,
    pub time: f32,
}

fn rot(a: f32) -> Mat2 {
    let c: f32 = a.cos();
    let s: f32 = a.sin();
    Mat2::from_cols_array(&[c, s, -s, c])
}

//const pi: f32 = (-1.0).acos();
const PI_: f32 = PI;
const PI2: f32 = PI_ * 2.0;

fn pmod(p: Vec2, r: f32) -> Vec2 {
    let mut a: f32 = p.x.atan2(p.y) + PI_ / r;
    let n: f32 = PI2 / r;
    a = (a / n).floor() * n;
    rot(-a).transpose() * p
}

fn box_(p: Vec3, b: Vec3) -> f32 {
    let d: Vec3 = p.abs() - b;
    d.x.max(d.y.max(d.z)).min(0.0) + d.max(Vec3::ZERO).length()
}

impl Inputs {
    fn ifs_box(&self, mut p: Vec3) -> f32 {
        let mut i = 0;
        while i < 5 {
            p = p.abs() - Vec3::splat(1.0);
            p = (rot(self.time * 0.3).transpose() * p.xy()).extend(p.z);
            p = (rot(self.time * 0.1).transpose() * p.xz())
                .extend(p.y)
                .xzy();
            i += 1;
        }
        p = (rot(self.time).transpose() * p.xz()).extend(p.y).xzy();
        box_(p, vec3(0.4, 0.8, 0.3))
    }

    fn map(&self, p: Vec3, _c_pos: Vec3) -> f32 {
        let mut p1: Vec3 = p;
        p1.x = (p1.x - 5.0).rem_euclid(10.0) - 5.0;
        p1.y = (p1.y - 5.0).rem_euclid(10.0) - 5.0;
        p1.z = p1.z.rem_euclid(16.0) - 8.0;
        p1 = pmod(p1.xy(), 5.0).extend(p1.z);
        self.ifs_box(p1)
    }

    pub fn main_image(&mut self, frag_color: &mut Vec4, frag_coord: Vec2) {
        let p: Vec2 =
            (frag_coord * 2.0 - self.resolution.xy()) / self.resolution.x.min(self.resolution.y);

        let c_pos: Vec3 = vec3(0.0, 0.0, -3.0 * self.time);
        // let c_pos: Vec3 = vec3(0.3 * (self.time * 0.8).sin(), 0.4 * (self.time * 0.3).cos(), -6.0 * self.time,);
        let c_dir: Vec3 = vec3(0.0, 0.0, -1.0).normalize();
        let c_up: Vec3 = vec3(self.time.sin(), 1.0, 0.0);
        let c_side: Vec3 = c_dir.cross(c_up);

        let ray: Vec3 = (c_side * p.x + c_up * p.y + c_dir).normalize();

        // Phantom Mode https://www.shadertoy.com/view/MtScWW by aiekick
        let mut acc: f32 = 0.0;
        let mut acc2: f32 = 0.0;
        let mut t: f32 = 0.0;

        let mut i = 0;
        while i < 99 {
            let pos: Vec3 = c_pos + ray * t;
            let mut dist: f32 = self.map(pos, c_pos);
            dist = dist.abs().max(0.02);
            let mut a: f32 = (-dist * 3.0).exp();
            if (pos.length() + 24.0 * self.time).rem_euclid(30.0) < 3.0 {
                a *= 2.0;
                acc2 += a;
            }
            acc += a;
            t += dist * 0.5;
            i += 1;
        }

        let col: Vec3 = vec3(
            acc * 0.01,
            acc * 0.011 + acc2 * 0.002,
            acc * 0.012 + acc2 * 0.005,
        );
        *frag_color = col.extend(1.0 - t * 0.03);
    }
}
