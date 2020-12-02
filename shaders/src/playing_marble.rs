//! Ported to Rust from <https://www.shadertoy.com/view/4ds3zn>
//!
//! Original comment:
//! ```glsl
//! // License Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License.
//! // Created by S. Guillitte 2015
//! ```

use crate::Channel;
use shared::*;
use spirv_std::glam::{Mat2, Vec2, Vec3, Vec3Swizzles, Vec4, Vec4Swizzles};

// Note: This cfg is incorrect on its surface, it really should be "are we compiling with std", but
// we tie #[no_std] above to the same condition, so it's fine.
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

pub struct Inputs<C0> {
    pub resolution: Vec3,
    pub time: f32,
    pub mouse: Vec4,
    pub channel0: C0,
}

const ZOOM: f32 = 1.0;

fn _cmul(a: Vec2, b: Vec2) -> Vec2 {
    Vec2::new(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x)
}
fn csqr(a: Vec2) -> Vec2 {
    Vec2::new(a.x * a.x - a.y * a.y, 2. * a.x * a.y)
}

fn rot(a: f32) -> Mat2 {
    Mat2::from_cols_array(&[a.cos(), a.sin(), -a.sin(), a.cos()])
}

//from iq
fn i_sphere(ro: Vec3, rd: Vec3, sph: Vec4) -> Vec2 {
    let oc: Vec3 = ro - sph.xyz();
    let b: f32 = oc.dot(rd);
    let c: f32 = oc.dot(oc) - sph.w * sph.w;
    let mut h: f32 = b * b - c;
    if h < 0.0 {
        return Vec2::splat(-1.0);
    }
    h = h.sqrt();
    Vec2::new(-b - h, -b + h)
}

fn map(mut p: Vec3) -> f32 {
    let mut res: f32 = 0.0;
    let c: Vec3 = p;
    let mut i = 0;
    while i < 10 {
        p = 0.7 * p.abs() / p.dot(p) - Vec3::splat(0.7);
        p = csqr(p.yz()).extend(p.x).zxy();
        p = p.zxy();
        res += (-19.0 * p.dot(c).abs()).exp();
        i += 1;
    }
    res / 2.0
}

impl<C0: Channel> Inputs<C0> {
    fn raymarch(&self, ro: Vec3, rd: Vec3, tminmax: Vec2) -> Vec3 {
        let mut t: f32 = tminmax.x;
        let dt: f32 = 0.02;
        //let dt: f32 = 0.2 - 0.195 * (self.time * 0.05).cos(); //animated
        let mut col: Vec3 = Vec3::zero();
        let mut c: f32 = 0.0;
        let mut i = 0;
        while i < 64 {
            t += dt * (-2.0 * c).exp();
            if t > tminmax.y {
                break;
            }
            let _pos: Vec3 = ro + t * rd;

            c = map(ro + t * rd);

            col = 0.99 * col + 0.08 * Vec3::new(c * c, c, c * c * c); //green

            // col = 0.99 * col + 0.08 * Vec3::new(c * c * c, c * c, c); //blue
            i += 1;
        }
        col
    }
    pub fn main_image(&mut self, frag_color: &mut Vec4, frag_coord: Vec2) {
        let time: f32 = self.time;
        let q: Vec2 = frag_coord / self.resolution.xy();
        let mut p: Vec2 = Vec2::splat(-1.0) + 2.0 * q;
        p.x *= self.resolution.x / self.resolution.y;
        let mut m: Vec2 = Vec2::zero();
        if self.mouse.z > 0.0 {
            m = self.mouse.xy() / self.resolution.xy() * 3.14;
        }
        m = m - Vec2::splat(0.5);

        // camera

        let mut ro: Vec3 = ZOOM * Vec3::splat(4.0);
        ro = (rot(m.y).transpose() * ro.yz()).extend(ro.x).zxy();
        ro = (rot(m.x + 0.1 * time).transpose() * ro.xz())
            .extend(ro.y)
            .xzy();
        let ta: Vec3 = Vec3::zero();
        let ww: Vec3 = (ta - ro).normalize();
        let uu: Vec3 = (ww.cross(Vec3::new(0.0, 1.0, 0.0))).normalize();
        let vv: Vec3 = (uu.cross(ww)).normalize();
        let rd: Vec3 = (p.x * uu + p.y * vv + 4.0 * ww).normalize();

        let tmm: Vec2 = i_sphere(ro, rd, Vec4::new(0.0, 0.0, 0.0, 2.0));
        // raymarch
        let mut col: Vec3 = self.raymarch(ro, rd, tmm);
        if tmm.x < 0.0 {
            col = self.channel0.sample_cube(rd).xyz();
        } else {
            let mut nor: Vec3 = (ro + tmm.x * rd) / 2.;
            nor = rd.reflect(nor);
            let fre: f32 = (0.5 + nor.dot(rd).clamp(0.0, 1.0)).powf(3.0) * 1.3;
            col += self.channel0.sample_cube(nor).xyz() * fre;
        }

        //shade

        col = 0.5 * (Vec3::one() + col).ln();
        col = col.clamp(Vec3::zero(), Vec3::one());

        *frag_color = col.extend(1.0);
    }
}
