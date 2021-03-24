//! Ported to Rust from <https://www.shadertoy.com/view/MslSDN>
//!
//! Original comment:
//! ```glsl
//! // Created by Sebastien Durand - 2014
//! // License Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License.
//! ```

use shared::*;
use glam::{
    const_vec3, vec2, vec3, Mat2, Vec2, Vec2Swizzles, Vec3, Vec3Swizzles, Vec4, Vec4Swizzles,
};

// Note: This cfg is incorrect on its surface, it really should be "are we compiling with std", but
// we tie #[no_std] above to the same condition, so it's fine.
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

pub struct Inputs {
    pub resolution: Vec3,
    pub time: f32,
    pub mouse: Vec4,
}

pub struct State {
    inputs: Inputs,

    a: [Vec2; 15],
    t1: [Vec2; 5],
    t2: [Vec2; 5],

    l: Vec3,

    t_morph: f32,
    mat2_rot: Mat2,
}

impl State {
    pub fn new(inputs: Inputs) -> Self {
        State {
            inputs,

            a: [
                Vec2::zero(),
                Vec2::zero(),
                Vec2::zero(),
                Vec2::zero(),
                Vec2::zero(),
                Vec2::zero(),
                Vec2::zero(),
                Vec2::zero(),
                Vec2::zero(),
                Vec2::zero(),
                Vec2::zero(),
                Vec2::zero(),
                Vec2::zero(),
                Vec2::zero(),
                Vec2::zero(),
            ],
            t1: [
                Vec2::zero(),
                Vec2::zero(),
                Vec2::zero(),
                Vec2::zero(),
                Vec2::zero(),
            ],
            t2: [
                Vec2::zero(),
                Vec2::zero(),
                Vec2::zero(),
                Vec2::zero(),
                Vec2::zero(),
            ],

            l: vec3(1.0, 0.72, 1.0).normalize(),

            t_morph: 0.0,
            mat2_rot: Mat2::zero(),
        }
    }
}

fn u(a: Vec2, b: Vec2) -> f32 {
    a.x * b.y - b.x * a.y
}

const Y: Vec3 = const_vec3!([0.0, 1.0, 0.0]);
// const E: Vec3 = Y * 0.01;
const _E: Vec3 = const_vec3!([0.0, 0.01, 0.0]);

// Distance to Bezier
// inspired by [iq:https://www.shadertoy.com/view/ldj3Wh]
// calculate distance to 2D bezier curve on xy but without forgeting the z component of p
// total distance is corrected using pytagore just before return
fn bezier(mut m: Vec2, mut n: Vec2, mut o: Vec2, p: Vec3) -> Vec2 {
    let q: Vec2 = p.xy();
    m -= q;
    n -= q;
    o -= q;
    let x: f32 = u(m, o);
    let y: f32 = 2.0 * u(n, m);
    let z: f32 = 2.0 * u(o, n);
    let i: Vec2 = o - m;
    let j: Vec2 = o - n;
    let k: Vec2 = n - m;
    let s: Vec2 = 2. * (x * i + y * j + z * k);
    let mut r: Vec2 = m + (y * z - x * x) * vec2(s.y, -s.x) / s.dot(s);
    let t: f32 = ((u(r, i) + 2.0 * u(k, r)) / (x + x + y + z)).clamp(0.0, 1.0); // parametric position on curve
    r = m + t * (k + k + t * (j - k)); // distance on 2D xy space
    vec2((r.dot(r) + p.z * p.z).sqrt(), t) // distance on 3D space
}

fn smin(a: f32, b: f32, k: f32) -> f32 {
    let h: f32 = (0.5 + 0.5 * (b - a) / k).clamp(0.0, 1.0);
    mix(b, a, h) - k * h * (1. - h)
}

impl State {
    // Distance to scene
    fn m(&self, mut p: Vec3) -> f32 {
        // Distance to Teapot ---------------------------------------------------
        // precalcul first part of teapot spout
        let h: Vec2 = bezier(self.t1[2], self.t1[3], self.t1[4], p);
        let mut a: f32 = 99.0;
        // distance to teapot handle (-.06 => make the thickness)
        let b: f32 = (bezier(self.t2[0], self.t2[1], self.t2[2], p)
            .x
            .min(bezier(self.t2[2], self.t2[3], self.t2[4], p).x)
            - 0.06)
            // max p.y-.9 => cut the end of the spout
            .min(
                (p.y - 0.9).max(
                    // distance to second part of teapot spout (abs(dist,r1)-dr) => enable to make the spout hole
                    ((bezier(self.t1[0], self.t1[1], self.t1[2], p).x - 0.07).abs() - 0.01)
                        // distance to first part of teapot spout (tickness incrase with pos on curve)
                        .min(h.x * (1. - 0.75 * h.y) - 0.08),
                ),
            );

        // distance to teapot body => use rotation symetry to simplify calculation to a distance to 2D bezier curve
        let qq: Vec3 = vec3((p.dot(p) - p.y * p.y).sqrt(), p.y, 0.0);
        // the substraction of .015 enable to generate a small thickness arround bezier to help convergance
        // the .8 factor help convergance
        let mut i = 0;
        while i < 13 {
            a = a.min((bezier(self.a[i], self.a[i + 1], self.a[i + 2], qq).x - 0.015) * 0.7);
            i += 2;
        }
        // smooth minimum to improve quality at junction of handle and spout to the body
        let d_teapot: f32 = smin(a, b, 0.02);

        // Distance to other shapes ---------------------------------------------
        let mut d_shape: f32;
        let id_morph: i32 = ((0.5 + (self.inputs.time) / (2.0 * 3.141592658)).floor() % 3.0) as i32;

        if id_morph == 1 {
            p = (self.mat2_rot.transpose() * p.xz()).extend(p.y).xzy();
            let d: Vec3 = (p - vec3(0.0, 0.5, 0.0)).abs() - vec3(0.8, 0.7, 0.8);
            d_shape = d.x.max(d.y.max(d.z)).min(0.0) + d.max(Vec3::zero()).length();
        } else if id_morph == 2 {
            p -= vec3(0.0, 0.55, 0.0);
            let d1: Vec3 = p.abs() - vec3(0.67, 0.67, 0.67 * 1.618);
            let d3: Vec3 = p.abs() - vec3(0.67 * 1.618, 0.67, 0.67);
            d_shape = d1.x.max(d1.y.max(d1.z)).min(0.0) + d1.max(Vec3::zero()).length();
            d_shape =
                d_shape.min(d3.x.max(d3.y.max(d3.z)).min(0.0) + d3.max(Vec3::zero()).length());
        } else {
            d_shape = (p - vec3(0.0, 0.45, 0.0)).length() - 1.1;
        }

        // !!! The morphing is here !!!
        mix(d_teapot, d_shape, self.t_morph.abs())
    }
}

// HSV to RGB conversion
// [iq: https://www.shadertoy.com/view/MsS3Wc]
fn hsv2rgb_smooth(x: f32, y: f32, z: f32) -> Vec3 {
    let mut rgb: Vec3 =
        (((x * Vec3::splat(6.0) + vec3(0.0, 4.0, 2.0)).rem_euclid(6.0) - Vec3::splat(3.0)).abs()
            - Vec3::one())
        .clamp(Vec3::zero(), Vec3::one());
    rgb = rgb * rgb * (Vec3::splat(3.0) - 2.0 * rgb); // cubic smoothing
    z * mix(Vec3::one(), rgb, y)
}

impl State {
    fn normal(&self, p: Vec3, ray: Vec3, t: f32) -> Vec3 {
        let pitch: f32 = 0.4 * t / self.inputs.resolution.x;
        let d: Vec2 = vec2(-1.0, 1.0) * pitch;
        // tetrahedral offsets
        let p0: Vec3 = p + d.xxx();
        let p1: Vec3 = p + d.xyy();
        let p2: Vec3 = p + d.yxy();
        let p3: Vec3 = p + d.yyx();
        let f0: f32 = self.m(p0);
        let f1: f32 = self.m(p1);
        let f2: f32 = self.m(p2);
        let f3: f32 = self.m(p3);
        let grad: Vec3 = p0 * f0 + p1 * f1 + p2 * f2 + p3 * f3 - p * (f0 + f1 + f2 + f3);
        // prevent normals pointing away from camera (caused by precision errors)
        (grad - (grad.dot(ray).max(0.0)) * ray).normalize()
    }

    pub fn main_image(&mut self, frag_color: &mut Vec4, frag_coord: Vec2) {
        let aa: f32 = 3.14159 / 4.0;
        self.mat2_rot = Mat2::from_cols_array(&[aa.cos(), aa.sin(), -aa.sin(), aa.cos()]);

        // Morphing step
        self.t_morph = (self.inputs.time * 0.5).cos();
        self.t_morph *= self.t_morph * self.t_morph * self.t_morph * self.t_morph;

        // Teapot body profil (8 quadratic curves)
        self.a[0] = vec2(0.0, 0.0);
        self.a[1] = vec2(0.64, 0.0);
        self.a[2] = vec2(0.64, 0.03);
        self.a[3] = vec2(0.8, 0.12);
        self.a[4] = vec2(0.8, 0.3);
        self.a[5] = vec2(0.8, 0.48);
        self.a[6] = vec2(0.64, 0.9);
        self.a[7] = vec2(0.6, 0.93);
        self.a[8] = vec2(0.56, 0.9);
        self.a[9] = vec2(0.56, 0.96);
        self.a[10] = vec2(0.12, 1.02);
        self.a[11] = vec2(0.0, 1.05);
        self.a[12] = vec2(0.16, 1.14);
        self.a[13] = vec2(0.2, 1.2);
        self.a[14] = vec2(0.0, 1.2);
        // Teapot spout (2 quadratic curves)
        self.t1[0] = vec2(1.16, 0.96);
        self.t1[1] = vec2(1.04, 0.9);
        self.t1[2] = vec2(1.0, 0.72);
        self.t1[3] = vec2(0.92, 0.48);
        self.t1[4] = vec2(0.72, 0.42);
        // Teapot handle (2 quadratic curves)
        self.t2[0] = vec2(-0.6, 0.78);
        self.t2[1] = vec2(-1.16, 0.84);
        self.t2[2] = vec2(-1.16, 0.63);
        self.t2[3] = vec2(-1.2, 0.42);
        self.t2[4] = vec2(-0.72, 0.24);

        // Configure camera
        let r: Vec2 = self.inputs.resolution.xy();
        let m: Vec2 = self.inputs.mouse.xy() / r;
        let q: Vec2 = frag_coord / r;
        let mut p: Vec2 = q + q - Vec2::one();
        p.x *= r.x / r.y;
        let mut j: f32 = 0.0;
        let mut s: f32 = 1.0;
        let mut h: f32 = 0.1;
        let mut t: f32 = 5.0 + 0.2 * self.inputs.time + 4.0 * m.x;
        let o: Vec3 = 2.9 * vec3(t.cos(), 0.7 - m.y, t.sin());
        let w: Vec3 = (Y * 0.4 - o).normalize();
        let u: Vec3 = w.cross(Y).normalize();
        let v: Vec3 = u.cross(w);
        let d: Vec3 = (p.x * u + p.y * v + w + w).normalize();
        let n: Vec3;
        let x: Vec3;

        // Ray marching
        t = 0.0;
        let mut i = 0;
        while i < 48 {
            if h < 0.0001 || t > 4.7 {
                break;
            }
            h = self.m(o + d * t);
            t += h;
            i += 1;
        }

        // Background colour change as teapot complementaries colours (using HSV)
        let mut c: Vec3 = mix(
            hsv2rgb_smooth(0.5 + self.inputs.time * 0.02, 0.35, 0.4),
            hsv2rgb_smooth(-0.5 + self.inputs.time * 0.02, 0.35, 0.7),
            q.y,
        );

        // Calculate color on point
        if h < 0.001 {
            x = o + t * d;
            n = self.normal(x, d, t); //normalize(vec3(M(x+E.yxx)-M(x-E.yxx),M(x+E)-M(x-E),M(x+E.xxy)-M(x-E.xxy)));

            // Calculate Shadows
            let mut i = 0;
            while i < 20 {
                j += 0.02;
                s = s.min(self.m(x + self.l * j) / j);
                i += 1;
            }
            // Teapot color rotation in HSV color space
            let c1: Vec3 = hsv2rgb_smooth(0.9 + self.inputs.time * 0.02, 1.0, 1.0);
            // Shading
            c = mix(
                c,
                mix(
                    (((3.0 * s).clamp(0.0, 1.0) + 0.3) * c1).sqrt(),
                    Vec3::splat(self.l.reflect(n).dot(d).max(0.0).powf(99.0)),
                    0.4,
                ),
                2.0 * n.dot(-d),
            );
        }

        c *= (16.0 * q.x * q.y * (1.0 - q.x) * (1.0 - q.y)).powf(0.16); // Vigneting
        *frag_color = c.extend(1.0);
    }
}
