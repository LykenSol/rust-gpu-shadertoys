//! Ported to Rust from <https://www.shadertoy.com/view/XsBSRV>
//!
//! Original comment:
//! ```glsl
//! // On/Off Spikes, fragment shader by movAX13h, oct 2014
//! ```

use glam::{
    const_vec3, vec2, vec3, Mat2, Mat3, Vec2, Vec2Swizzles, Vec3, Vec3Swizzles, Vec4, Vec4Swizzles,
};
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

pub struct State {
    inputs: Inputs,

    // globals
    glow: f32,
    bite: f32,
    sphere_col: Vec3,
    sun: Vec3,
    focus: f32,
    far: f32,
}

impl State {
    pub fn new(inputs: Inputs) -> State {
        State {
            inputs,

            glow: 0.0,
            bite: 0.0,
            sphere_col: Vec3::zero(),
            sun: SUN_POS.normalize(),
            focus: 5.0,
            far: 23.0,
        }
    }
}

const HARD_SHADOW: bool = true;
const GLOW: bool = true;
const EDGES: bool = true;
const NUM_TENTACLES: i32 = 6;
const BUMPS: bool = true;
const NUM_BUMPS: i32 = 8;
const BACKGROUND: bool = true;
const SUN_POS: Vec3 = const_vec3!([15.0, 15.0, -15.0]);
const SUN_SPHERE: bool = false;

const SPHERE_COL: Vec3 = const_vec3!([0.6, 0.3, 0.1]);
const MOUTH_COL: Vec3 = const_vec3!([0.9, 0.6, 0.1]);
const TENTACLE_COL: Vec3 = const_vec3!([0.06; 3]);

const GAMMA: f32 = 2.2;

//---
const PI2: f32 = 6.283185307179586476925286766559;
const PIH: f32 = 1.5707963267949;

// Using the nebula function of the "Star map shader" by morgan3d
// as environment map and light sphere texture (https://www.shadertoy.com/view/4sBXzG)
const _PI: f32 = 3.1415927;
const NUM_OCTAVES: i32 = 4;
fn hash(n: f32) -> f32 {
    (n.sin() * 1e4).gl_fract()
}
fn hash_vec2(p: Vec2) -> f32 {
    (1e4 * (17.0 * p.x + p.y * 0.1).sin() * (0.1 + (p.y * 13.0 + p.x).sin().abs())).gl_fract()
}
fn noise(x: f32) -> f32 {
    let i: f32 = x.floor();
    let f: f32 = x.gl_fract();
    let u: f32 = f * f * (3.0 - 2.0 * f);
    mix(hash(i), hash(i + 1.0), u)
}
fn noise_vec2(x: Vec2) -> f32 {
    let i: Vec2 = x.floor();
    let f: Vec2 = x.gl_fract();
    let a: f32 = hash_vec2(i);
    let b: f32 = hash_vec2(i + vec2(1.0, 0.0));
    let c: f32 = hash_vec2(i + vec2(0.0, 1.0));
    let d: f32 = hash_vec2(i + vec2(1.0, 1.0));
    let u: Vec2 = f * f * (Vec2::splat(3.0) - 2.0 * f);
    mix(a, b, u.x) + (c - a) * u.y * (1.0 - u.x) + (d - b) * u.x * u.y
}
fn noise2(mut x: Vec2) -> f32 {
    let mut v: f32 = 0.0;
    let mut a: f32 = 0.5;
    let shift: Vec2 = Vec2::splat(100.0);
    let rot: Mat2 =
        Mat2::from_cols_array(&[0.5_f32.cos(), 0.5_f32.sin(), -0.5_f32.sin(), 0.50_f32.cos()]);
    let mut i = 0;
    while i < NUM_OCTAVES {
        v += a * noise_vec2(x);
        x = rot * x * 2.0 + shift;
        a *= 0.5;
        i += 1;
    }
    v
}
fn square(x: f32) -> f32 {
    x * x
}
fn rotation(yaw: f32, pitch: f32) -> Mat3 {
    Mat3::from_cols_array(&[
        yaw.cos(),
        0.0,
        -yaw.sin(),
        0.0,
        1.0,
        0.0,
        yaw.sin(),
        0.0,
        yaw.cos(),
    ]) * Mat3::from_cols_array(&[
        1.0,
        0.0,
        0.0,
        0.0,
        pitch.cos(),
        pitch.sin(),
        0.0,
        -pitch.sin(),
        pitch.cos(),
    ])
}
fn nebula(dir: Vec3) -> Vec3 {
    let purple: f32 = dir.x.abs();
    let yellow: f32 = noise(dir.y);
    let streaky_hue: Vec3 = vec3(purple + yellow, yellow * 0.7, purple);
    let puffy_hue: Vec3 = vec3(0.8, 0.1, 1.0);
    let streaky: f32 = 1.0_f32.min(
        8.0 * (noise2(
            dir.yz() * square(dir.x) * 13.0 + dir.xy() * square(dir.z) * 7.0 + vec2(150.0, 2.0),
        ))
        .powf(10.0),
    );
    let puffy: f32 = square(noise2(dir.xz() * 4.0 + vec2(30.0, 10.0)) * dir.y);

    (puffy_hue * puffy * (1.0 - streaky) + streaky * streaky_hue)
        .clamp(Vec3::zero(), Vec3::one())
        .powf(1.0 / 2.2)
}
// ---

fn sd_box(p: Vec3, b: Vec3) -> f32 {
    let d: Vec3 = p.abs() - b;
    d.x.max(d.y.max(d.z)).min(0.0) + d.max(Vec3::zero()).length()
}

fn sd_sphere(p: Vec3, r: f32) -> f32 {
    p.length() - r
}

fn sd_capped_cylinder(p: Vec3, h: Vec2) -> f32 {
    let d: Vec2 = vec2(p.xy().length(), p.z).abs() - h;
    d.x.max(d.y).min(0.0) + d.max(Vec2::zero()).length()
}

fn rotate(p: Vec2, a: f32) -> Vec2 {
    let mut r: Vec2 = Vec2::zero();
    r.x = p.x * a.cos() - p.y * a.sin();
    r.y = p.x * a.sin() + p.y * a.cos();
    r
}

// polynomial smooth min (k = 0.1); by iq
fn smin(a: f32, b: f32, k: f32) -> f32 {
    let h: f32 = (0.5 + 0.5 * (b - a) / k).clamp(0.0, 1.0);
    mix(b, a, h) - k * h * (1.0 - h)
}

#[derive(Clone, Copy, Default)]
struct Hit {
    d: f32,
    color: Vec3,
    edge: f32,
}

impl State {
    fn scene(&self, p: Vec3) -> Hit {
        let mut d: f32;
        let mut d1: f32;
        let mut d2: f32;
        let d3: f32;
        let f: f32;
        let mut e: f32 = 0.15;

        let mut q: Vec3 = p;
        q = rotate(q.xy(), 1.5).extend(q.z);

        // center sphere
        d1 = sd_sphere(q, 0.3);
        // d = d1;
        let mut col: Vec3 = self.sphere_col;

        // tentacles
        let r: f32 = q.length();
        let mut a: f32 = q.z.atan2(q.x);
        a += 0.4 * (r - self.inputs.time).sin();

        q = vec3(a * NUM_TENTACLES as f32 / PI2, q.y, q.xz().length()); // circular domain
        q = vec3(q.x.rem_euclid(1.0) - 0.5 * 1.0, q.y, q.z); // repetition

        d3 = sd_capped_cylinder(
            q - vec3(0.0, 0.0, 0.9 + self.bite),
            vec2(0.1 - (r - self.bite) / 18.0, 0.8),
        );
        d2 = d3.min(sd_box(
            q - vec3(0.0, 0.0, 0.1 + self.bite),
            vec3(0.2, 0.2, 0.2),
        )); // close box
        d2 = smin(
            d2,
            sd_box(q - vec3(0.0, 0.0, 0.4 + self.bite), vec3(0.2, 0.05, 0.4)),
            0.1,
        ); // wide box

        f = smoothstep(0.11, 0.28, d2 - d1);
        col = mix(MOUTH_COL, col, f);
        e = mix(e, 0.0, f);
        d = smin(d1, d2, 0.24);

        col = mix(TENTACLE_COL, col, smoothstep(0.0, 0.48, d3 - d));

        if SUN_SPHERE {
            d = d.min(sd_sphere(p - self.sun, 0.1));
        }

        if BUMPS {
            let mut i = 0;
            while i < NUM_BUMPS {
                d2 = i as f32;
                d1 = sd_sphere(
                    p - 0.18
                        * smoothstep(0.1, 1.0, self.glow)
                        * vec3(
                            (4.0 * self.inputs.time + d2 * 0.6).sin(),
                            (5.3 * self.inputs.time + d2 * 1.4).sin(),
                            (5.8 * self.inputs.time + d2 * 0.6).cos(),
                        ),
                    0.03,
                );

                d = smin(d1, d, 0.2);
                //d = min(d1, d);
                i += 1;
            }
        }

        if BACKGROUND {
            q = p;
            q = q.yz().rem_euclid(1.0).extend(q.x).zxy();
            q -= vec3(-0.6, 0.5, 0.5);
            d1 = sd_box(q, vec3(0.1, 0.48, 0.48));
            if d1 < d {
                d = d1;
                col = Vec3::splat(0.1);
            }
        }

        Hit {
            d,
            color: col,
            edge: e,
        }
    }

    fn normal(&self, p: Vec3) -> Vec3 {
        let c: f32 = self.scene(p).d;
        let h: Vec2 = vec2(0.01, 0.0);
        vec3(
            self.scene(p + h.xyy()).d - c,
            self.scene(p + h.yxy()).d - c,
            self.scene(p + h.yyx()).d - c,
        )
        .normalize()
    }

    // by srtuss
    fn edges(&self, p: Vec3) -> f32 {
        let mut acc: f32 = 0.0;
        let h: f32 = 0.01;
        acc += self.scene(p + vec3(-h, -h, -h)).d;
        acc += self.scene(p + vec3(-h, -h, h)).d;
        acc += self.scene(p + vec3(-h, h, -h)).d;
        acc += self.scene(p + vec3(-h, h, h)).d;
        acc += self.scene(p + vec3(h, -h, -h)).d;
        acc += self.scene(p + vec3(h, -h, h)).d;
        acc += self.scene(p + vec3(h, h, -h)).d;
        acc += self.scene(p + vec3(h, h, h)).d;
        acc / h
    }

    fn colorize(&self, hit: Hit, n: Vec3, dir: Vec3, light_pos: Vec3) -> Vec3 {
        let diffuse: f32 = 0.3 * n.dot(light_pos).max(0.0);

        let ref_: Vec3 = dir.reflect(n).normalize();
        let specular: f32 = 0.4 * ref_.dot(light_pos).max(0.0).powf(6.5);

        hit.color + diffuse * Vec3::splat(0.9) + specular * Vec3::splat(1.0)
    }

    pub fn main_image(&mut self, frag_color: &mut Vec4, frag_coord: Vec2) {
        //self.time = self.inputs.time;
        self.glow = (2.0 * (self.inputs.time * 0.7 - 5.0).sin())
            .min(1.0)
            .max(0.0);
        self.bite = smoothstep(0.0, 1.0, 1.6 * (self.inputs.time * 0.7).sin());
        self.sphere_col = SPHERE_COL * self.glow;

        let pos: Vec2 = (frag_coord * 2.0 - self.inputs.resolution.xy()) / self.inputs.resolution.y;

        let d: f32 = (1.5 * (0.3 * self.inputs.time).sin()).clamp(0.5, 1.0);
        let mut cp: Vec3 = vec3(
            10.0 * d,
            -2.3 * d,
            -6.2 * d + 4.0 * (2.0 * (self.inputs.time * 0.5).sin().clamp(0.0, 1.0)),
        ); // anim curious spectator

        if self.inputs.mouse.z > 0.5 {
            let mrel: Vec2 =
                self.inputs.mouse.xy() / self.inputs.resolution.xy() - Vec2::splat(0.5);
            let mdis: f32 = 8.0 + 6.0 * mrel.y;
            cp = vec3(
                mdis * (-mrel.x * PIH).cos(),
                4.0 * mrel.y,
                mdis * (-mrel.x * PIH).sin(),
            );
        }

        let ct: Vec3 = vec3(0.0, 0.0, 0.0);
        let cd: Vec3 = (ct - cp).normalize();
        let cu: Vec3 = vec3(0.0, 1.0, 0.0);
        let cs: Vec3 = cd.cross(cu);
        let mut dir: Vec3 = (cs * pos.x + cu * pos.y + cd * self.focus).normalize();

        let mut h: Hit = Hit::default();
        let mut col: Vec3;
        let mut ray: Vec3 = cp;
        let mut dist: f32 = 0.0;

        // raymarch scene
        let mut i = 0;
        while i < 60 {
            h = self.scene(ray);

            if h.d < 0.0001 {
                break;
            }

            dist += h.d;
            ray += dir * h.d * 0.9;

            if dist > self.far {
                dist = self.far;
                break;
            }
            i += 1;
        }

        let m: f32 = 1.0 - dist / self.far;
        let n: Vec3 = self.normal(ray);
        col = self.colorize(h, n, dir, self.sun) * m;

        if EDGES {
            let edge: f32 = self.edges(ray);
            col = mix(
                col,
                Vec3::zero(),
                h.edge * edge * smoothstep(0.3, 0.35, ray.length()),
            );
        }

        let neb: Vec3 = nebula(n);
        col += self.glow.min(0.1) * neb.zxy();

        // HARD SHADOW with low number of rm iterations (from obj to sun)
        if HARD_SHADOW {
            let mut ray1: Vec3 = ray;
            dir = (SUN_POS - ray1).normalize();
            ray1 += n * 0.002;

            let sun_dist: f32 = (SUN_POS - ray1).length();
            dist = 0.0;

            let mut i = 0;
            while i < 35 {
                h = self.scene(ray1 + dir * dist);
                dist += h.d;
                if h.d.abs() < 0.001 {
                    break;
                }
                i += 1;
            }

            col -= Vec3::splat(
                0.24 * smoothstep(0.5, -0.3, dist.min(sun_dist) / sun_dist.max(0.0001)),
            );
        }

        // ILLUMINATION & free shadow with low number of rm iterations (from obj to sphere)
        if GLOW {
            dir = (-ray).normalize();
            ray += n * 0.002;

            let sphere_dist: f32 = (ray.length() - 0.3).max(0.0001);
            dist = 0.0;

            let mut i = 0;
            while i < 35 {
                h = self.scene(ray + dir * dist);
                dist += h.d;
                if h.d.abs() < 0.001 {
                    break;
                }
                i += 1;
            }

            let neb1: Vec3 = nebula(rotation(0.0, self.inputs.time * 0.4).transpose() * dir).zxy();

            col += (0.7 * self.sphere_col + self.glow * neb1)
                * (0.6 * (smoothstep(3.0, 0.0, sphere_dist)) * dist.min(sphere_dist) / sphere_dist
                    + 0.6 * smoothstep(0.1, 0.0, sphere_dist));
        }

        col -= Vec3::splat(0.2 * smoothstep(0.6, 3.7, pos.length()));
        col = col.clamp(Vec3::zero(), Vec3::one());
        col = col.powf_vec(vec3(2.2, 2.4, 2.5)) * 3.9;
        col = col.powf_vec(Vec3::splat(1.0 / GAMMA));

        *frag_color = col.extend(1.0);
    }
}
