//! Ported to Rust from <https://www.shadertoy.com/view/Ms2SD1>
//!
//! Original comment:
//! ```glsl
//! /*
//! * "Seascape" by Alexander Alekseev aka TDM - 2014
//! * License Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License.
//! * Contact: tdmaav@gmail.com
//! */
//! ```

use shared::*;
use glam::{
    const_mat2, const_vec3, vec2, vec3, Mat2, Mat3, Vec2, Vec3, Vec3Swizzles, Vec4,
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

const NUM_STEPS: usize = 8;
const PI: f32 = 3.141592;
const _EPSILON: f32 = 1e-3;
impl Inputs {
    fn epsilon_nrm(&self) -> f32 {
        0.1 / self.resolution.x
    }
}
const AA: bool = true;

// sea
const ITER_GEOMETRY: usize = 3;
const ITER_FRAGMENT: usize = 5;
const SEA_HEIGHT: f32 = 0.6;
const SEA_CHOPPY: f32 = 4.0;
const SEA_SPEED: f32 = 0.8;
const SEA_FREQ: f32 = 0.16;
const SEA_BASE: Vec3 = const_vec3!([0.0, 0.09, 0.18]);
// const SEA_WATER_COLOR: Vec3 = const_vec3!([0.8, 0.9, 0.6]) * 0.6;
const SEA_WATER_COLOR: Vec3 = const_vec3!([0.8 * 0.6, 0.9 * 0.6, 0.6 * 0.6]);
impl Inputs {
    fn sea_time(&self) -> f32 {
        1.0 + self.time * SEA_SPEED
    }
}
const OCTAVE_M: Mat2 = const_mat2!([1.6, 1.2, -1.2, 1.6]);

// math
fn from_euler(ang: Vec3) -> Mat3 {
    let a1: Vec2 = vec2(ang.x.sin(), ang.x.cos());
    let a2: Vec2 = vec2(ang.y.sin(), ang.y.cos());
    let a3: Vec2 = vec2(ang.z.sin(), ang.z.cos());
    Mat3::from_cols(
        vec3(
            a1.y * a3.y + a1.x * a2.x * a3.x,
            a1.y * a2.x * a3.x + a3.y * a1.x,
            -a2.y * a3.x,
        ),
        vec3(-a2.y * a1.x, a1.y * a2.y, a2.x),
        vec3(
            a3.y * a1.x * a2.x + a1.y * a3.x,
            a1.x * a3.x - a1.y * a3.y * a2.x,
            a2.y * a3.y,
        ),
    )
}
fn hash(p: Vec2) -> f32 {
    let h: f32 = p.dot(vec2(127.1, 311.7));
    (h.sin() * 43758.5453123).gl_fract()
}
fn noise(p: Vec2) -> f32 {
    let i: Vec2 = p.floor();
    let f: Vec2 = p.gl_fract();
    let u: Vec2 = f * f * (Vec2::splat(3.0) - 2.0 * f);
    -1.0 + 2.0
        * mix(
            mix(hash(i + vec2(0.0, 0.0)), hash(i + vec2(1.0, 0.0)), u.x),
            mix(hash(i + vec2(0.0, 1.0)), hash(i + vec2(1.0, 1.0)), u.x),
            u.y,
        )
}

// lighting
fn diffuse(n: Vec3, l: Vec3, p: f32) -> f32 {
    (n.dot(l) * 0.4 + 0.6).powf(p)
}
fn specular(n: Vec3, l: Vec3, e: Vec3, s: f32) -> f32 {
    let nrm: f32 = (s + 8.0) / (PI * 8.0);
    (e.reflect(n).dot(l).max(0.0)).powf(s) * nrm
}

// sky
fn get_sky_color(mut e: Vec3) -> Vec3 {
    e.y = (e.y.max(0.0) * 0.8 + 0.2) * 0.8;
    vec3((1.0 - e.y).powf(2.0), 1.0 - e.y, 0.6 + (1.0 - e.y) * 0.4) * 1.1
}

// sea
fn sea_octave(mut uv: Vec2, choppy: f32) -> f32 {
    uv += Vec2::splat(noise(uv));
    let mut wv: Vec2 = Vec2::one() - uv.sin().abs();
    let swv: Vec2 = uv.cos().abs();
    wv = mix(wv, swv, wv);
    (1.0 - (wv.x * wv.y).powf(0.65)).powf(choppy)
}

impl Inputs {
    fn map(&self, p: Vec3) -> f32 {
        let mut freq: f32 = SEA_FREQ;
        let mut amp: f32 = SEA_HEIGHT;
        let mut choppy: f32 = SEA_CHOPPY;
        let mut uv: Vec2 = p.xz();
        uv.x *= 0.75;

        let mut d: f32;
        let mut h: f32 = 0.0;

        let mut i = 0;
        while i < ITER_GEOMETRY {
            d = sea_octave((uv + Vec2::splat(self.sea_time())) * freq, choppy);
            d += sea_octave((uv - Vec2::splat(self.sea_time())) * freq, choppy);
            h += d * amp;
            let octave_m = OCTAVE_M;
            uv = octave_m.transpose() * uv;
            freq *= 1.9;
            amp *= 0.22;
            choppy = mix(choppy, 1.0, 0.2);
            i += 1;
        }
        p.y - h
    }

    fn map_detailed(&self, p: Vec3) -> f32 {
        let mut freq: f32 = SEA_FREQ;
        let mut amp: f32 = SEA_HEIGHT;
        let mut choppy: f32 = SEA_CHOPPY;
        let mut uv: Vec2 = p.xz();
        uv.x *= 0.75;
        let mut d: f32;
        let mut h: f32 = 0.0;
        let mut i = 0;
        while i < ITER_FRAGMENT {
            d = sea_octave((uv + Vec2::splat(self.sea_time())) * freq, choppy);
            d += sea_octave((uv - Vec2::splat(self.sea_time())) * freq, choppy);
            h += d * amp;
            let octave_m = OCTAVE_M;
            uv = octave_m.transpose() * uv;
            freq *= 1.9;
            amp *= 0.22;
            choppy = mix(choppy, 1.0, 0.2);
            i += 1;
        }
        p.y - h
    }
}

fn get_sea_color(p: Vec3, n: Vec3, l: Vec3, eye: Vec3, dist: Vec3) -> Vec3 {
    let mut fresnel: f32 = (1.0 - n.dot(-eye)).clamp(0.0, 1.0);
    fresnel = fresnel.powf(3.0) * 0.5;

    let reflected: Vec3 = get_sky_color(eye.reflect(n));
    let refracted: Vec3 = SEA_BASE + diffuse(n, l, 80.0) * SEA_WATER_COLOR * 0.12;

    let mut color: Vec3 = mix(refracted, reflected, fresnel);
    let atten: f32 = (1.0 - dist.dot(dist) * 0.001).max(0.0);
    color += SEA_WATER_COLOR * (p.y - SEA_HEIGHT) * 0.18 * atten;

    color += Vec3::splat(specular(n, l, eye, 60.0));
    color
}

impl Inputs {
    // tracing
    fn get_normal(&self, p: Vec3, eps: f32) -> Vec3 {
        let mut n: Vec3 = Vec3::zero();
        n.y = self.map_detailed(p);
        n.x = self.map_detailed(vec3(p.x + eps, p.y, p.z)) - n.y;
        n.z = self.map_detailed(vec3(p.x, p.y, p.z + eps)) - n.y;
        n.y = eps;
        n.normalize()
    }

    fn height_map_tracing(&self, ori: Vec3, dir: Vec3, p: &mut Vec3) -> f32 {
        let mut tm: f32 = 0.0;
        let mut tx: f32 = 1000.0;
        let mut hx: f32 = self.map(ori + dir * tx);
        if hx > 0.0 {
            return tx;
        }
        let mut hm: f32 = self.map(ori + dir * tm);
        let mut tmid: f32 = 0.0;
        let mut i = 0;
        while i < NUM_STEPS {
            tmid = mix(tm, tx, hm / (hm - hx));
            *p = ori + dir * tmid;
            let hmid: f32 = self.map(*p);
            if hmid < 0.0 {
                tx = tmid;
                hx = hmid;
            } else {
                tm = tmid;
                hm = hmid;
            }
            i += 1;
        }
        tmid
    }

    fn get_pixel(&self, coord: Vec2, time: f32) -> Vec3 {
        let mut uv: Vec2 = coord / self.resolution.xy();
        uv = uv * 2.0 - Vec2::one();
        uv.x *= self.resolution.x / self.resolution.y;
        // ray
        let ang: Vec3 = vec3((time * 3.0).sin() * 0.1, time.sin() * 0.2 + 0.3, time);
        let ori: Vec3 = vec3(0.0, 3.5, time * 5.0);
        let mut dir: Vec3 = uv.extend(-2.0).normalize();
        dir.z += uv.length() * 0.14;
        dir = from_euler(ang).transpose() * dir.normalize();
        // tracing
        let mut p: Vec3 = Vec3::zero();
        self.height_map_tracing(ori, dir, &mut p);
        let dist: Vec3 = p - ori;
        let n: Vec3 = self.get_normal(p, dist.dot(dist) * self.epsilon_nrm());
        let light: Vec3 = vec3(0.0, 1.0, 0.8).normalize();
        // color
        mix(
            get_sky_color(dir),
            get_sea_color(p, n, light, dir, dist),
            smoothstep(0.0, -0.02, dir.y).powf(0.2),
        )
    }

    // main
    pub fn main_image(&self, frag_color: &mut Vec4, frag_coord: Vec2) {
        let time: f32 = self.time * 0.3 + self.mouse.x * 0.01;
        let mut color: Vec3;
        if AA {
            color = Vec3::zero();
            let mut i = -1;
            while i <= 1 {
                let mut j = -1;
                while j <= 1 {
                    let uv: Vec2 = frag_coord + vec2(i as f32, j as f32) / 3.0;
                    color += self.get_pixel(uv, time);
                    j += 1;
                }
                i += 1;
            }
            color /= 9.0;
        } else {
            color = self.get_pixel(frag_coord, time);
        }
        // post
        *frag_color = color.powf(0.65).extend(1.0);
    }
}
