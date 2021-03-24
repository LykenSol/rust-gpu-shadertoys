//! Ported to Rust from <https://www.shadertoy.com/view/4tdSWr>

use shared::*;
use glam::{const_mat2, const_vec3, vec2, vec3, Mat2, Vec2, Vec3, Vec3Swizzles, Vec4};

// Note: This cfg is incorrect on its surface, it really should be "are we compiling with std", but
// we tie #[no_std] above to the same condition, so it's fine.
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

pub struct Inputs {
    pub resolution: Vec3,
    pub time: f32,
}

const CLOUD_SCALE: f32 = 1.1;
const SPEED: f32 = 0.03;
const CLOUD_DARK: f32 = 0.5;
const CLOUD_LIGHT: f32 = 0.3;
const CLOUD_COVER: f32 = 0.2;
const CLOUD_ALPHA: f32 = 8.0;
const SKY_TINT: f32 = 0.5;
const SKY_COLOUR1: Vec3 = const_vec3!([0.2, 0.4, 0.6]);
const SKY_COLOUR2: Vec3 = const_vec3!([0.4, 0.7, 1.0]);

const M: Mat2 = const_mat2!([1.6, 1.2, -1.2, 1.6]);

fn hash(mut p: Vec2) -> Vec2 {
    p = vec2(p.dot(vec2(127.1, 311.7)), p.dot(vec2(269.5, 183.3)));
    Vec2::splat(-1.0) + 2.0 * (p.sin() * 43758.5453123).gl_fract()
}

fn noise(p: Vec2) -> f32 {
    const K1: f32 = 0.366025404; // (sqrt(3)-1)/2;
    const K2: f32 = 0.211324865; // (3-sqrt(3))/6;
    let i: Vec2 = (p + Vec2::splat((p.x + p.y) * K1)).floor();
    let a: Vec2 = p - i + Vec2::splat((i.x + i.y) * K2);
    let o: Vec2 = if a.x > a.y {
        vec2(1.0, 0.0)
    } else {
        vec2(0.0, 1.0)
    }; //vec2 of = 0.5 + 0.5*vec2(sign(a.x-a.y), sign(a.y-a.x));
    let b: Vec2 = a - o + Vec2::splat(K2);
    let c: Vec2 = a - Vec2::splat(1.0 - 2.0 * K2);
    let h: Vec3 = (Vec3::splat(0.5) - vec3(a.dot(a), b.dot(b), c.dot(c))).max(Vec3::zero());
    let n: Vec3 = (h * h * h * h)
        * vec3(
            a.dot(hash(i + Vec2::zero())),
            b.dot(hash(i + o)),
            c.dot(hash(i + Vec2::splat(1.0))),
        );
    n.dot(Vec3::splat(70.0))
}

fn fbm(mut n: Vec2) -> f32 {
    let mut total: f32 = 0.0;
    let mut amplitude: f32 = 0.1;
    let mut i = 0;
    while i < 7 {
        total += noise(n) * amplitude;
        let m = M;
        n = m.transpose() * n;
        amplitude *= 0.4;
        i += 1;
    }
    total
}

// -----------------------------------------------

impl Inputs {
    pub fn main_image(&self, frag_color: &mut Vec4, frag_coord: Vec2) {
        let p: Vec2 = frag_coord / self.resolution.xy();
        let mut uv: Vec2 = p * vec2(self.resolution.x / self.resolution.y, 1.0);
        let mut time: f32 = self.time * SPEED;
        let q: f32 = fbm(uv * CLOUD_SCALE * 0.5);

        //ridged noise shape
        let mut r: f32 = 0.0;
        uv *= CLOUD_SCALE;
        uv -= Vec2::splat(q - time);
        let mut weight: f32 = 0.8;
        let mut i = 0;
        while i < 8 {
            r += (weight * noise(uv)).abs();
            let m = M;
            uv = m.transpose() * uv + Vec2::splat(time);
            weight *= 0.7;
            i += 1;
        }

        //noise shape
        let mut f: f32 = 0.0;
        uv = p * vec2(self.resolution.x / self.resolution.y, 1.0);
        uv *= CLOUD_SCALE;
        uv -= Vec2::splat(q - time);
        weight = 0.7;
        let mut i = 0;
        while i < 8 {
            f += weight * noise(uv);
            let m = M;
            uv = m.transpose() * uv + Vec2::splat(time);
            weight *= 0.6;
            i += 1;
        }

        f *= r + f;

        //noise colour
        let mut c: f32 = 0.0;
        time = self.time * SPEED * 2.0;
        uv = p * vec2(self.resolution.x / self.resolution.y, 1.0);
        uv *= CLOUD_SCALE * 2.0;
        uv -= Vec2::splat(q - time);
        weight = 0.4;
        let mut i = 0;
        while i < 7 {
            c += weight * noise(uv);
            let m = M;
            uv = m.transpose() * uv + Vec2::splat(time);
            weight *= 0.6;
            i += 1;
        }

        //noise ridge colour
        let mut c1: f32 = 0.0;
        time = self.time * SPEED * 3.0;
        uv = p * vec2(self.resolution.x / self.resolution.y, 1.0);
        uv *= CLOUD_SCALE * 3.0;
        uv -= Vec2::splat(q - time);
        weight = 0.4;
        let mut i = 0;
        while i < 7 {
            c1 += (weight * noise(uv)).abs();
            let m = M;
            uv = m.transpose() * uv + Vec2::splat(time);
            weight *= 0.6;
            i += 1;
        }

        c += c1;

        let skycolour: Vec3 = mix(SKY_COLOUR2, SKY_COLOUR1, p.y);
        let cloudcolour: Vec3 =
            vec3(1.1, 1.1, 0.9) * (CLOUD_DARK + CLOUD_LIGHT * c).clamp(0.0, 1.0);

        f = CLOUD_COVER + CLOUD_ALPHA * f * r;

        let result: Vec3 = mix(
            skycolour,
            (SKY_TINT * skycolour + cloudcolour).clamp(Vec3::zero(), Vec3::splat(1.0)),
            (f + c).clamp(0.0, 1.0),
        );

        *frag_color = result.extend(1.0);
    }
}
