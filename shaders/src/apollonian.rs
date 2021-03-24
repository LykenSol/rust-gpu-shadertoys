//! Ported to Rust from <https://www.shadertoy.com/view/4ds3zn>
//!
//! Original comment:
//! ```glsl
//! // Created by inigo quilez - iq/2013
//! // License Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License.
//! //
//! // I can't recall where I learnt about this fractal.
//! //
//! // Coloring and fake occlusions are done by orbit trapping, as usual.
//! ```

use glam::{vec2, vec3, Vec2, Vec2Swizzles, Vec3, Vec3Swizzles, Vec4};
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
    orb: Vec4,
}

impl State {
    pub fn new(inputs: Inputs) -> Self {
        State {
            inputs,
            orb: Vec4::ZERO,
        }
    }
}

const HW_PERFORMANCE: usize = 1;

// Antialiasing level
const AA: usize = if HW_PERFORMANCE == 0 {
    1
} else {
    2 // Make it 3 if you have a fast machine
};

impl State {
    fn map(&mut self, mut p: Vec3, s: f32) -> f32 {
        let mut scale: f32 = 1.0;
        self.orb = Vec4::splat(1000.0);
        let mut i = 0;
        while i < 8 {
            p = Vec3::splat(-1.0) + 2.0 * (0.5 * p + Vec3::splat(0.5)).gl_fract();

            let r2: f32 = p.dot(p);

            self.orb = self.orb.min(p.abs().extend(r2));

            let k: f32 = s / r2;
            p *= k;
            scale *= k;
            i += 1;
        }
        0.25 * p.y.abs() / scale
    }

    fn trace(&mut self, ro: Vec3, rd: Vec3, s: f32) -> f32 {
        let maxd = 30.0;
        let mut t: f32 = 0.01;

        let mut i = 0;
        while i < 512 {
            let precis = 0.001 * t;

            let h: f32 = self.map(ro + rd * t, s);
            if h < precis || t > maxd {
                break;
            }
            t += h;
            i += 1;
        }
        if t > maxd {
            t = -1.0;
        }
        t
    }

    fn calc_normal(&mut self, pos: Vec3, t: f32, s: f32) -> Vec3 {
        let precis: f32 = 0.001 * t;

        let e: Vec2 = vec2(1.0, -1.0) * precis;
        (e.xyy() * self.map(pos + e.xyy(), s)
            + e.yyx() * self.map(pos + e.yyx(), s)
            + e.yxy() * self.map(pos + e.yxy(), s)
            + e.xxx() * self.map(pos + e.xxx(), s))
        .normalize()
    }

    fn render(&mut self, ro: Vec3, rd: Vec3, anim: f32) -> Vec3 {
        // trace
        let mut col: Vec3 = Vec3::ZERO;
        let t: f32 = self.trace(ro, rd, anim);
        if t > 0.0 {
            let tra: Vec4 = self.orb;
            let pos: Vec3 = ro + t * rd;
            let nor: Vec3 = self.calc_normal(pos, t, anim);

            // lighting
            let light1: Vec3 = vec3(0.577, 0.577, -0.577);
            let light2: Vec3 = vec3(-0.707, 0.000, 0.707);
            let key: f32 = light1.dot(nor).clamp(0.0, 1.0);
            let bac: f32 = (0.2 + 0.8 * light2.dot(nor)).clamp(0.0, 1.0);
            let amb: f32 = 0.7 + 0.3 * nor.y;
            let ao: f32 = (tra.w * 2.0).clamp(0.0, 1.0).powf(1.2);

            let mut brdf: Vec3 = 1.0 * vec3(0.40, 0.40, 0.40) * amb * ao;
            brdf += 1.0 * vec3(1.00, 1.00, 1.00) * key * ao;
            brdf += 1.0 * vec3(0.40, 0.40, 0.40) * bac * ao;

            // material
            let mut rgb: Vec3 = Vec3::ONE;
            rgb = mix(rgb, vec3(1.0, 0.80, 0.2), (6.0 * tra.y).clamp(0.0, 1.0));
            rgb = mix(
                rgb,
                vec3(1.0, 0.55, 0.0),
                (1.0 - 2.0 * tra.z).clamp(0.0, 1.0).powf(8.0),
            );

            // color
            col = rgb * brdf * (-0.2 * t).exp();
        }

        col.sqrt()
    }

    pub fn main_image(&mut self, frag_color: &mut Vec4, frag_coord: Vec2) {
        let time: f32 = self.inputs.time * 0.25 + 0.01 * self.inputs.mouse.x;
        let anim: f32 = 1.1 + 0.5 * smoothstep(-0.3, 0.3, (0.1 * self.inputs.time).cos());
        let mut tot: Vec3 = Vec3::ZERO;

        let mut jj = 0;
        while jj < AA {
            let mut ii = 0;
            while ii < AA {
                let q: Vec2 = frag_coord + vec2(ii as f32, jj as f32) / AA as f32;
                let p: Vec2 = (2.0 * q - self.inputs.resolution.xy()) / self.inputs.resolution.y;

                // camera
                let ro: Vec3 = vec3(
                    2.8 * (0.1 + 0.33 * time).cos(),
                    0.4 + 0.30 * (0.37 * time).cos(),
                    2.8 * (0.5 + 0.35 * time).cos(),
                );
                let ta: Vec3 = vec3(
                    1.9 * (1.2 + 0.41 * time).cos(),
                    0.4 + 0.10 * (0.27 * time).cos(),
                    1.9 * (2.0 + 0.38 * time).cos(),
                );
                let roll: f32 = 0.2 * (0.1 * time).cos();
                let cw: Vec3 = (ta - ro).normalize();
                let cp: Vec3 = vec3(roll.sin(), roll.cos(), 0.0);
                let cu: Vec3 = cw.cross(cp).normalize();
                let cv: Vec3 = cu.cross(cw).normalize();
                let rd: Vec3 = (p.x * cu + p.y * cv + 2.0 * cw).normalize();

                tot += self.render(ro, rd, anim);
                ii += 1;
            }
            jj += 1;
        }

        tot = tot / (AA * AA) as f32;

        *frag_color = tot.extend(1.0);
    }

    pub fn main_vr(
        &mut self,
        frag_color: &mut Vec4,
        _frag_coord: Vec2,
        frag_ray_ori: Vec3,
        freag_ray_dir: Vec3,
    ) {
        let _time: f32 = self.inputs.time * 0.25 + 0.01 * self.inputs.mouse.x;
        let anim: f32 = 1.1 + 0.5 * smoothstep(-0.3, 0.3, (0.1 * self.inputs.time).cos());

        let col: Vec3 = self.render(frag_ray_ori + vec3(0.82, 1.2, -0.3), freag_ray_dir, anim);
        *frag_color = col.extend(1.0);
    }
}
