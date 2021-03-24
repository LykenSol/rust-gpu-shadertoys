//! Ported to Rust from <https://www.shadertoy.com/view/3l23Rh>
//!
//! Original comment:
//! ```glsl
//! // Protean clouds by nimitz (twitter: @stormoid)
//! // https://www.shadertoy.com/view/3l23Rh
//! // License Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License
//! // Contact the author for other licensing options
//!
//! /*
//! 	Technical details:
//!
//! 	The main volume noise is generated from a deformed periodic grid, which can produce
//! 	a large range of noise-like patterns at very cheap evalutation cost. Allowing for multiple
//! 	fetches of volume gradient computation for improved lighting.
//!
//! 	To further accelerate marching, since the volume is smooth, more than half the the density
//! 	information isn't used to rendering or shading but only as an underlying volume	distance to
//! 	determine dynamic step size, by carefully selecting an equation	(polynomial for speed) to
//! 	step as a function of overall density (not necessarialy rendered) the visual results can be
//! 	the	same as a naive implementation with ~40% increase in rendering performance.
//!
//! 	Since the dynamic marching step size is even less uniform due to steps not being rendered at all
//! 	the fog is evaluated as the difference of the fog integral at each rendered step.
//!
//! */
//! ```

use shared::*;
use glam::{
    const_mat3, vec2, vec3, vec4, Mat2, Mat3, Vec2, Vec3, Vec3Swizzles, Vec4, Vec4Swizzles,
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
    prm1: f32,
    bs_mo: Vec2,
}

impl State {
    pub fn new(inputs: Inputs) -> Self {
        State {
            inputs,
            prm1: 0.0,
            bs_mo: Vec2::zero(),
        }
    }
}

fn rot(a: f32) -> Mat2 {
    let c: f32 = a.cos();
    let s: f32 = a.sin();
    Mat2::from_cols_array(&[c, s, -s, c])
}

// const m3: Mat3 = const_mat3!([
//     0.33338, 0.56034, -0.71817, -0.87887, 0.32651, -0.15323, 0.15162, 0.69596, 0.61339
// ]) * 1.93;

const M3: Mat3 = const_mat3!([
    0.33338 * 1.93,
    0.56034 * 1.93,
    -0.71817 * 1.93,
    -0.87887 * 1.93,
    0.32651 * 1.93,
    -0.15323 * 1.93,
    0.15162 * 1.93,
    0.69596 * 1.93,
    0.61339 * 1.93
]);

fn mag2(p: Vec2) -> f32 {
    p.dot(p)
}
fn linstep(mn: f32, mx: f32, x: f32) -> f32 {
    ((x - mn) / (mx - mn)).clamp(0.0, 1.0)
}

fn disp(t: f32) -> Vec2 {
    vec2((t * 0.22).sin() * 1.0, (t * 0.175).cos() * 1.0) * 2.0
}

impl State {
    fn map(&self, mut p: Vec3) -> Vec2 {
        let mut p2: Vec3 = p;
        p2 = (p2.xy() - disp(p.z)).extend(p2.z);
        p = (rot(
            (p.z + self.inputs.time).sin() * (0.1 + self.prm1 * 0.05) + self.inputs.time * 0.09
        )
        .transpose()
            * p.xy())
        .extend(p.z);
        let cl: f32 = mag2(p2.xy());
        let mut d: f32 = 0.0;
        p *= 0.61;
        let mut z: f32 = 1.0;
        let mut trk: f32 = 1.0;
        let dsp_amp: f32 = 0.1 + self.prm1 * 0.2;
        let mut i = 0;
        while i < 5 {
            p += (p.zxy() * 0.75 * trk + Vec3::splat(self.inputs.time) * trk * 0.8).sin() * dsp_amp;
            d -= (p.cos().dot(p.yzx().sin()) * z).abs();
            z *= 0.57;
            trk *= 1.4;
            let m3 = M3;
            p = m3.transpose() * p;
            i += 1;
        }
        d = (d + self.prm1 * 3.0).abs() + self.prm1 * 0.3 - 2.5 + self.bs_mo.y;
        vec2(d + cl * 0.2 + 0.25, cl)
    }

    fn render(&self, ro: Vec3, rd: Vec3, time: f32) -> Vec4 {
        let mut rez: Vec4 = Vec4::zero();
        const LDST: f32 = 8.0;
        let _lpos: Vec3 = (disp(time + LDST) * 0.5).extend(time + LDST);
        let mut t: f32 = 1.5;
        let mut fog_t: f32 = 0.0;

        let mut i = 0;
        while i < 130 {
            if rez.w > 0.99 {
                break;
            }

            let pos: Vec3 = ro + t * rd;
            let mpv: Vec2 = self.map(pos);
            let den: f32 = (mpv.x - 0.3).clamp(0.0, 1.0) * 1.12;
            let dn: f32 = (mpv.x + 2.0).clamp(0.0, 3.0);

            let mut col: Vec4 = Vec4::zero();
            if mpv.x > 0.6 {
                col = ((vec3(5.0, 0.4, 0.2)
                    + Vec3::splat(mpv.y * 0.1 + (pos.z * 0.4).sin() * 0.5 + 1.8))
                .sin()
                    * 0.5
                    + Vec3::splat(0.5))
                .extend(0.08);
                col *= den * den * den;
                col = (col.xyz() * linstep(4.0, -2.5, mpv.x) * 2.3).extend(col.w);
                let mut dif: f32 =
                    ((den - self.map(pos + Vec3::splat(0.8)).x) / 9.0).clamp(0.001, 1.0);
                dif += ((den - self.map(pos + Vec3::splat(0.35)).x) / 2.5).clamp(0.001, 1.0);
                col = (col.xyz()
                    * den
                    * (vec3(0.005, 0.045, 0.075) + 1.5 * vec3(0.033, 0.07, 0.03) * dif))
                    .extend(col.w);
            }

            let fog_c = (t * 0.2 - 2.2).exp();
            col += vec4(0.06, 0.11, 0.11, 0.1) * (fog_c - fog_t).clamp(0.0, 1.0);
            fog_t = fog_c;
            rez = rez + col * (1.0 - rez.w);
            t += (0.5 - dn * dn * 0.05).clamp(0.09, 0.3);

            i += 1;
        }

        rez.clamp(Vec4::zero(), Vec4::one())
    }
}

fn getsat(c: Vec3) -> f32 {
    let mi: f32 = c.x.min(c.y).min(c.z);
    let ma: f32 = c.x.max(c.y).max(c.z);
    (ma - mi) / (ma + 1e-7)
}

//from my "Will it blend" shader (https://www.shadertoy.com/view/lsdGzN)
fn i_lerp(a: Vec3, b: Vec3, x: f32) -> Vec3 {
    let mut ic: Vec3 = mix(a, b, x) + vec3(1e-6, 0.0, 0.0);
    let sd: f32 = (getsat(ic) - mix(getsat(a), getsat(b), x)).abs();
    let dir: Vec3 = vec3(
        2.0 * ic.x - ic.y - ic.z,
        2.0 * ic.y - ic.x - ic.z,
        2.0 * ic.z - ic.y - ic.x,
    )
    .normalize();
    let lgt: f32 = Vec3::splat(1.0).dot(ic);
    let ff: f32 = dir.dot(ic.normalize());
    ic += 1.5 * dir * sd * ff * lgt;
    ic.clamp(Vec3::zero(), Vec3::one())
}

impl State {
    pub fn main_image(&mut self, frag_color: &mut Vec4, frag_coord: Vec2) {
        let q: Vec2 = frag_coord / self.inputs.resolution.xy();
        let p: Vec2 = (frag_coord - 0.5 * self.inputs.resolution.xy()) / self.inputs.resolution.y;
        self.bs_mo =
            (self.inputs.mouse.xy() - 0.5 * self.inputs.resolution.xy()) / self.inputs.resolution.y;

        let time: f32 = self.inputs.time * 3.;
        let mut ro: Vec3 = vec3(0.0, 0.0, time);

        ro += vec3(
            self.inputs.time.sin() * 0.5,
            (self.inputs.time.sin() * 1.0) * 0.0,
            0.0,
        );

        let dsp_amp: f32 = 0.85;
        ro = (ro.xy() + disp(ro.z) * dsp_amp).extend(ro.z);
        let tgt_dst: f32 = 3.5;

        let target: Vec3 =
            (ro - (disp(time + tgt_dst) * dsp_amp).extend(time + tgt_dst)).normalize();
        ro.x -= self.bs_mo.x * 2.;
        let mut rightdir: Vec3 = target.cross(vec3(0.0, 1.0, 0.0)).normalize();
        let updir: Vec3 = rightdir.cross(target).normalize();
        rightdir = updir.cross(target).normalize();
        let mut rd: Vec3 = ((p.x * rightdir + p.y * updir) * 1.0 - target).normalize();
        rd = (rot(-disp(time + 3.5).x * 0.2 + self.bs_mo.x).transpose() * rd.xy()).extend(rd.z);
        self.prm1 = smoothstep(-0.4, 0.4, (self.inputs.time * 0.3).sin());
        let scn: Vec4 = self.render(ro, rd, time);

        let mut col: Vec3 = scn.xyz();
        col = i_lerp(col.zyx(), col, (1.0 - self.prm1).clamp(0.05, 1.0));

        col = col.powf_vec(vec3(0.55, 0.65, 0.6)) * vec3(1.0, 0.97, 0.9);

        col *= (16.0 * q.x * q.y * (1.0 - q.x) * (1.0 - q.y)).powf(0.12) * 0.7 + 0.3; //Vign

        *frag_color = col.extend(1.0);
    }
}
