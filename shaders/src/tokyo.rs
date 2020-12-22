//! Ported to Rust from <https://www.shadertoy.com/view/Xtf3zn>
//!
//! Original comment:
//! ```glsl
//! // Created by Reinder Nijhoff 2014
//! // Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.
//! // @reindernijhoff
//! //
//! // https://www.shadertoy.com/view/Xtf3zn
//! //
//! // Tokyo by night in the rain. The car model is made by Eiffie
//! // (Shiny Toy': https://www.shadertoy.com/view/ldsGWB).
//! // I have never been in Tokyo btw.
//! ```

use shared::*;
use spirv_std::glam::{
    const_mat2, const_vec3, vec2, vec3, vec4, Mat2, Vec2, Vec2Swizzles, Vec3, Vec3Swizzles, Vec4,
    Vec4Swizzles,
};

// Note: This cfg is incorrect on its surface, it really should be "are we compiling with std", but
// we tie #[no_std] above to the same condition, so it's fine.
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

#[derive(Clone, Copy)]
pub struct Inputs {
    pub resolution: Vec3,
    pub time: f32,
}

pub struct State {
    inputs: Inputs,

    d_l: f32, // minimal distance to light

    int1: Vec3,
    int2: Vec3,
    nor1: Vec3,
    lint1: Vec4,
    lint2: Vec4,
}

impl State {
    pub fn new(inputs: Inputs) -> State {
        State {
            inputs,

            d_l: 0.0,

            int1: Vec3::zero(),
            int2: Vec3::zero(),
            nor1: Vec3::zero(),
            lint1: Vec4::zero(),
            lint2: Vec4::zero(),
        }
    }
}

const BUMPMAP: bool = false;
const MARCHSTEPS: i32 = 128;
const MARCHSTEPSREFLECTION: i32 = 48;
const LIGHTINTENSITY: f32 = 5.0;

//----------------------------------------------------------------------

//const backgroundColor: Vec3 = const_vec3!(0.2,0.4,0.6) * 0.09;
const BACKGROUND_COLOR: Vec3 = const_vec3!([0.2 * 0.09, 0.4 * 0.09, 0.6 * 0.09]);
impl State {
    fn time(&self) -> f32 {
        self.inputs.time + 90.0
    }
}

//----------------------------------------------------------------------
// noises

fn hash(n: f32) -> f32 {
    (n.sin() * 687.3123).gl_fract()
}

fn noise(x: Vec2) -> f32 {
    let p: Vec2 = x.floor();
    let mut f: Vec2 = x.gl_fract();
    f = f * f * (Vec2::splat(3.0) - 2.0 * f);
    let n: f32 = p.x + p.y * 157.0;
    mix(
        mix(hash(n + 0.0), hash(n + 1.0), f.x),
        mix(hash(n + 157.0), hash(n + 158.0), f.x),
        f.y,
    )
}

const M2: Mat2 = const_mat2!([0.80, -0.60, 0.60, 0.80]);

fn fbm(mut p: Vec2) -> f32 {
    let mut f: f32 = 0.0;
    f += 0.5000 * noise(p);
    p = M2 * p * 2.02;
    f += 0.2500 * noise(p);
    p = M2 * p * 2.03;
    f += 0.1250 * noise(p);
    // p = M2 * p * 2.01;
    // f += 0.0625*noise( p );

    f / 0.9375
}

//----------------------------------------------------------------------
// distance primitives

fn ud_round_box(p: Vec3, b: Vec3, r: f32) -> f32 {
    (p.abs() - b).max(Vec3::zero()).length() - r
}

fn sd_box(p: Vec3, b: Vec3) -> f32 {
    let d: Vec3 = p.abs() - b;
    d.x.max(d.y.max(d.z)).min(0.0) + d.max(Vec3::zero()).length()
}

fn sd_sphere(p: Vec3, s: f32) -> f32 {
    p.length() - s
}

fn sd_cylinder(p: Vec3, h: Vec2) -> f32 {
    let d: Vec2 = vec2(p.xz().length(), p.y).abs() - h;
    d.x.max(d.y).min(0.0) + d.max(Vec2::zero()).length()
}

//----------------------------------------------------------------------
// distance operators

fn op_u(d2: f32, d1: f32) -> f32 {
    d1.min(d2)
}
fn op_s(d2: f32, d1: f32) -> f32 {
    (-d1).max(d2)
}
fn smin(a: f32, b: f32, k: f32) -> f32 {
    //from iq
    -((-k * a).exp() + (-k * b).exp()).ln() / k
}

//----------------------------------------------------------------------
// Map functions

// car model is made by Eiffie
// shader 'Shiny Toy': https://www.shadertoy.com/view/ldsGWB

fn map_car(p0: Vec3) -> f32 {
    let mut p: Vec3 = p0 + vec3(0.0, 1.24, 0.0);
    let mut r: f32 = p.yz().length();
    let mut d: f32 = vec3(p.x.abs() - 0.35, r - 1.92, -p.y + 1.4)
        .max(Vec3::zero())
        .length()
        - 0.05;
    d = d.max(p.z - 1.0);
    p = p0 + vec3(0.0, -0.22, 0.39);
    p = (p.xz().abs() - vec2(0.5300, 0.9600)).extend(p.y).xzy();
    p.x = p.x.abs();
    r = p.yz().length();
    d = smin(
        d,
        vec3(p.x - 0.08, r - 0.25, -p.y - 0.08)
            .max(Vec3::zero())
            .length()
            - 0.04,
        8.0,
    );
    d = d.max(-(p.x - 0.165).max(r - 0.24));
    let d2: f32 = vec2((p.x - 0.13).max(0.0), r - 0.2).length() - 0.02;
    d = d.min(d2);

    d
}

impl State {
    fn map(&mut self, p: Vec3) -> f32 {
        let mut pd: Vec3 = p;
        let mut d: f32;

        pd.x = pd.x.abs();
        pd.z *= -p.x.gl_sign();

        let ch: f32 = hash(((pd.z + 18.0 * self.time()) / 40.0).floor());
        let lh: f32 = hash((pd.z / 13.0).floor());

        let pdm: Vec3 = vec3(pd.x, pd.y, pd.z.rem_euclid(10.0) - 5.0);
        self.d_l = sd_sphere(vec3(pdm.x - 8.1, pdm.y - 4.5, pdm.z), 0.1);

        self.d_l = op_u(
            self.d_l,
            sd_box(
                vec3(pdm.x - 12.0, pdm.y - 9.5 - lh, pd.z.rem_euclid(91.0) - 45.5),
                vec3(0.2, 4.5, 0.2),
            ),
        );
        self.d_l = op_u(
            self.d_l,
            sd_box(
                vec3(
                    pdm.x - 12.0,
                    pdm.y - 11.5 + lh,
                    pd.z.rem_euclid(31.0) - 15.5,
                ),
                vec3(0.22, 5.5, 0.2),
            ),
        );
        self.d_l = op_u(
            self.d_l,
            sd_box(
                vec3(pdm.x - 12.0, pdm.y - 8.5 - lh, pd.z.rem_euclid(41.0) - 20.5),
                vec3(0.24, 3.5, 0.2),
            ),
        );

        if lh > 0.5 {
            self.d_l = op_u(
                self.d_l,
                sd_box(
                    vec3(pdm.x - 12.5, pdm.y - 2.75 - lh, pd.z.rem_euclid(13.) - 6.5),
                    vec3(0.1, 0.25, 3.2),
                ),
            );
        }

        let pm: Vec3 = vec3(
            (pd.x + (pd.z * 4.0).floor() * 0.25).rem_euclid(0.5) - 0.25,
            pd.y,
            pd.z.rem_euclid(0.25) - 0.125,
        );
        d = ud_round_box(pm, vec3(0.245, 0.1, 0.12), 0.005);

        d = op_s(d, -(p.x + 8.));
        d = op_u(d, pd.y);

        let mut pdc: Vec3 = vec3(
            pd.x,
            pd.y,
            (pd.z + 18.0 * self.time()).rem_euclid(40.0) - 20.0,
        );

        // car
        if ch > 0.75 {
            pdc.x += (ch - 0.75) * 4.0;
            self.d_l = op_u(
                self.d_l,
                sd_sphere(vec3((pdc.x - 5.0).abs() - 1.05, pdc.y - 0.55, pdc.z), 0.025),
            );
            self.d_l = op_u(
                self.d_l,
                sd_sphere(
                    vec3((pdc.x - 5.0).abs() - 1.2, pdc.y - 0.65, pdc.z + 6.05),
                    0.025,
                ),
            );

            d = op_u(d, map_car((pdc - vec3(5.0, -0.025, -2.3)) * 0.45));
        }

        d = op_u(d, 13. - pd.x);
        d = op_u(
            d,
            sd_cylinder(vec3(pdm.x - 8.5, pdm.y, pdm.z), vec2(0.075, 4.5)),
        );
        d = op_u(d, self.d_l);

        d
    }

    //----------------------------------------------------------------------

    fn calc_normal_simple(&mut self, pos: Vec3) -> Vec3 {
        let e: Vec2 = vec2(1.0, -1.0) * 0.005;

        let n: Vec3 = (e.xyy() * self.map(pos + e.xyy())
            + e.yyx() * self.map(pos + e.yyx())
            + e.yxy() * self.map(pos + e.yxy())
            + e.xxx() * self.map(pos + e.xxx()))
        .normalize();
        n
    }

    fn calc_normal(&mut self, pos: Vec3) -> Vec3 {
        let mut n: Vec3 = self.calc_normal_simple(pos);
        if pos.y > 0.12 {
            return n;
        }

        if BUMPMAP {
            let mut oc: Vec2 =
                (vec2(pos.x + (pos.z * 4.0).floor() * 0.25, pos.z) * vec2(2.0, 4.0)).floor();

            if pos.x.abs() < 8. {
                oc = pos.xz();
            }

            let p: Vec3 = pos * 250.;
            let mut xn: Vec3 = 0.05 * vec3(noise(p.xz()) - 0.5, 0., noise(p.zx()) - 0.5);
            xn += 0.1 * vec3(fbm(oc) - 0.5, 0., fbm(oc.yx()) - 0.5);

            n = (xn + n).normalize();
        }

        n
    }

    fn intersect(&mut self, mut ro: Vec3, mut rd: Vec3) -> f32 {
        let precis: f32 = 0.001;
        let mut h: f32;
        let mut t: f32 = 0.0;
        self.int1 = Vec3::splat(-500.0);
        self.int2 = Vec3::splat(-500.0);
        self.lint1 = Vec4::splat(-500.0);
        self.lint2 = Vec4::splat(-500.0);
        let mut mld: f32 = 100.0;

        let mut i = 0;
        while i < MARCHSTEPS {
            h = self.map(ro + rd * t);
            if self.d_l < mld {
                mld = self.d_l;
                self.lint1 = (ro + rd * t).extend(self.lint1.w);
                self.lint1.w = self.d_l.abs();
            }
            if h < precis {
                self.int1 = ro + rd * t;
                break;
            }
            t += h.max(precis * 2.);
            i += 1;
        }

        if self.int1.z < -400.0 || t > 300.0 {
            // check intersection with plane y = -0.1;
            let d: f32 = -(ro.y + 0.1) / rd.y;
            if d > 0.0 {
                self.int1 = ro + rd * d;
            } else {
                return -1.0;
            }
        }

        ro = ro + rd * t;
        self.nor1 = self.calc_normal(ro);
        ro += 0.01 * self.nor1;
        rd = rd.reflect(self.nor1);
        t = 0.0;
        mld = 100.;

        let mut i = 0;
        while i < MARCHSTEPSREFLECTION {
            h = self.map(ro + rd * t);
            if self.d_l < mld {
                mld = self.d_l;
                self.lint2 = (ro + rd * t).extend(self.lint2.w);
                self.lint2.w = self.d_l.abs();
            }
            if h < precis {
                self.int2 = ro + rd * t;
                return 1.0;
            }
            t += h.max(precis * 2.0);
            i += 1;
        }

        0.0
    }

    //----------------------------------------------------------------------
    // shade

    fn shade(&self, ro: Vec3, pos: Vec3, nor: Vec3) -> Vec3 {
        let mut col: Vec3 = Vec3::splat(0.5);

        if pos.x.abs() > 15.0 || pos.x.abs() < 8.0 {
            col = Vec3::splat(0.02);
        }
        if pos.y < 0.01 {
            if self.int1.x.abs() < 0.1 {
                col = Vec3::splat(0.9);
            }
            if (self.int1.x.abs() - 7.4).abs() < 0.1 {
                col = Vec3::splat(0.9);
            }
        }

        let sh: f32 = nor.dot(vec3(-0.3, 0.3, -0.5).normalize()).clamp(0.0, 1.0);
        col *= sh * BACKGROUND_COLOR;

        if pos.x.abs() > 12.9 && pos.y > 9. {
            // windows
            let ha: f32 = hash(133.1234 * (pos.y / 3.0).floor() + ((pos.z) / 3.0).floor());
            if ha > 0.95 {
                col = ((ha - 0.95) * 10.) * vec3(1., 0.7, 0.4);
            }
        }

        col = mix(
            BACKGROUND_COLOR,
            col,
            ((0.1 * pos.y).max(0.25) - 0.065 * pos.distance(ro))
                .min(0.0)
                .exp(),
        );

        col
    }

    fn get_light_color(&self, pos: Vec3) -> Vec3 {
        let mut lcol: Vec3 = vec3(1.0, 0.7, 0.5);

        let mut pd: Vec3 = pos;
        pd.x = pd.x.abs();
        pd.z *= -pos.x.gl_sign();

        let ch: f32 = hash(((pd.z + 18. * self.time()) / 40.0).floor());
        let mut pdc: Vec3 = vec3(
            pd.x,
            pd.y,
            (pd.z + 18.0 * self.time()).rem_euclid(40.0) - 20.0,
        );

        if ch > 0.75 {
            // car
            pdc.x += (ch - 0.75) * 4.;
            if sd_sphere(vec3((pdc.x - 5.0).abs() - 1.05, pdc.y - 0.55, pdc.z), 0.25) < 2. {
                lcol = vec3(1., 0.05, 0.01);
            }
        }
        if pd.y > 2.0 && pd.x.abs() > 10.0 && pd.y < 5.0 {
            let fl: f32 = (pd.z / 13.0).floor();
            lcol = 0.4 * lcol + 0.5 * vec3(hash(0.1562 + fl), hash(0.423134 + fl), 0.0);
        }
        if pd.x.abs() > 10. && pd.y > 5.0 {
            let fl: f32 = (pd.z / 2.0).floor();
            lcol = 0.5 * lcol
                + 0.5 * vec3(hash(0.1562 + fl), hash(0.923134 + fl), hash(0.423134 + fl));
        }

        lcol
    }
}

fn random_start(co: Vec2) -> f32 {
    0.8 + 0.2 * hash(co.dot(vec2(123.42, 117.853)) * 412.453)
}

impl State {
    //----------------------------------------------------------------------
    // main

    pub fn main_image(&mut self, frag_color: &mut Vec4, frag_coord: Vec2) {
        let mut q: Vec2 = frag_coord / self.inputs.resolution.xy();
        let mut p: Vec2 = -Vec2::one() + 2.0 * q;
        p.x *= self.inputs.resolution.x / self.inputs.resolution.y;

        if q.y < 0.12 || q.y >= 0.88 {
            *frag_color = vec4(0.0, 0.0, 0.0, 1.0);
            return;
        } else {
            // camera
            let z: f32 = self.time();
            let x: f32 = -10.9 + 1. * (self.time() * 0.2).sin();
            let ro: Vec3 = vec3(x, 1.3 + 0.3 * (self.time() * 0.26).cos(), z - 1.);
            let ta: Vec3 = vec3(
                -8.0,
                1.3 + 0.4 * (self.time() * 0.26).cos(),
                z + 4. + (self.time() * 0.04).cos(),
            );

            let ww: Vec3 = (ta - ro).normalize();
            let uu: Vec3 = ww.cross(vec3(0.0, 1.0, 0.0)).normalize();
            let vv: Vec3 = uu.cross(ww).normalize();
            let rd: Vec3 = (-p.x * uu + p.y * vv + 2.2 * ww).normalize();

            let mut col: Vec3 = BACKGROUND_COLOR;

            // raymarch
            let ints: f32 = self.intersect(ro + random_start(p) * rd, rd);
            if ints > -0.5 {
                // calculate reflectance
                let mut r: f32 = 0.09;
                if self.int1.y > 0.129 {
                    r = 0.025
                        * hash(
                            133.1234 * (self.int1.y / 3.0).floor() + (self.int1.z / 3.0).floor(),
                        );
                }
                if self.int1.x.abs() < 8.0 {
                    if self.int1.y < 0.01 {
                        // road
                        r = 0.007 * fbm(self.int1.xz());
                    } else {
                        // car
                        r = 0.02;
                    }
                }
                if self.int1.x.abs() < 0.1 {
                    r *= 4.0;
                }
                if (self.int1.x.abs() - 7.4).abs() < 0.1 {
                    r *= 4.0;
                }

                r *= 2.0;

                col = self.shade(ro, self.int1, self.nor1);

                if ints > 0.5 {
                    let tmp = self.calc_normal_simple(self.int2);
                    col += r * self.shade(self.int1, self.int2, tmp);
                }
                if self.lint2.w > 0. {
                    col += (r * LIGHTINTENSITY * (-self.lint2.w * 7.0).exp())
                        * self.get_light_color(self.lint2.xyz());
                }
            }

            // Rain (by Dave Hoskins)
            let st: Vec2 = 256.
                * (p * vec2(0.5, 0.01) + vec2(self.time() * 0.13 - q.y * 0.6, self.time() * 0.13));
            let mut f: f32 = noise(st) * noise(st * 0.773) * 1.55;
            f = 0.25 + (f.abs().powf(13.0) * 13.0).clamp(0.0, q.y * 0.14);

            if self.lint1.w > 0.0 {
                col += (f * LIGHTINTENSITY * (-self.lint1.w * 7.0).exp())
                    * self.get_light_color(self.lint1.xyz());
            }

            col += 0.25 * f * (Vec3::splat(0.2) + BACKGROUND_COLOR);

            // post processing
            col = col.clamp(Vec3::zero(), Vec3::one()).powf(0.4545);
            col *= 1.2 * vec3(1.0, 0.99, 0.95);
            col = (1.06 * col - Vec3::splat(0.03)).clamp(Vec3::zero(), Vec3::one());
            q.y = (q.y - 0.12) * (1. / 0.76);
            col *= Vec3::splat(0.5)
                + Vec3::splat(0.5) * (16.0 * q.x * q.y * (1.0 - q.x) * (1.0 - q.y)).powf(0.1);

            *frag_color = col.extend(1.0);
        }
    }
}
