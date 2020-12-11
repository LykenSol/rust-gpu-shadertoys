//! Ported to Rust from <https://www.shadertoy.com/view/Xsd3zf>
//!
//! Original comment:
//! ```glsl
//! /*
//! //
//! /* Panteleymonov Aleksandr Konstantinovich 2015
//! //
//! // if i write this string my code will be 0 chars, :) */
//! */
//! ```

use shared::*;
use spirv_std::glam::{
    const_vec4, vec2, vec3, vec4, Mat3, Vec2, Vec2Swizzles, Vec3, Vec3Swizzles, Vec4, Vec4Swizzles,
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

const ITERATIONS: f32 = 15.0;
const DEPTH: f32 = 0.0125;
const LAYERS: f32 = 8.0;
const LAYERSBLOB: i32 = 20;
const STEP: f32 = 1.0;
const FAR: f32 = 10000.0;

pub struct State {
    inputs: Inputs,
    radius: f32,
    zoom: f32,

    light: Vec3,
    seed: Vec2,
    iteratorc: f32,
    powr: f32,
    res: f32,

    nray: Vec3,
    nray1: Vec3,
    nray2: Vec3,
    mxc: f32,
}

impl State {
    pub fn new(inputs: Inputs) -> Self {
        State {
            inputs,
            radius: 0.25, // radius of Snowflakes. maximum for this demo 0.25.
            zoom: 4.0,    // use this to change details. optimal 0.1 - 4.0.
            light: vec3(0.0, 0.0, 1.0),
            seed: vec2(0.0, 0.0),
            iteratorc: ITERATIONS,
            powr: 0.0,
            res: 0.0,

            nray: Vec3::zero(),
            nray1: Vec3::zero(),
            nray2: Vec3::zero(),
            mxc: 1.0,
        }
    }
}

const NC0: Vec4 = const_vec4!([0.0, 157.0, 113.0, 270.0]);
const NC1: Vec4 = const_vec4!([1.0, 158.0, 114.0, 271.0]);

fn hash4(n: Vec4) -> Vec4 {
    (n.sin() * 1399763.5453123).gl_fract()
}
fn noise2(x: Vec2) -> f32 {
    let p: Vec2 = x.floor();
    let mut f: Vec2 = x.gl_fract();
    f = f * f * (Vec2::splat(3.0) - 2.0 * f);
    let n: f32 = p.x + p.y * 157.0;
    let h: Vec4 = hash4(Vec4::splat(n) + vec4(NC0.x, NC0.y, NC1.x, NC1.y));
    let s1: Vec2 = mix(h.xy(), h.zw(), f.xx());
    mix(s1.x, s1.y, f.y)
}

fn noise222(x: Vec2, y: Vec2, z: Vec2) -> f32 {
    let lx: Vec4 = vec4(x.x * y.x, x.y * y.x, x.x * y.y, x.y * y.y);
    let p: Vec4 = lx.floor();
    let mut f: Vec4 = lx.gl_fract();
    f = f * f * (Vec4::splat(3.0) - 2.0 * f);
    let n: Vec2 = p.xz() + p.yw() * 157.0;
    let h: Vec4 = mix(
        hash4(n.xxyy() + NC0.xyxy()),
        hash4(n.xxyy() + NC1.xyxy()),
        f.xxzz(),
    );
    mix(h.xz(), h.yw(), f.yw()).dot(z)
}

fn noise3(x: Vec3) -> f32 {
    let p: Vec3 = x.floor();
    let mut f: Vec3 = x.gl_fract();
    f = f * f * (Vec3::splat(3.0) - 2.0 * f);
    let n: f32 = p.x + p.yz().dot(vec2(157.0, 113.0));
    let s1: Vec4 = mix(
        hash4(Vec4::splat(n) + NC0),
        hash4(Vec4::splat(n) + NC1),
        f.xxxx(),
    );
    return mix(mix(s1.x, s1.y, f.y), mix(s1.z, s1.w, f.y), f.z);
}
fn noise3_2(x: Vec3) -> Vec2 {
    vec2(noise3(x), noise3(x + Vec3::splat(100.0)))
}

impl State {
    fn map(&self, rad: Vec2) -> f32 {
        let a: f32;
        if self.res < 0.0015 {
            //a = noise2(rad.xy*20.6)*0.9+noise2(rad.xy*100.6)*0.1;
            a = noise222(rad, vec2(20.6, 100.6), vec2(0.9, 0.1));
        } else if self.res < 0.005 {
            //let a1: f32 = mix(noise2(rad.xy()*10.6),1.0,l);
            //a = texture(iChannel0,rad*0.3).x;
            a = noise2(rad * 20.6);
            //if a1<a {a=a1;}
        } else {
            a = noise2(rad * 10.3);
        }
        a - 0.5
    }
}
impl State {
    fn dist_obj(&self, pos: Vec3, mut ray: Vec3, mut r: f32, seed: Vec2) -> Vec3 {
        let rq: f32 = r * r;
        let mut dist: Vec3 = ray * FAR;

        let norm: Vec3 = vec3(0.0, 0.0, 1.0);
        let invn: f32 = 1.0 / norm.dot(ray);
        let mut depthi: f32 = DEPTH;
        if invn < 0.0 {
            depthi = -depthi;
        }
        let mut ds: f32 = 2.0 * depthi * invn;
        let mut r1: Vec3 = ray * (norm.dot(pos) - depthi) * invn - pos;
        let op1: Vec3 = r1 + norm * depthi;
        let len1: f32 = op1.dot(op1);
        let mut r2: Vec3 = r1 + ray * ds;
        let op2: Vec3 = r2 - norm * depthi;
        let len2: f32 = op2.dot(op2);
        let n: Vec3 = ray.cross(norm).normalize();
        let mind: f32 = pos.dot(n);
        let n2: Vec3 = ray.cross(n);
        let d: f32 = n2.dot(pos) / n2.dot(norm);
        let invd: f32 = 0.2 / DEPTH;

        if (len1 < rq || len2 < rq) || (mind.abs() < r && d <= DEPTH && d >= -DEPTH) {
            let _r3: Vec3 = r2;
            let len: f32 = len1;
            if len >= rq {
                let n3: Vec3 = norm.cross(n);
                let a: f32 = 1.0 / (rq - mind * mind).sqrt() * ray.dot(n3).abs();
                let dt: Vec3 = ray / a;
                r1 = -d * norm - mind * n - dt;
                if len2 >= rq {
                    r2 = -d * norm - mind * n + dt;
                }
                ds = (r2 - r1).dot(ray);
            }
            ds = (ds.abs() + 0.1) / (ITERATIONS);
            ds = mix(DEPTH, ds, 0.2);
            if ds > 0.01 {
                ds = 0.01;
            }
            let ir: f32 = 0.35 / r;
            r *= self.zoom;
            ray = ray * ds * 5.0;
            let mut m: f32 = 0.0;
            while m < ITERATIONS {
                if m >= self.iteratorc {
                    break;
                }
                let mut l: f32 = r1.xy().length(); //r1.xy().dot(r1.xy()).sqrt();
                let mut c3: Vec2 = (r1.xy() / l).abs();
                if c3.x > 0.5 {
                    c3 = (c3 * 0.5 + vec2(-c3.y, c3.x) * 0.86602540).abs();
                }
                let g: f32 = l + c3.x * c3.x; //*1.047197551;
                l *= self.zoom;
                let mut h: f32 = l - r - 0.1;
                l = l.powf(self.powr) + 0.1;
                h = h.max(mix(self.map(c3 * l + seed), 1.0, (r1.z * invd).abs())) + g * ir - 0.245; //0.7*0.35=0.245 //*0.911890636
                if (h < self.res * 20.0) || r1.z.abs() > DEPTH + 0.01 {
                    break;
                }
                r1 += ray * h;
                ray *= 0.99;
                m += 1.0;
            }
            if r1.z.abs() < DEPTH + 0.01 {
                dist = r1 + pos;
            }
        }
        dist
    }

    fn filter_flake(
        &mut self,
        mut color: Vec4,
        pos: Vec3,
        ray: Vec3,
        ray1: Vec3,
        ray2: Vec3,
    ) -> Vec4 {
        let d: Vec3 = self.dist_obj(pos, ray, self.radius, self.seed);
        let n1: Vec3 = self.dist_obj(pos, ray1, self.radius, self.seed);
        let n2: Vec3 = self.dist_obj(pos, ray2, self.radius, self.seed);

        let lq: Vec3 = vec3(d.dot(d), n1.dot(n1), n2.dot(n2));
        if lq.x < FAR || lq.y < FAR || lq.z < FAR {
            let n: Vec3 = (n1 - d).cross(n2 - d).normalize();
            if lq.x < FAR && lq.y < FAR && lq.z < FAR {
                self.nray = n; //(self.nray+n).normalize();
                               //self.nray1 = (ray1+n).normalize();
                               //self.nray2 = (ray2+n).normalize();
            }
            let da: f32 = n.dot(self.light).abs().powf(3.0);
            let mut cf: Vec3 = mix(vec3(0.0, 0.4, 1.0), color.xyz() * 10.0, n.dot(ray).abs());
            cf = mix(cf, Vec3::splat(2.0), da);
            color = (mix(
                color.xyz(),
                cf,
                self.mxc * self.mxc * (0.5 + n.dot(ray).abs() * 0.5),
            ))
            .extend(color.w);
        }

        color
    }

    pub fn main_image(&mut self, frag_color: &mut Vec4, frag_coord: Vec2) {
        let time: f32 = self.inputs.time * 0.2; //*0.1;
        self.res = 1.0 / self.inputs.resolution.y;
        let p: Vec2 = (-self.inputs.resolution.xy() + 2.0 * frag_coord) * self.res;

        let mut rotate: Vec3;
        let mut mr: Mat3;
        let mut ray: Vec3;
        let mut ray1: Vec3;
        let mut ray2: Vec3;
        let mut pos: Vec3 = vec3(0.0, 0.0, 1.0);

        *frag_color = vec4(0.0, 0.0, 0.0, 0.0);
        self.nray = Vec3::zero();
        self.nray1 = Vec3::zero();
        self.nray2 = Vec3::zero();

        let mut refcolor: Vec4 = Vec4::zero();
        self.iteratorc = ITERATIONS - LAYERS;

        let mut addrot: Vec2 = Vec2::zero();
        if self.inputs.mouse.z > 0.0 {
            addrot = (self.inputs.mouse.xy() - self.inputs.resolution.xy() * 0.5) * self.res;
        }

        let mut mxcl: f32 = 1.0;
        let mut addpos: Vec3 = Vec3::zero();
        pos.z = 1.0;
        self.mxc = 1.0;
        self.radius = 0.25;
        let mzd: f32 = (self.zoom - 0.1) / LAYERS;
        let mut i = 0;
        while i < LAYERSBLOB {
            let p2: Vec2 = p - Vec2::splat(0.25) + Vec2::splat(0.1 * i as f32);
            ray = p2.extend(2.0) - self.nray * 2.0;
            //ray = self.nray;//*0.6;
            ray1 = (ray + vec3(0.0, self.res * 2.0, 0.0)).normalize();
            ray2 = (ray + vec3(self.res * 2.0, 0.0, 0.0)).normalize();
            ray = ray.normalize();
            let mut sb: Vec2 = ray.xy() * pos.length() / pos.normalize().dot(ray) + vec2(0.0, time);
            self.seed = (sb + vec2(0.0, pos.z)).floor() + Vec2::splat(pos.z);
            let mut seedn: Vec3 = self.seed.extend(pos.z);
            sb = sb.floor();
            if noise3(seedn) > 0.2 && i < LAYERS as i32 {
                self.powr = noise3(seedn * 10.0) * 1.9 + 0.1;
                rotate = (((Vec2::splat(0.5) - noise3_2(seedn)) * time * 5.0).sin() * 0.3 + addrot)
                    .extend(0.0);
                rotate.z = (0.5 - noise3(seedn + vec3(10.0, 3.0, 1.0))) * time * 5.0;
                seedn.z += time * 0.5;
                addpos = (sb + vec2(0.25, 0.25 - time) + noise3_2(seedn) * 0.5).extend(addpos.z);
                let sins: Vec3 = rotate.sin();
                let coss: Vec3 = rotate.cos();
                mr = Mat3::from_cols(
                    vec3(coss.x, 0.0, sins.x),
                    vec3(0.0, 1.0, 0.0),
                    vec3(-sins.x, 0.0, coss.x),
                );
                mr = Mat3::from_cols(
                    vec3(1.0, 0.0, 0.0),
                    vec3(0.0, coss.y, sins.y),
                    vec3(0.0, -sins.y, coss.y),
                ) * mr;
                mr = Mat3::from_cols(
                    vec3(coss.z, sins.z, 0.0),
                    vec3(-sins.z, coss.z, 0.0),
                    vec3(0.0, 0.0, 1.0),
                ) * mr;

                self.light = mr.transpose() * vec3(1.0, 0.0, 1.0).normalize();
                // let cc: Vec4 = self.filter_flake(
                // *frag_color,
                // mr.transpose() * (pos + addpos),
                // (mr.transpose() * ray + self.nray * 0.1).normalize(),
                // (mr.transpose() * ray1 + self.nray * 0.1).normalize(),
                // (mr.transpose() * ray2 + self.nray * 0.1).normalize(),
                // );
                let mut cc: Vec4 = self.filter_flake(
                    *frag_color,
                    mr.transpose() * (pos + addpos),
                    mr.transpose() * ray,
                    mr.transpose() * ray1,
                    mr.transpose() * ray2,
                );
                if false {
                    if i > 0
                        && self.nray.dot(self.nray) != 0.0
                        && self.nray1.dot(self.nray1) != 0.0
                        && self.nray2.dot(self.nray2) != 0.0
                    {
                        refcolor = self.filter_flake(
                            refcolor,
                            mr.transpose() * (pos + addpos),
                            self.nray,
                            self.nray1,
                            self.nray2,
                        );
                    }
                    cc += refcolor * 0.5;
                }
                *frag_color = mix(cc, *frag_color, frag_color.w.min(1.0));
            }
            seedn = sb.extend(pos.z) + vec3(0.5, 1000.0, 300.0);
            if noise3(seedn * 10.0) > 0.4 {
                let raf: f32 = 0.3 + noise3(seedn * 100.0);
                addpos =
                    (sb + vec2(0.2, 0.2 - time) + noise3_2(seedn * 100.0) * 0.6).extend(addpos.z);
                let mut l: f32 = (ray * ray.dot(pos + addpos) - pos - addpos).length();
                l = (1.0 - l * 10.0 * raf).max(0.0);
                *frag_color +=
                    vec4(1.0, 1.2, 3.0, 1.0) * l.powf(5.0) * ((0.6 + raf).powf(2.0) - 0.6) * mxcl;
            }
            self.mxc -= 1.1 / LAYERS;
            pos.z += STEP;
            self.iteratorc += 2.0;
            mxcl -= 1.1 / LAYERSBLOB as f32;
            self.zoom -= mzd;
            i += 1;
        }

        let cr: Vec3 = mix(Vec3::zero(), vec3(0.0, 0.0, 0.4), (-0.55 + p.y) * 2.0);
        *frag_color = (frag_color.xyz()
            + mix(
                (cr - frag_color.xyz()) * 0.1,
                vec3(0.2, 0.5, 1.0),
                ((-p.y + 1.0) * 0.5).clamp(0.0, 1.0),
            ))
        .extend(frag_color.z);

        *frag_color = Vec4::one().min(*frag_color);
    }
}
