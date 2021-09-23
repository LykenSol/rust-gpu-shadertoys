//! Ported to Rust from <https://www.shadertoy.com/view/4sXBRn>
//!
//! Original comment:
//! ```glsl
//! // Luminescence by Martijn Steinrucken aka BigWings - 2017
//! // Email:countfrolic@gmail.com Twitter:@The_ArtOfCode
//! // License Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License.
//!
//! // My entry for the monthly challenge (May 2017) on r/proceduralgeneration
//! // Use the mouse to look around. Uncomment the SINGLE define to see one specimen by itself.
//! // Code is a bit of a mess, too lazy to clean up. Hope you like it!
//!
//! // Music by Klaus Lunde
//! // https://soundcloud.com/klauslunde/zebra-tribute
//!
//! // YouTube: The Art of Code -> https://www.youtube.com/channel/UCcAlTqd9zID6aNX3TzwxJXg
//! // Twitter: @The_ArtOfCode
//! ```

use glam::{const_vec3, vec2, vec3, Mat3, Vec2, Vec3, Vec3Swizzles, Vec4, Vec4Swizzles};
use shared::*;

// Note: This cfg is incorrect on its surface, it really should be "are we compiling with std", but
// we tie #[no_std] above to the same condition, so it's fine.
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

#[derive(Clone, Copy)]
pub struct Inputs {
    pub resolution: Vec3,
    pub time: f32,
    pub mouse: Vec4,
}

pub struct State {
    inputs: Inputs,

    bg: Vec3,     // global background color
    accent: Vec3, // color of the phosphorecence

    cam: Camera,
}

impl State {
    pub fn new(inputs: Inputs) -> State {
        State {
            inputs,

            bg: Vec3::ZERO,
            accent: Vec3::ZERO,

            cam: Camera::default(),
        }
    }
}

const INVERTMOUSE: f32 = -1.0;

const MAX_STEPS: f32 = 100.0;
const VOLUME_STEPS: f32 = 8.0;
const SINGLE: bool = false;
const _MIN_DISTANCE: f32 = 0.1;
const MAX_DISTANCE: f32 = 100.0;
const HIT_DISTANCE: f32 = 0.01;

fn b(x: f32, y: f32, z: f32, w: f32) -> f32 {
    smoothstep(x - z, x + z, w) * smoothstep(y + z, y - z, w)
}
fn sat(x: f32) -> f32 {
    x.clamp(0.0, 1.0)
}
fn sin(x: f32) -> f32 {
    x.sin() * 0.5 + 0.5
}

const _LF: Vec3 = const_vec3!([1.0, 0.0, 0.0]);
const UP: Vec3 = const_vec3!([0.0, 1.0, 0.0]);
const _FW: Vec3 = const_vec3!([0.0, 0.0, 1.0]);

const _HALF_PI: f32 = 1.570796326794896619;
const PI: f32 = 3.141592653589793238;
const TWO_PI: f32 = 6.283185307179586;

const ACCENT_COLOR1: Vec3 = const_vec3!([1.0, 0.1, 0.5]);
const SECOND_COLOR1: Vec3 = const_vec3!([0.1, 0.5, 1.0]);
const ACCENT_COLOR2: Vec3 = const_vec3!([1.0, 0.5, 0.1]);
const SECOND_COLOR2: Vec3 = const_vec3!([0.1, 0.5, 0.6]);

fn _n1(x: f32) -> f32 {
    (x.sin() * 5346.1764).gl_fract()
}
fn _n2(x: f32, y: f32) -> f32 {
    _n1(x + y * 23414.324)
}

fn n3(mut p: Vec3) -> f32 {
    p = (p * 0.3183099 + Vec3::splat(0.1)).gl_fract();
    p *= 17.0;
    (p.x * p.y * p.z * (p.x + p.y + p.z)).gl_fract()
}

#[derive(Clone, Copy, Default)]
struct Ray {
    o: Vec3,
    d: Vec3,
}

#[derive(Default)]
struct Camera {
    p: Vec3,       // the position of the camera
    forward: Vec3, // the camera forward vector
    left: Vec3,    // the camera left vector
    up: Vec3,      // the camera up vector

    center: Vec3,  // the center of the screen, in world coords
    i: Vec3,       // where the current ray intersects the screen, in world coords
    ray: Ray,      // the current ray: from cam pos, through current uv projected on screen
    look_at: Vec3, // the lookat point
    zoom: f32,     // the zoom factor
}

#[derive(Clone, Copy, Default)]
struct De {
    // data type used to pass the various bits of information used to shade a de object
    d: f32, // final distance to field
    m: f32, // material
    uv: Vec3,
    pump: f32,

    id: Vec3,
    pos: Vec3, // the world-space coordinate of the fragment
}

#[derive(Default)]
struct Rc {
    // data type used to handle a repeated coordinate
    id: Vec3, // holds the floor'ed coordinate of each cell. Used to identify the cell.
    h: Vec3,  // half of the size of the cell
    p: Vec3,  // the repeated coordinate
              // c: Vec3;		// the center of the cell, world coordinates
}

fn repeat(pos: Vec3, size: Vec3) -> Rc {
    let mut o: Rc = Rc::default();
    o.h = size * 0.5;
    o.id = (pos / size).floor(); // used to give a unique id to each cell

    o.p = pos.rem_euclid_vec(size) - o.h;
    //o.c = o.id*size+o.h;

    o
}

impl State {
    fn camera_setup(&mut self, uv: Vec2, position: Vec3, look_at: Vec3, zoom: f32) {
        self.cam.p = position;
        self.cam.look_at = look_at;
        self.cam.forward = (self.cam.look_at - self.cam.p).normalize();
        self.cam.left = UP.cross(self.cam.forward);
        self.cam.up = self.cam.forward.cross(self.cam.left);
        self.cam.zoom = zoom;

        self.cam.center = self.cam.p + self.cam.forward * self.cam.zoom;
        self.cam.i = self.cam.center + self.cam.left * uv.x + self.cam.up * uv.y;

        self.cam.ray.o = self.cam.p; // ray origin = camera position
        self.cam.ray.d = (self.cam.i - self.cam.p).normalize(); // ray direction is the vector from the cam pos through the point on the imaginary screen
    }
}

// ============== Functions I borrowed ;)

//  3 out, 1 in... DAVE HOSKINS
fn n31(p: f32) -> Vec3 {
    let mut p3: Vec3 = (Vec3::splat(p) * vec3(0.1031, 0.11369, 0.13787)).gl_fract();
    p3 += Vec3::splat(p3.dot(p3.yzx() + Vec3::splat(19.19)));
    vec3(
        (p3.x + p3.y) * p3.z,
        (p3.x + p3.z) * p3.y,
        (p3.y + p3.z) * p3.x,
    )
    .gl_fract()
}

// DE functions from IQ
fn smin(a: f32, b: f32, k: f32) -> f32 {
    let h: f32 = (0.5 + 0.5 * (b - a) / k).clamp(0.0, 1.0);
    mix(b, a, h) - k * h * (1.0 - h)
}

fn smax(a: f32, b: f32, k: f32) -> f32 {
    let h: f32 = (0.5 + 0.5 * (b - a) / k).clamp(0.0, 1.0);
    mix(a, b, h) + k * h * (1.0 - h)
}

fn sd_sphere(p: Vec3, pos: Vec3, s: f32) -> f32 {
    (p - pos).length() - s
}

// From http://mercury.sexy/hg_sdf
fn p_mod_polar(p: &mut Vec2, repetitions: f32, fix: f32) -> Vec2 {
    let angle: f32 = TWO_PI / repetitions;
    let mut a: f32 = p.y.atan2(p.x) + angle / 2.;
    let r: f32 = p.length();
    let _c: f32 = (a / angle).floor();
    a = a.rem_euclid(angle) - (angle / 2.) * fix;
    *p = vec2(a.cos(), a.sin()) * r;

    *p
}

// -------------------------

fn dist(p: Vec2, p0: Vec2, p1: Vec2) -> f32 {
    //2d point-line distance

    let v: Vec2 = p1 - p0;
    let w: Vec2 = p - p0;

    let c1: f32 = w.dot(v);
    let c2: f32 = v.dot(v);

    // before P0
    if c1 <= 0. {
        return (p - p0).length();
    }

    let b: f32 = c1 / c2;
    let pb: Vec2 = p0 + b * v;
    (p - pb).length()
}

fn _closest_point(ro: Vec3, rd: Vec3, p: Vec3) -> Vec3 {
    // returns the closest point on ray r to point p
    ro + (p - ro).dot(rd).max(0.0) * rd
}

fn _ray_ray_ts(ro1: Vec3, rd1: Vec3, ro2: Vec3, rd2: Vec3) -> Vec2 {
    // returns the two t's for the closest point between two rays
    // ro+rd*t1 = ro2+rd2*t2

    let d_o: Vec3 = ro2 - ro1;
    let c_d: Vec3 = rd1.cross(rd2);
    let v: f32 = c_d.dot(c_d);

    let t1: f32 = d_o.cross(rd2).dot(c_d) / v;
    let t2: f32 = d_o.cross(rd1).dot(c_d) / v;
    vec2(t1, t2)
}

fn _dist_ray_segment(ro: Vec3, rd: Vec3, p1: Vec3, p2: Vec3) -> f32 {
    // returns the distance from ray r to line segment p1-p2
    let rd2: Vec3 = p2 - p1;
    let mut t: Vec2 = _ray_ray_ts(ro, rd, p1, rd2);

    t.x = t.x.max(0.0);
    t.y = t.y.clamp(0.0, rd2.length());

    let rp: Vec3 = ro + rd * t.x;
    let sp: Vec3 = p1 + rd2 * t.y;

    (rp - sp).length()
}

fn sph(ro: Vec3, rd: Vec3, pos: Vec3, radius: f32) -> Vec2 {
    // does a ray sphere intersection
    // returns a vec2 with distance to both intersections
    // if both a and b are MAX_DISTANCE then there is no intersection

    let oc: Vec3 = pos - ro;
    let l: f32 = rd.dot(oc);
    let det: f32 = l * l - oc.dot(oc) + radius * radius;
    if det < 0.0 {
        return Vec2::splat(MAX_DISTANCE);
    }

    let d: f32 = det.sqrt();
    let a: f32 = l - d;
    let b: f32 = l + d;

    vec2(a, b)
}

impl State {
    fn background(&self, r: Vec3) -> Vec3 {
        let x: f32 = r.x.atan2(r.z); // from -pi to pi
        let y: f32 = PI * 0.5 - r.y.acos(); // from -1/2pi to 1/2pi

        let mut col: Vec3 = self.bg * (1.0 + y);

        let t: f32 = self.inputs.time; // add god rays

        let a: f32 = r.x.sin();

        let mut beam: f32 = sat((10.0 * x + a * y * 5.0 + t).sin());
        beam *= sat((7.0 * x + a * y * 3.5 - t).sin());

        let mut beam2: f32 = sat((42.0 * x + a * y * 21.0 - t).sin());
        beam2 *= sat((34.0 * x + a * y * 17.0 + t).sin());

        beam += beam2;
        col *= 1.0 + beam * 0.05;

        col
    }
}

fn remap(a: f32, b: f32, c: f32, d: f32, t: f32) -> f32 {
    ((t - a) / (b - a)) * (d - c) + c
}

impl State {
    fn map(&self, mut p: Vec3, id: Vec3) -> De {
        let t: f32 = self.inputs.time * 2.0;

        let n: f32 = n3(id);

        let mut o: De = De::default();
        o.m = 0.0;

        let mut x: f32 = (p.y + n * TWO_PI) * 1. + t;
        let r: f32 = 1.;

        let pump: f32 = (x + x.cos()).cos() + (2.0 * x).sin() * 0.2 + (4.0 * x).sin() * 0.02;

        x = t + n * TWO_PI;
        p.y -= ((x + x.cos()).cos() + (2.0 * x).sin() * 0.2) * 0.6;
        p = (p.xz() * (1.0 + pump * 0.2)).extend(p.y).xzy();

        let d1: f32 = sd_sphere(p, vec3(0.0, 0.0, 0.0), r);
        let d2: f32 = sd_sphere(p, vec3(0.0, -0.5, 0.0), r);

        o.d = smax(d1, -d2, 0.1);
        o.m = 1.0;

        if p.y < 0.5 {
            let sway: f32 = (t + p.y + n * TWO_PI).sin() * smoothstep(0.5, -3.0, p.y) * n * 0.3;
            p.x += sway * n; // add some sway to the tentacles
            p.z += sway * (1. - n);

            let mut mp: Vec3 = p;
            mp = p_mod_polar(&mut mp.xz(), 6.0, 0.0).extend(mp.y).xzy();

            let mut d3: f32 =
                (mp.xz() - vec2(0.2, 0.1)).length() - remap(0.5, -3.5, 0.1, 0.01, mp.y);
            if d3 < o.d {
                o.m = 2.;
            }
            d3 += ((mp.y * 10.0).sin() + (mp.y * 23.0).sin()) * 0.03;

            let d32: f32 =
                (mp.xz() - vec2(0.2, 0.1)).length() - remap(0.5, -3.5, 0.1, 0.04, mp.y) * 0.5;
            d3 = d3.min(d32);
            o.d = smin(o.d, d3, 0.5);

            if p.y < 0.2 {
                let mut op: Vec3 = p;
                op = p_mod_polar(&mut op.xz(), 13.0, 1.0).extend(op.y).xzy();

                let d4: f32 =
                    (op.xz() - vec2(0.85, 0.0)).length() - remap(0.5, -3.0, 0.04, 0.0, op.y);
                if d4 < o.d {
                    o.m = 3.0;
                }
                o.d = smin(o.d, d4, 0.15);
            }
        }
        o.pump = pump;
        o.uv = p;

        o.d *= 0.8;
        o
    }
    fn calc_normal(&self, o: De) -> Vec3 {
        let eps: Vec3 = vec3(0.01, 0.0, 0.0);
        let nor: Vec3 = vec3(
            self.map(o.pos + eps.xyy(), o.id).d - self.map(o.pos - eps.xyy(), o.id).d,
            self.map(o.pos + eps.yxy(), o.id).d - self.map(o.pos - eps.yxy(), o.id).d,
            self.map(o.pos + eps.yyx(), o.id).d - self.map(o.pos - eps.yyx(), o.id).d,
        );
        nor.normalize()
    }

    fn cast_ray(&self, r: Ray) -> De {
        let mut d: f32 = 0.0;
        let _d_s: f32 = MAX_DISTANCE;

        let _pos: Vec3 = vec3(0.0, 0.0, 0.0);
        let _n: Vec3 = Vec3::ZERO;
        let mut o: De = De::default();
        let mut s: De = De::default();

        let mut d_c: f32 = MAX_DISTANCE;
        let mut p: Vec3 = Vec3::ZERO;
        let mut q: Rc = Rc::default();
        let t: f32 = self.inputs.time;
        let grid: Vec3 = vec3(6.0, 30.0, 6.0);

        let mut i = 0.0;
        while i < MAX_STEPS {
            p = r.o + r.d * d;

            if SINGLE {
                s = self.map(p, Vec3::ZERO);
            } else {
                p.y -= t; // make the move up
                p.x += t; // make cam fly forward

                q = repeat(p, grid);

                let r_c: Vec3 = ((2. * Vec3::ZERO.step(r.d) - Vec3::ONE) * q.h - q.p) / r.d; // ray to cell boundary
                d_c = r_c.x.min(r_c.y).min(r_c.z) + 0.01; // distance to cell just past boundary

                let n: f32 = n3(q.id);
                q.p += (n31(n) - Vec3::splat(0.5)) * grid * vec3(0.5, 0.7, 0.5);

                if dist(q.p.xz(), r.d.xz(), Vec2::ZERO) < 1.1 {
                    //if(DistRaySegment(q.p, r.d, vec3(0., -6., 0.), vec3(0., -3.3, 0)) <1.1)
                    s = self.map(q.p, q.id);
                } else {
                    s.d = d_c;
                }
            }

            if s.d < HIT_DISTANCE || d > MAX_DISTANCE {
                break;
            }
            d += s.d.min(d_c); // move to distance to next cell or surface, whichever is closest
            i += 1.0;
        }

        if s.d < HIT_DISTANCE {
            o.m = s.m;
            o.d = d;
            o.id = q.id;
            o.uv = s.uv;
            o.pump = s.pump;

            if SINGLE {
                o.pos = p;
            } else {
                o.pos = q.p;
            }
        }

        o
    }

    fn vol_tex(&self, uv: Vec3, mut p: Vec3, scale: f32, pump: f32) -> f32 {
        // uv = the surface pos
        // p = the volume shell pos

        p.y *= scale;

        let mut s2: f32 = 5. * p.x / TWO_PI;
        let _id: f32 = s2.floor();
        s2 = s2.gl_fract();
        let ep: Vec2 = vec2(s2 - 0.5, p.y - 0.6);
        let ed: f32 = ep.length();
        let e: f32 = b(0.35, 0.45, 0.05, ed);

        let mut s: f32 = sin(s2 * TWO_PI * 15.0);
        s = s * s;
        s = s * s;
        s *= smoothstep(1.4, -0.3, uv.y - (s2 * TWO_PI).cos() * 0.2 + 0.3)
            * smoothstep(-0.6, -0.3, uv.y);

        let t: f32 = self.inputs.time * 5.0;
        let mask: f32 = sin(p.x * TWO_PI * 2.0 + t);
        s *= mask * mask * 2.0;

        s + e * pump * 2.0
    }
}

fn jelly_tex(mut p: Vec3) -> Vec4 {
    let s: Vec3 = vec3(p.x.atan2(p.z), p.xz().length(), p.y);

    let mut b: f32 = 0.75 + (s.x * 6.0).sin() * 0.25;
    b = mix(1., b, s.y * s.y);

    p.x += (s.z * 10.0).sin() * 0.1;
    let mut b2: f32 = (s.x * 26.0).cos() - s.z - 0.7;

    b2 = smoothstep(0.1, 0.6, b2);
    Vec4::splat(b + b2)
}

impl State {
    fn render(&mut self, _uv: Vec2, cam_ray: Ray, _depth: f32) -> Vec3 {
        // outputs a color

        self.bg = self.background(self.cam.ray.d);

        let mut col: Vec3 = self.bg;
        let o: De = self.cast_ray(cam_ray);

        let _t: f32 = self.inputs.time;
        let l: Vec3 = UP;

        if o.m > 0.0 {
            let n: Vec3 = self.calc_normal(o);
            let lambert: f32 = sat(n.dot(l));
            let r: Vec3 = cam_ray.d.reflect(n);
            let fresnel: f32 = sat(1.0 + cam_ray.d.dot(n));
            let _trans: f32 = (1.0 - fresnel) * 0.5;
            let ref_: Vec3 = self.background(r);
            let mut fade: f32 = 0.0;

            if o.m == 1.0 {
                // hood color
                let mut density: f32 = 0.0;
                let mut i = 0.0;
                while i < VOLUME_STEPS {
                    let sd: f32 = sph(o.uv, cam_ray.d, Vec3::ZERO, 0.8 + i * 0.015).x;
                    if sd != MAX_DISTANCE {
                        let intersect: Vec2 = o.uv.xz() + cam_ray.d.xz() * sd;

                        let uv: Vec3 =
                            vec3(intersect.x.atan2(intersect.y), intersect.length(), o.uv.z);
                        density += self.vol_tex(o.uv, uv, 1.4 + i * 0.03, o.pump);
                    }
                    i += 1.0;
                }
                let vol_tex: Vec4 = self.accent.extend(density / VOLUME_STEPS);

                let mut dif: Vec3 = jelly_tex(o.uv).xyz();
                dif *= lambert.max(0.2);

                col = mix(col, vol_tex.xyz(), vol_tex.w);
                col = mix(col, dif, 0.25);

                col += fresnel * ref_ * sat(UP.dot(n));

                //fade
                fade = fade.max(smoothstep(0.0, 1.0, fresnel));
            } else if o.m == 2.0 {
                // inside tentacles
                let dif: Vec3 = self.accent;
                col = mix(self.bg, dif, fresnel);

                col *= mix(0.6, 1.0, smoothstep(0.0, -1.5, o.uv.y));

                let mut prop: f32 = o.pump + 0.25;
                prop *= prop * prop;
                col += (1.0 - fresnel).powf(20.0) * dif * prop;

                fade = fresnel;
            } else if o.m == 3.0 {
                // outside tentacles
                let dif: Vec3 = self.accent;
                let d: f32 = smoothstep(100.0, 13.0, o.d);
                col = mix(self.bg, dif, (1.0 - fresnel).powf(5.0) * d);
            }

            fade = fade.max(smoothstep(0.0, 100.0, o.d));
            col = mix(col, self.bg, fade);

            if o.m == 4. {
                col = vec3(1.0, 0.0, 0.0);
            }
        } else {
            col = self.bg;
        }

        col
    }

    pub fn main_image(&mut self, frag_color: &mut Vec4, frag_coord: Vec2) {
        let t: f32 = self.inputs.time * 0.04;

        let mut uv: Vec2 = frag_coord / self.inputs.resolution.xy();
        uv -= Vec2::splat(0.5);
        uv.y *= self.inputs.resolution.y / self.inputs.resolution.x;

        let mut m: Vec2 = self.inputs.mouse.xy() / self.inputs.resolution.xy();

        if m.x < 0.05 || m.x > 0.95 {
            // move cam automatically when mouse is not used
            m = vec2(t * 0.25, sin(t * PI) * 0.5 + 0.5);
        }

        self.accent = mix(ACCENT_COLOR1, ACCENT_COLOR2, sin(t * 15.456));
        self.bg = mix(SECOND_COLOR1, SECOND_COLOR2, sin(t * 7.345231));

        let turn: f32 = (0.1 - m.x) * TWO_PI;
        let s: f32 = turn.sin();
        let c: f32 = turn.cos();
        let rot_x: Mat3 = Mat3::from_cols_array(&[c, 0.0, s, 0.0, 1.0, 0.0, s, 0.0, -c]);

        let cam_dist: f32 = if SINGLE { -10.0 } else { -0.1 };

        let look_at: Vec3 = vec3(0.0, -1.0, 0.0);

        let cam_pos: Vec3 =
            rot_x.transpose() * vec3(0.0, INVERTMOUSE * cam_dist * ((m.y) * PI).cos(), cam_dist);

        self.camera_setup(uv, cam_pos + look_at, look_at, 1.0);

        let mut col: Vec3 = self.render(uv, self.cam.ray, 0.0);

        col = col.powf_vec(Vec3::splat(mix(1.5, 2.6, sin(t + PI)))); // post-processing
        let d: f32 = 1.0 - uv.dot(uv); // vignette
        col *= (d * d * d) + 0.1;

        *frag_color = col.extend(1.0);
    }
}
