//! Ported to Rust from <https://www.shadertoy.com/view/Xds3zN>
//!
//! Original comment:
//! ```glsl
//! // The MIT License
//! // Copyright © 2013 Inigo Quilez
//! // Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//!
//! // A list of useful distance function to simple primitives. All
//! // these functions (except for ellipsoid) return an exact
//! // euclidean distance, meaning they produce a better SDF than
//! // what you'd get if you were constructing them from boolean
//! // operations.
//! //
//! // More info here:
//! //
//! // https://www.iquilezles.org/www/articles/distfunctions/distfunctions.htm
//! ```

use glam::{vec2, vec3, Mat3, Vec2, Vec2Swizzles, Vec3, Vec3Swizzles, Vec4, Vec4Swizzles};
use shared::*;

// Note: This cfg is incorrect on its surface, it really should be "are we compiling with std", but
// we tie #[no_std] above to the same condition, so it's fine.
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

pub struct Inputs {
    pub resolution: Vec3,
    pub frame: i32,
    pub time: f32,
    pub mouse: Vec4,
}

const HW_PERFORMANCE: usize = 1;
const AA: i32 = if HW_PERFORMANCE == 0 {
    1
} else {
    2 // make this 2 or 3 for antialiasing
};

//------------------------------------------------------------------
fn dot2_vec2(v: Vec2) -> f32 {
    v.dot(v)
}
fn dot2_vec3(v: Vec3) -> f32 {
    v.dot(v)
}
fn ndot(a: Vec2, b: Vec2) -> f32 {
    a.x * b.x - a.y * b.y
}

fn _sd_plane(p: Vec3) -> f32 {
    p.y
}

fn sd_sphere(p: Vec3, s: f32) -> f32 {
    p.length() - s
}

fn sd_box(p: Vec3, b: Vec3) -> f32 {
    let d: Vec3 = p.abs() - b;
    d.x.max(d.y.max(d.z)).min(0.0) + d.max(Vec3::ZERO).length()
}

fn sd_bounding_box(mut p: Vec3, b: Vec3, e: f32) -> f32 {
    p = p.abs() - b;
    let q: Vec3 = (p + Vec3::splat(e)).abs() - Vec3::splat(e);

    (vec3(p.x, q.y, q.z).max(Vec3::ZERO).length() + p.x.max(q.y.max(q.z)).min(0.0))
        .min(vec3(q.x, p.y, q.z).max(Vec3::ZERO).length() + q.x.max(p.y.max(q.z)).min(0.0))
        .min(vec3(q.x, q.y, p.z).max(Vec3::ZERO).length() + q.x.max(q.y.max(p.z)).min(0.0))
}
// approximated
fn sd_ellipsoid(p: Vec3, r: Vec3) -> f32 {
    let k0: f32 = (p / r).length();
    let k1: f32 = (p / (r * r)).length();
    return k0 * (k0 - 1.0) / k1;
}

fn sd_torus(p: Vec3, t: Vec2) -> f32 {
    (vec2(p.xz().length() - t.x, p.y)).length() - t.y
}

fn sd_capped_torus(mut p: Vec3, sc: Vec2, ra: f32, rb: f32) -> f32 {
    p.x = p.x.abs();
    let k: f32 = if sc.y * p.x > sc.x * p.y {
        p.xy().dot(sc)
    } else {
        p.xy().length()
    };
    (p.dot(p) + ra * ra - 2.0 * ra * k).sqrt() - rb
}

fn sd_hex_prism(mut p: Vec3, h: Vec2) -> f32 {
    let _q: Vec3 = p.abs();

    let k: Vec3 = vec3(-0.8660254, 0.5, 0.57735);
    p = p.abs();
    p = (p.xy() - 2.0 * k.xy().dot(p.xy()).min(0.0) * k.xy()).extend(p.z);
    let d: Vec2 = vec2(
        (p.xy() - vec2(p.x.clamp(-k.z * h.x, k.z * h.x), h.x)).length() * (p.y - h.x).gl_sign(),
        p.z - h.y,
    );
    d.x.max(d.y).min(0.0) + d.max(Vec2::ZERO).length()
}

fn sd_octogon_prism(mut p: Vec3, r: f32, h: f32) -> f32 {
    let k: Vec3 = vec3(
        -0.9238795325, // sqrt(2+sqrt(2))/2
        0.3826834323,  // sqrt(2-sqrt(2))/2
        0.4142135623,
    ); // sqrt(2)-1
       // reflections
    p = p.abs();
    p = (p.xy() - 2.0 * vec2(k.x, k.y).dot(p.xy()).min(0.0) * vec2(k.x, k.y)).extend(p.z);
    p = (p.xy() - 2.0 * vec2(-k.x, k.y).dot(p.xy()).min(0.0) * vec2(-k.x, k.y)).extend(p.z);
    // polygon side
    p = (p.xy() - vec2(p.x.clamp(-k.z * r, k.z * r), r)).extend(p.z);
    let d: Vec2 = vec2(p.xy().length() * p.y.gl_sign(), p.z - h);
    d.x.max(d.y).min(0.0) + d.max(Vec2::ZERO).length()
}

fn sd_capsule(p: Vec3, a: Vec3, b: Vec3, r: f32) -> f32 {
    let pa: Vec3 = p - a;
    let ba: Vec3 = b - a;
    let h: f32 = (pa.dot(ba) / ba.dot(ba)).clamp(0.0, 1.0);
    (pa - ba * h).length() - r
}

fn sd_round_cone_vertical(p: Vec3, r1: f32, r2: f32, h: f32) -> f32 {
    let q: Vec2 = vec2(p.xz().length(), p.y);

    let b: f32 = (r1 - r2) / h;
    let a: f32 = (1.0 - b * b).sqrt();
    let k: f32 = q.dot(vec2(-b, a));

    if k < 0.0 {
        return q.length() - r1;
    }
    if k > a * h {
        return (q - vec2(0.0, h)).length() - r2;
    }

    q.dot(vec2(a, b)) - r1
}

fn sd_round_cone(p: Vec3, a: Vec3, b: Vec3, r1: f32, r2: f32) -> f32 {
    // sampling independent computations (only depend on shape)
    let ba: Vec3 = b - a;
    let l2: f32 = ba.dot(ba);
    let rr: f32 = r1 - r2;
    let a2: f32 = l2 - rr * rr;
    let il2: f32 = 1.0 / l2;

    // sampling dependant computations
    let pa: Vec3 = p - a;
    let y: f32 = pa.dot(ba);
    let z: f32 = y - l2;
    let x2: f32 = dot2_vec3(pa * l2 - ba * y);
    let y2: f32 = y * y * l2;
    let z2: f32 = z * z * l2;

    // single square root!
    let k: f32 = rr.gl_sign() * rr * rr * x2;
    if z.gl_sign() * a2 * z2 > k {
        return (x2 + z2).sqrt() * il2 - r2;
    }
    if y.gl_sign() * a2 * y2 < k {
        return (x2 + y2).sqrt() * il2 - r1;
    }
    ((x2 * a2 * il2).sqrt() + y * rr) * il2 - r1
}

fn sd_tri_prism(mut p: Vec3, mut h: Vec2) -> f32 {
    let k: f32 = 3.0_f32.sqrt();
    h.x *= 0.5 * k;
    p = (p.xy() / h.x).extend(p.z);
    p.x = p.x.abs() - 1.0;
    p.y = p.y + 1.0 / k;
    if p.x + k * p.y > 0.0 {
        p = (vec2(p.x - k * p.y, -k * p.x - p.y) / 2.0).extend(p.z);
    }
    p.x -= p.x.clamp(-2.0, 0.0);
    let d1: f32 = p.xy().length() * (-p.y).gl_sign() * h.x;
    let d2: f32 = p.z.abs() - h.y;
    vec2(d1, d2).max(Vec2::ZERO).length() + d1.max(d2).min(0.0)
}

// vertical
fn sd_cylinder_vertical(p: Vec3, h: Vec2) -> f32 {
    let d: Vec2 = vec2(p.xz().length(), p.y).abs() - h;
    d.x.max(d.y).min(0.0) + d.max(Vec2::ZERO).length()
}

// arbitrary orientation
fn sd_cylinder(p: Vec3, a: Vec3, b: Vec3, r: f32) -> f32 {
    let pa: Vec3 = p - a;
    let ba: Vec3 = b - a;
    let baba: f32 = ba.dot(ba);
    let paba: f32 = pa.dot(ba);

    let x: f32 = (pa * baba - ba * paba).length() - r * baba;
    let y: f32 = (paba - baba * 0.5).abs() - baba * 0.5;
    let x2: f32 = x * x;
    let y2: f32 = y * y * baba;
    let d: f32 = if x.max(y) < 0.0 {
        -x2.min(y2)
    } else {
        (if x > 0.0 { x2 } else { 0.0 }) + if y > 0.0 { y2 } else { 0.0 }
    };
    d.gl_sign() * d.abs().sqrt() / baba
}

// vertical
fn sd_cone(p: Vec3, c: Vec2, h: f32) -> f32 {
    let q: Vec2 = h * vec2(c.x, -c.y) / c.y;
    let w: Vec2 = vec2(p.xz().length(), p.y);

    let a: Vec2 = w - q * (w.dot(q) / q.dot(q)).clamp(0.0, 1.0);
    let b: Vec2 = w - q * vec2((w.x / q.x).clamp(0.0, 1.0), 1.0);
    let k: f32 = q.y.gl_sign();
    let d: f32 = a.dot(a).min(b.dot(b));
    let s: f32 = (k * (w.x * q.y - w.y * q.x)).max(k * (w.y - q.y));
    d.sqrt() * s.gl_sign()
}

fn sd_capped_cone_vertical(p: Vec3, h: f32, r1: f32, r2: f32) -> f32 {
    let q: Vec2 = vec2(p.xz().length(), p.y);

    let k1: Vec2 = vec2(r2, h);
    let k2: Vec2 = vec2(r2 - r1, 2.0 * h);
    let ca: Vec2 = vec2(
        q.x - q.x.min(if q.y < 0.0 { r1 } else { r2 }),
        q.y.abs() - h,
    );
    let cb: Vec2 = q - k1 + k2 * ((k1 - q).dot(k2) / dot2_vec2(k2)).clamp(0.0, 1.0);
    let s: f32 = if cb.x < 0.0 && ca.y < 0.0 { -1.0 } else { 1.0 };
    s * dot2_vec2(ca).min(dot2_vec2(cb)).sqrt()
}

fn sd_capped_cone(p: Vec3, a: Vec3, b: Vec3, ra: f32, rb: f32) -> f32 {
    let rba: f32 = rb - ra;
    let baba: f32 = (b - a).dot(b - a);
    let papa: f32 = (p - a).dot(p - a);
    let paba: f32 = (p - a).dot(b - a) / baba;

    let x: f32 = (papa - paba * paba * baba).sqrt();

    let cax: f32 = 0.0_f32.max(x - (if paba < 0.5 { ra } else { rb }));
    let cay: f32 = (paba - 0.5).abs() - 0.5;

    let k: f32 = rba * rba + baba;
    let f: f32 = ((rba * (x - ra) + paba * baba) / k).clamp(0.0, 1.0);

    let cbx: f32 = x - ra - f * rba;
    let cby: f32 = paba - f;

    let s: f32 = if cbx < 0.0 && cay < 0.0 { -1.0 } else { 1.0 };

    s * (cax * cax + cay * cay * baba)
        .min(cbx * cbx + cby * cby * baba)
        .sqrt()
}

// c is the sin/cos of the desired cone angle
fn sd_solid_angle(pos: Vec3, c: Vec2, ra: f32) -> f32 {
    let p: Vec2 = vec2(pos.xz().length(), pos.y);
    let l: f32 = p.length() - ra;
    let m: f32 = (p - c * p.dot(c).clamp(0.0, ra)).length();
    l.max(m * (c.y * p.x - c.x * p.y).gl_sign())
}

fn sd_octahedron(mut p: Vec3, s: f32) -> f32 {
    p = p.abs();
    let m: f32 = p.x + p.y + p.z - s;

    // exact distance
    if false {
        let mut o: Vec3 = (3.0 * p - Vec3::splat(m)).min(Vec3::ZERO);
        o = (6.0 * p - Vec3::splat(m) * 2.0 - o * 3.0 + Vec3::splat(o.x + o.y + o.z))
            .max(Vec3::ZERO);
        return (p - s * o / (o.x + o.y + o.z)).length();
    }

    // exact distance
    if true {
        let q: Vec3;
        if 3.0 * p.x < m {
            q = p;
        } else if 3.0 * p.y < m {
            q = p.yzx();
        } else if 3.0 * p.z < m {
            q = p.zxy();
        } else {
            return m * 0.57735027;
        }
        let k: f32 = (0.5 * (q.z - q.y + s)).clamp(0.0, s);
        return vec3(q.x, q.y - s + k, q.z - k).length();
    }

    // bound, not exact
    if false {
        return m * 0.57735027;
    }

    unreachable!();
}

fn sd_pyramid(mut p: Vec3, h: f32) -> f32 {
    let m2: f32 = h * h + 0.25;

    // symmetry
    p = p.xz().abs().extend(p.y).xzy();
    p = if p.z > p.x { p.zx() } else { p.xz() }.extend(p.y).xzy();
    p = (p.xz() - Vec2::splat(0.5)).extend(p.y).xzy();

    // project into face plane (2D)
    let q: Vec3 = vec3(p.z, h * p.y - 0.5 * p.x, h * p.x + 0.5 * p.y);

    let s: f32 = (-q.x).max(0.0);
    let t: f32 = ((q.y - 0.5 * p.z) / (m2 + 0.25)).clamp(0.0, 1.0);

    let a: f32 = m2 * (q.x + s) * (q.x + s) + q.y * q.y;
    let b: f32 = m2 * (q.x + 0.5 * t) * (q.x + 0.5 * t) + (q.y - m2 * t) * (q.y - m2 * t);

    let d2: f32 = if q.y.min(-q.x * m2 - q.y * 0.5) > 0.0 {
        0.0
    } else {
        a.min(b)
    };

    // recover 3D and scale, and add sign
    ((d2 + q.z * q.z) / m2).sqrt() * q.z.max(-p.y).gl_sign()
}

// la,lb=semi axis, h=height, ra=corner
fn sd_rhombus(mut p: Vec3, la: f32, lb: f32, h: f32, ra: f32) -> f32 {
    p = p.abs();
    let b: Vec2 = vec2(la, lb);
    let f: f32 = (ndot(b, b - 2.0 * p.xz()) / b.dot(b)).clamp(-1.0, 1.0);
    let q: Vec2 = vec2(
        (p.xz() - 0.5 * b * vec2(1.0 - f, 1.0 + f)).length()
            * (p.x * b.y + p.z * b.x - b.x * b.y).gl_sign()
            - ra,
        p.y - h,
    );
    q.x.max(q.y).min(0.0) + q.max(Vec2::ZERO).length()
}

//------------------------------------------------------------------

fn op_u(d1: Vec2, d2: Vec2) -> Vec2 {
    if d1.x < d2.x {
        d1
    } else {
        d2
    }
}

//------------------------------------------------------------------

impl Inputs {
    fn zero(&self) -> i32 {
        let frame = self.frame;
        if frame >= 0 {
            0
        } else {
            frame
        }
    }
}

//------------------------------------------------------------------

fn map(pos: Vec3) -> Vec2 {
    let mut res: Vec2 = vec2(1e10, 0.0);

    res = op_u(
        res,
        vec2(sd_sphere(pos - vec3(-2.0, 0.25, 0.0), 0.25), 26.9),
    );

    // bounding box
    if sd_box(pos - vec3(0.0, 0.3, -1.0), vec3(0.35, 0.3, 2.5)) < res.x {
        // more primitives
        res = op_u(
            res,
            vec2(
                sd_bounding_box(pos - vec3(0.0, 0.25, 0.0), vec3(0.3, 0.25, 0.2), 0.025),
                16.9,
            ),
        );
        res = op_u(
            res,
            vec2(
                sd_torus((pos - vec3(0.0, 0.30, 1.0)).xzy(), vec2(0.25, 0.05)),
                25.0,
            ),
        );
        res = op_u(
            res,
            vec2(
                sd_cone(pos - vec3(0.0, 0.45, -1.0), vec2(0.6, 0.8), 0.45),
                55.0,
            ),
        );
        res = op_u(
            res,
            vec2(
                sd_capped_cone_vertical(pos - vec3(0.0, 0.25, -2.0), 0.25, 0.25, 0.1),
                13.67,
            ),
        );
        res = op_u(
            res,
            vec2(
                sd_solid_angle(pos - vec3(0.0, 0.00, -3.0), vec2(3.0, 4.0) / 5.0, 0.4),
                49.13,
            ),
        );
    }

    // bounding box
    if sd_box(pos - vec3(1.0, 0.3, -1.0), vec3(0.35, 0.3, 2.5)) < res.x {
        // more primitives
        res = op_u(
            res,
            vec2(
                sd_capped_torus(
                    (pos - vec3(1.0, 0.30, 1.0)) * vec3(1.0, -1.0, 1.0),
                    vec2(0.866025, -0.5),
                    0.25,
                    0.05,
                ),
                8.5,
            ),
        );
        res = op_u(
            res,
            vec2(
                sd_box(pos - vec3(1.0, 0.25, 0.0), vec3(0.3, 0.25, 0.1)),
                3.0,
            ),
        );
        res = op_u(
            res,
            vec2(
                sd_capsule(
                    pos - vec3(1.0, 0.00, -1.0),
                    vec3(-0.1, 0.1, -0.1),
                    vec3(0.2, 0.4, 0.2),
                    0.1,
                ),
                31.9,
            ),
        );
        res = op_u(
            res,
            vec2(
                sd_cylinder_vertical(pos - vec3(1.0, 0.25, -2.0), vec2(0.15, 0.25)),
                8.0,
            ),
        );
        res = op_u(
            res,
            vec2(
                sd_hex_prism(pos - vec3(1.0, 0.2, -3.0), vec2(0.2, 0.05)),
                18.4,
            ),
        );
    }

    // bounding box
    if sd_box(pos - vec3(-1.0, 0.3, -1.0), vec3(0.35, 0.35, 2.5)) < res.x {
        // more primitives
        res = op_u(
            res,
            vec2(sd_pyramid(pos - vec3(-1.0, -0.6, -3.0), 1.0), 13.56),
        );
        res = op_u(
            res,
            vec2(sd_octahedron(pos - vec3(-1.0, 0.15, -2.0), 0.35), 23.56),
        );
        res = op_u(
            res,
            vec2(
                sd_tri_prism(pos - vec3(-1.0, 0.15, -1.0), vec2(0.3, 0.05)),
                43.5,
            ),
        );
        res = op_u(
            res,
            vec2(
                sd_ellipsoid(pos - vec3(-1.0, 0.25, 0.0), vec3(0.2, 0.25, 0.05)),
                43.17,
            ),
        );
        res = op_u(
            res,
            vec2(
                sd_rhombus((pos - vec3(-1.0, 0.34, 1.0)).xzy(), 0.15, 0.25, 0.04, 0.08),
                17.0,
            ),
        );
    }
    // bounding box
    if sd_box(pos - vec3(2.0, 0.3, -1.0), vec3(0.35, 0.3, 2.5)) < res.x {
        // more primitives
        res = op_u(
            res,
            vec2(
                sd_octogon_prism(pos - vec3(2.0, 0.2, -3.0), 0.2, 0.05),
                51.8,
            ),
        );
        res = op_u(
            res,
            vec2(
                sd_cylinder(
                    pos - vec3(2.0, 0.15, -2.0),
                    vec3(0.1, -0.1, 0.0),
                    vec3(-0.2, 0.35, 0.1),
                    0.08,
                ),
                31.2,
            ),
        );
        res = op_u(
            res,
            vec2(
                sd_capped_cone(
                    pos - vec3(2.0, 0.10, -1.0),
                    vec3(0.1, 0.0, 0.0),
                    vec3(-0.2, 0.40, 0.1),
                    0.15,
                    0.05,
                ),
                46.1,
            ),
        );
        res = op_u(
            res,
            vec2(
                sd_round_cone(
                    pos - vec3(2.0, 0.15, 0.0),
                    vec3(0.1, 0.0, 0.0),
                    vec3(-0.1, 0.35, 0.1),
                    0.15,
                    0.05,
                ),
                51.7,
            ),
        );
        res = op_u(
            res,
            vec2(
                sd_round_cone_vertical(pos - vec3(2.0, 0.20, 1.0), 0.2, 0.1, 0.3),
                37.0,
            ),
        );
    }

    res
}

// http://iquilezles.org/www/articles/boxfunctions/boxfunctions.htm
fn i_box(ro: Vec3, rd: Vec3, rad: Vec3) -> Vec2 {
    let m: Vec3 = 1.0 / rd;
    let n: Vec3 = m * ro;
    let k: Vec3 = m.abs() * rad;
    let t1: Vec3 = -n - k;
    let t2: Vec3 = -n + k;
    vec2(t1.x.max(t1.y).max(t1.z), t2.x.min(t2.y).min(t2.z))
}

fn raycast(ro: Vec3, rd: Vec3) -> Vec2 {
    let mut res: Vec2 = vec2(-1.0, -1.0);

    let mut tmin: f32 = 1.0;
    let mut tmax: f32 = 20.0;

    // raytrace floor plane
    let tp1: f32 = (0.0 - ro.y) / rd.y;
    if tp1 > 0.0 {
        tmax = tmax.min(tp1);
        res = vec2(tp1, 1.0);
    }
    //else return res;

    // raymarch primitives
    let tb: Vec2 = i_box(ro - vec3(0.0, 0.4, -0.5), rd, vec3(2.5, 0.41, 3.0));
    if tb.x < tb.y && tb.y > 0.0 && tb.x < tmax {
        //return vec2(tb.x,2.0);
        tmin = tb.x.max(tmin);
        tmax = tb.y.min(tmax);

        let mut t: f32 = tmin;
        let mut i = 0;
        while i < 70 && t < tmax {
            let h: Vec2 = map(ro + rd * t);
            if h.x.abs() < (0.0001 * t) {
                res = vec2(t, h.y);
                break;
            }
            t += h.x;
            i += 1;
        }
    }

    res
}

impl Inputs {
    // http://iquilezles.org/www/articles/rmshadows/rmshadows.htm
    fn calc_softshadow(&self, ro: Vec3, rd: Vec3, mint: f32, mut tmax: f32) -> f32 {
        // bounding volume
        let tp: f32 = (0.8 - ro.y) / rd.y;
        if tp > 0.0 {
            tmax = tmax.min(tp);
        }

        let mut res: f32 = 1.0;
        let mut t: f32 = mint;
        let mut i = self.zero();
        while i < 24 {
            let h: f32 = map(ro + rd * t).x;
            let s: f32 = (8.0 * h / t).clamp(0.0, 1.0);
            res = res.min(s * s * (3.0 - 2.0 * s));
            t += h.clamp(0.02, 0.2);
            if res < 0.004 || t > tmax {
                break;
            }
            i += 1;
        }
        res.clamp(0.0, 1.0)
    }

    // http://iquilezles.org/www/articles/normalsSDF/normalsSDF.htm
    fn calc_normal(&self, pos: Vec3) -> Vec3 {
        if false {
            let e: Vec2 = vec2(1.0, -1.0) * 0.5773 * 0.0005;
            (e.xyy() * map(pos + e.xyy()).x
                + e.yyx() * map(pos + e.yyx()).x
                + e.yxy() * map(pos + e.yxy()).x
                + e.xxx() * map(pos + e.xxx()).x)
                .normalize()
        } else {
            // inspired by tdhooper and klems - a way to prevent the compiler from inlining map() 4 times
            let mut n: Vec3 = Vec3::ZERO;
            let mut i = self.zero();
            while i < 4 {
                let e: Vec3 = 0.5773
                    * (2.0
                        * vec3(
                            (((i + 3) >> 1) & 1) as f32,
                            ((i >> 1) & 1) as f32,
                            (i & 1) as f32,
                        )
                        - Vec3::ONE);
                n += e * map(pos + 0.0005 * e).x;
                //if n.x+n.y+n.z>100.0 {break;}
                i += 1;
            }
            n.normalize()
        }
    }

    fn calc_ao(&self, pos: Vec3, nor: Vec3) -> f32 {
        let mut occ: f32 = 0.0;
        let mut sca: f32 = 1.0;
        let mut i = self.zero();
        while i < 5 {
            let h: f32 = 0.01 + 0.12 * i as f32 / 4.0;
            let d: f32 = map(pos + h * nor).x;
            occ += (h - d) * sca;
            sca *= 0.95;
            if occ > 0.35 {
                break;
            }
            i += 1;
        }
        (1.0 - 3.0 * occ).clamp(0.0, 1.0) * (0.5 + 0.5 * nor.y)
    }
}

// http://iquilezles.org/www/articles/checkerfiltering/checkerfiltering.htm
fn checkers_grad_box(p: Vec2, dpdx: Vec2, dpdy: Vec2) -> f32 {
    // filter kernel
    let w: Vec2 = dpdx.abs() + dpdy.abs() + Vec2::splat(0.001);
    // analytical integral (box filter)
    let i: Vec2 = 2.0
        * ((((p - 0.5 * w) * 0.5).gl_fract() - Vec2::splat(0.5)).abs()
            - (((p + 0.5 * w) * 0.5).gl_fract() - Vec2::splat(0.5)).abs())
        / w;
    // xor pattern
    0.5 - 0.5 * i.x * i.y
}

impl Inputs {
    fn render(&self, ro: Vec3, rd: Vec3, rdx: Vec3, rdy: Vec3) -> Vec3 {
        // background
        let mut col: Vec3 = vec3(0.7, 0.7, 0.9) - Vec3::splat(rd.y.max(0.0) * 0.3);

        // raycast scene
        let res: Vec2 = raycast(ro, rd);
        let t: f32 = res.x;
        let m: f32 = res.y;
        if m > -0.5 {
            let pos: Vec3 = ro + t * rd;
            let nor: Vec3 = if m < 1.5 {
                vec3(0.0, 1.0, 0.0)
            } else {
                self.calc_normal(pos)
            };
            let ref_: Vec3 = rd.reflect(nor);

            // material
            col = Vec3::splat(0.2) + 0.2 * (Vec3::splat(m) * 2.0 + vec3(0.0, 1.0, 2.0)).sin();
            let mut ks: f32 = 1.0;

            if m < 1.5 {
                // project pixel footprint into the plane
                let dpdx: Vec3 = ro.y * (rd / rd.y - rdx / rdx.y);
                let dpdy: Vec3 = ro.y * (rd / rd.y - rdy / rdy.y);

                let f: f32 = checkers_grad_box(3.0 * pos.xz(), 3.0 * dpdx.xz(), 3.0 * dpdy.xz());
                col = Vec3::splat(0.15) + f * Vec3::splat(0.05);
                ks = 0.4;
            }

            // lighting
            let occ: f32 = self.calc_ao(pos, nor);

            let mut lin: Vec3 = Vec3::ZERO;

            // sun
            {
                let lig: Vec3 = (vec3(-0.5, 0.4, -0.6)).normalize();
                let hal: Vec3 = (lig - rd).normalize();
                let mut dif: f32 = nor.dot(lig).clamp(0.0, 1.0);
                //if( dif>0.0001 )
                dif *= self.calc_softshadow(pos, lig, 0.02, 2.5);
                let mut spe: f32 = nor.dot(hal).clamp(0.0, 1.0).powf(16.0);
                spe *= dif;
                spe *= 0.04 + 0.96 * (1.0 - hal.dot(lig)).clamp(0.0, 1.0).powf(5.0);
                lin += col * 2.20 * dif * vec3(1.30, 1.00, 0.70);
                lin += 5.00 * spe * vec3(1.30, 1.00, 0.70) * ks;
            }
            // sky
            {
                let mut dif: f32 = (0.5 + 0.5 * nor.y).clamp(0.0, 1.0).sqrt();
                dif *= occ;
                let mut spe: f32 = smoothstep(-0.2, 0.2, ref_.y);
                spe *= dif;
                spe *= 0.04 + 0.96 * (1.0 + nor.dot(rd)).clamp(0.0, 1.0).powf(5.0);
                //if( spe>0.001 )
                spe *= self.calc_softshadow(pos, ref_, 0.02, 2.5);
                lin += col * 0.60 * dif * vec3(0.40, 0.60, 1.15);
                lin += 2.00 * spe * vec3(0.40, 0.60, 1.30) * ks;
            }
            // back
            {
                let mut dif: f32 = nor.dot(vec3(0.5, 0.0, 0.6).normalize()).clamp(0.0, 1.0)
                    * (1.0 - pos.y).clamp(0.0, 1.0);
                dif *= occ;
                lin += col * 0.55 * dif * vec3(0.25, 0.25, 0.25);
            }
            // sss
            {
                let mut dif: f32 = (1.0 + nor.dot(rd)).clamp(0.0, 1.0).powf(2.0);
                dif *= occ;
                lin += col * 0.25 * dif * vec3(1.00, 1.00, 1.00);
            }

            col = lin;

            col = mix(col, vec3(0.7, 0.7, 0.9), 1.0 - (-0.0001 * t * t * t).exp());
        }
        col.clamp(Vec3::ZERO, Vec3::ONE)
    }
}

fn set_camera(ro: Vec3, ta: Vec3, cr: f32) -> Mat3 {
    let cw: Vec3 = (ta - ro).normalize();
    let cp: Vec3 = vec3(cr.sin(), cr.cos(), 0.0);
    let cu: Vec3 = cw.cross(cp).normalize();
    let cv: Vec3 = cu.cross(cw);
    Mat3::from_cols(cu, cv, cw)
}

impl Inputs {
    pub fn main_image(&mut self, frag_color: &mut Vec4, frag_coord: Vec2) {
        let mo: Vec2 = self.mouse.xy() / self.resolution.xy();
        let time: f32 = 32.0 + self.time * 1.5;

        // camera
        let ta: Vec3 = vec3(0.5, -0.5, -0.6);
        let ro: Vec3 = ta
            + vec3(
                4.5 * (0.1 * time + 7.0 * mo.x).cos(),
                1.3 + 2.0 * mo.y,
                4.5 * (0.1 * time + 7.0 * mo.x).sin(),
            );
        // camera-to-world transformation
        let ca: Mat3 = set_camera(ro, ta, 0.0);

        let mut tot: Vec3 = Vec3::ZERO;
        let mut p: Vec2;

        let mut m = self.zero();
        while m < AA {
            let mut n = self.zero();
            while n < AA {
                // pixel coordinates
                let o: Vec2 = vec2(m as f32, n as f32) / AA as f32 - Vec2::splat(0.5);
                if AA > 1 {
                    p = (2.0 * (frag_coord + o) - self.resolution.xy()) / self.resolution.y;
                } else {
                    p = (2.0 * frag_coord - self.resolution.xy()) / self.resolution.y;
                }

                // ray direction
                let rd: Vec3 = ca * p.extend(2.5).normalize();

                // ray differentials
                let px: Vec2 = (2.0 * (frag_coord + vec2(1.0, 0.0)) - self.resolution.xy())
                    / self.resolution.y;
                let py: Vec2 = (2.0 * (frag_coord + vec2(0.0, 1.0)) - self.resolution.xy())
                    / self.resolution.y;
                let rdx: Vec3 = ca * px.extend(2.5).normalize();
                let rdy: Vec3 = ca * py.extend(2.5).normalize();

                // render
                let mut col: Vec3 = self.render(ro, rd, rdx, rdy);

                // gain
                // col = col*3.0/(2.5+col);

                // gamma
                col = col.powf_vec(Vec3::splat(0.4545));

                tot += col;
                n += 1;
            }
            m += 1;
        }
        tot /= (AA * AA) as f32;

        *frag_color = tot.extend(1.0);
    }
}
