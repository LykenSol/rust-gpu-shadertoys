//! Ported to Rust from <https://www.shadertoy.com/view/lsKcDD>
//!
//! Original comment:
//! ```glsl
//! //
//! // Testing Sebastian Aaltonen's soft shadow improvement
//! //
//! // The technique is based on estimating a better closest point in ray
//! // at each step by triangulating from the previous march step.
//! //
//! // More info about the technique at slide 39 of this presentation:
//! // https://www.dropbox.com/s/s9tzmyj0wqkymmz/Claybook_Simulation_Raytracing_GDC18.pptx?dl=0
//! //
//! // Traditional technique: http://iquilezles.org/www/articles/rmshadows/rmshadows.htm
//! //
//! // Go to lines 54 to compare both.
//! ```

use glam::{vec2, vec3, Mat3, Vec2, Vec2Swizzles, Vec3, Vec3Swizzles, Vec4};
use shared::*;

// Note: This cfg is incorrect on its surface, it really should be "are we compiling with std", but
// we tie #[no_std] above to the same condition, so it's fine.
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

pub struct Inputs {
    pub resolution: Vec3,
    pub time: f32,
}

// make this 1 is your machine is too slow
const AA: usize = 2;

//------------------------------------------------------------------

fn sd_plane(p: Vec3) -> f32 {
    p.y
}

fn sd_box(p: Vec3, b: Vec3) -> f32 {
    let d: Vec3 = p.abs() - b;
    d.x.max(d.y.max(d.z)).min(0.0) + d.max(Vec3::zero()).length()
}

//------------------------------------------------------------------

fn map(pos: Vec3) -> f32 {
    let qos: Vec3 = vec3((pos.x + 0.5).gl_fract() - 0.5, pos.y, pos.z);
    return sd_plane(pos - vec3(0.0, 0.00, 0.0))
        .min(sd_box(qos - vec3(0.0, 0.25, 0.0), vec3(0.2, 0.5, 0.2)));
}

//------------------------------------------------------------------

fn calc_softshadow(ro: Vec3, rd: Vec3, mint: f32, tmax: f32, technique: i32) -> f32 {
    let mut res: f32 = 1.0;
    let mut t: f32 = mint;
    let mut ph: f32 = 1e10; // big, such that y = 0 on the first iteration
    let mut i = 0;
    while i < 32 {
        let h: f32 = map(ro + rd * t);

        // traditional technique
        if technique == 0 {
            res = res.min(10.0 * h / t);
        }
        // improved technique
        else {
            // use this if you are getting artifact on the first iteration, or unroll the
            // first iteration out of the loop
            //float y = (i==0) ? 0.0 : h*h/(2.0*ph);

            let y: f32 = h * h / (2.0 * ph);
            let d: f32 = (h * h - y * y).sqrt();
            res = res.min(10.0 * d / (t - y).max(0.0));
            ph = h;
        }
        t += h;

        if res < 0.0001 || t > tmax {
            break;
        }
        i += 1;
    }
    res.clamp(0.0, 1.0)
}

fn calc_normal(pos: Vec3) -> Vec3 {
    let e: Vec2 = vec2(1.0, -1.0) * 0.5773 * 0.0005;
    (e.xyy() * map(pos + e.xyy())
        + e.yyx() * map(pos + e.yyx())
        + e.yxy() * map(pos + e.yxy())
        + e.xxx() * map(pos + e.xxx()))
    .normalize()
}

fn cast_ray(ro: Vec3, rd: Vec3) -> f32 {
    let mut tmin: f32 = 1.0;
    let mut tmax: f32 = 20.0;

    if true {
        // bounding volume
        let tp1: f32 = (0.0 - ro.y) / rd.y;
        if tp1 > 0.0 {
            tmax = tmax.min(tp1);
        }
        let tp2: f32 = (1.0 - ro.y) / rd.y;
        if tp2 > 0.0 {
            if ro.y > 1.0 {
                tmin = tmin.max(tp2);
            } else {
                tmax = tmax.min(tp2);
            }
        }
    }
    let mut t: f32 = tmin;
    let mut i = 0;
    while i < 64 {
        let precis: f32 = 0.0005 * t;
        let res: f32 = map(ro + rd * t);
        if res < precis || t > tmax {
            break;
        }
        t += res;
        i += 1;
    }

    if t > tmax {
        t = -1.0;
    }
    t
}

fn calc_ao(pos: Vec3, nor: Vec3) -> f32 {
    let mut occ: f32 = 0.0;
    let mut sca: f32 = 1.0;
    let mut i = 0;
    while i < 5 {
        let h: f32 = 0.001 + 0.15 * i as f32 / 4.0;
        let d: f32 = map(pos + h * nor);
        occ += (h - d) * sca;
        sca *= 0.95;
        i += 1;
    }
    (1.0 - 1.5 * occ).clamp(0.0, 1.0)
}

fn render(ro: Vec3, rd: Vec3, technique: i32) -> Vec3 {
    let mut col: Vec3 = Vec3::zero();
    let t: f32 = cast_ray(ro, rd);

    if t > -0.5 {
        let pos: Vec3 = ro + t * rd;
        let nor: Vec3 = calc_normal(pos);

        // material
        let mate: Vec3 = Vec3::splat(0.3);
        // key light
        let lig: Vec3 = vec3(-0.1, 0.3, 0.6).normalize();
        let hal: Vec3 = (lig - rd).normalize();
        let dif: f32 =
            nor.dot(lig).clamp(0.0, 1.0) * calc_softshadow(pos, lig, 0.01, 3.0, technique);

        let spe: f32 = nor.dot(hal).clamp(0.0, 1.0).powf(16.0)
            * dif
            * (0.04 + 0.96 * (1.0 + hal.dot(rd)).clamp(0.0, 1.0).powf(5.0));

        col = mate * 4.0 * dif * vec3(1.00, 0.70, 0.5);
        col += 12.0 * spe * vec3(1.00, 0.70, 0.5);

        // ambient light
        let occ: f32 = calc_ao(pos, nor);
        let amb: f32 = (0.5 + 0.5 * nor.y).clamp(0.0, 1.0);
        col += mate * amb * occ * vec3(0.0, 0.08, 0.1);

        // fog
        col *= (-0.0005 * t * t * t).exp();
    }

    col
}

fn set_camera(ro: Vec3, ta: Vec3, cr: f32) -> Mat3 {
    let cw: Vec3 = (ta - ro).normalize();
    let cp: Vec3 = vec3(cr.sin(), cr.cos(), 0.0);
    let cu: Vec3 = cw.cross(cp).normalize();
    let cv: Vec3 = cu.cross(cw).normalize();
    Mat3::from_cols(cu, cv, cw)
}

impl Inputs {
    pub fn main_image(&mut self, frag_color: &mut Vec4, frag_coord: Vec2) {
        // camera
        let an: f32 = 12.0 - (0.1 * self.time).sin();
        let ro: Vec3 = vec3(3.0 * (0.1 * an).cos(), 1.0, -3.0 * (0.1 * an).sin());
        let ta: Vec3 = vec3(0.0, -0.4, 0.0);
        // camera-to-world transformation
        let ca: Mat3 = set_camera(ro, ta, 0.0);

        let technique: i32 = if (self.time / 2.0).gl_fract() > 0.5 {
            1
        } else {
            0
        };

        let mut tot: Vec3 = Vec3::zero();

        let mut m = 0;
        while m < AA {
            let mut n = 0;
            while n < AA {
                // pixel coordinates
                let o: Vec2 = vec2(m as f32, n as f32) / AA as f32 - Vec2::splat(0.5);
                let p: Vec2 = (-self.resolution.xy() + 2.0 * (frag_coord + o)) / self.resolution.y;

                // ray direction
                let rd: Vec3 = ca * p.extend(2.0).normalize();

                // render
                let mut col: Vec3 = render(ro, rd, technique);

                // gamma
                col = col.powf(0.4545);

                tot += col;

                n += 1;
            }
            m += 1;
        }
        tot /= (AA * AA) as f32;

        *frag_color = tot.extend(1.0);
    }
}
