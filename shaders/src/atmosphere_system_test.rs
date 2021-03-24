//! Ported to Rust from <https://www.shadertoy.com/view/XtBXDz>
//!
//! Original comment:
//! ```glsl
//! // ----------------------------------------------------------------------------
//! // Rayleigh and Mie scattering atmosphere system
//! //
//! // implementation of the techniques described here:
//! // http://www.scratchapixel.com/old/lessons/3d-advanced-lessons/simulating-the-colors-of-the-sky/atmospheric-scattering/
//! // ----------------------------------------------------------------------------
//! ```

use glam::{const_vec3, vec2, vec3, Mat3, Vec2, Vec3, Vec3Swizzles, Vec4};
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
    sun_dir: Vec3,
}

impl State {
    pub fn new(inputs: Inputs) -> Self {
        State {
            inputs,
            sun_dir: vec3(0.0, 1.0, 0.0),
        }
    }
}

const PI: f32 = 3.14159265359;

#[derive(Copy, Clone)]
struct Ray {
    origin: Vec3,
    direction: Vec3,
}
const _BIAS: f32 = 1e-4; // small offset to avoid self-intersections

struct Sphere {
    origin: Vec3,
    radius: f32,
    _material: i32,
}

struct _Plane {
    direction: Vec3,
    distance: f32,
    material: i32,
}

fn rotate_around_x(angle_degrees: f32) -> Mat3 {
    let angle: f32 = angle_degrees.deg_to_radians();
    let _sin: f32 = angle.sin();
    let _cos: f32 = angle.cos();
    Mat3::from_cols_array(&[1.0, 0.0, 0.0, 0.0, _cos, -_sin, 0.0, _sin, _cos])
}

fn get_primary_ray(cam_local_point: Vec3, cam_origin: &mut Vec3, cam_look_at: &mut Vec3) -> Ray {
    let fwd: Vec3 = (*cam_look_at - *cam_origin).normalize();
    let mut up: Vec3 = vec3(0.0, 1.0, 0.0);
    let right: Vec3 = up.cross(fwd);
    up = fwd.cross(right);

    Ray {
        origin: *cam_origin,
        direction: (fwd + up * cam_local_point.y + right * cam_local_point.x).normalize(),
    }
}

fn isect_sphere(ray: Ray, sphere: Sphere, t0: &mut f32, t1: &mut f32) -> bool {
    let rc: Vec3 = sphere.origin - ray.origin;
    let radius2: f32 = sphere.radius * sphere.radius;
    let tca: f32 = rc.dot(ray.direction);
    let d2: f32 = rc.dot(rc) - tca * tca;
    if d2 > radius2 {
        return false;
    }
    let thc: f32 = (radius2 - d2).sqrt();
    *t0 = tca - thc;
    *t1 = tca + thc;
    true
}

// scattering coefficients at sea level (m)
const BETA_R: Vec3 = const_vec3!([5.5e-6, 13.0e-6, 22.4e-6]); // Rayleigh
const BETA_M: Vec3 = const_vec3!([21e-6, 21e-6, 21e-6]); // Mie

// scale height (m)
// thickness of the atmosphere if its density were uniform
const H_R: f32 = 7994.0; // Rayleigh
const H_M: f32 = 1200.0; // Mie

fn rayleigh_phase_func(mu: f32) -> f32 {
    3.0 * (1.0 + mu*mu)
	/ //------------------------
	(16.0 * PI)
}

// Henyey-Greenstein phase function factor [-1, 1]
// represents the average cosine of the scattered directions
// 0 is isotropic scattering
// > 1 is forward scattering, < 1 is backwards
const G: f32 = 0.76;
fn henyey_greenstein_phase_func(mu: f32) -> f32 {
    (1. - G*G)
	/ //---------------------------------------------
	((4. * PI) * (1. + G*G - 2.*G*mu).powf(1.5))
}

// Schlick Phase Function factor
// Pharr and  Humphreys [2004] equivalence to g above
const K: f32 = 1.55 * G - 0.55 * (G * G * G);
fn schlick_phase_func(mu: f32) -> f32 {
    (1. - K*K)
	/ //-------------------------------------------
	(4. * PI * (1. + K*mu) * (1. + K*mu))
}

const EARTH_RADIUS: f32 = 6360e3; // (m)
const ATMOSPHERE_RADIUS: f32 = 6420e3; // (m)

const SUN_POWER: f32 = 20.0;

const ATMOSPHERE: Sphere = Sphere {
    origin: Vec3::zero(),
    radius: ATMOSPHERE_RADIUS,
    _material: 0,
};

const NUM_SAMPLES: i32 = 16;
const NUM_SAMPLES_LIGHT: i32 = 8;

fn get_sun_light(ray: Ray, optical_depth_r: &mut f32, optical_depth_m: &mut f32) -> bool {
    let mut t0: f32 = 0.0;
    let mut t1: f32 = 0.0;
    isect_sphere(ray, ATMOSPHERE, &mut t0, &mut t1);

    let mut march_pos: f32 = 0.0;
    let march_step: f32 = t1 / NUM_SAMPLES_LIGHT as f32;

    let mut i = 0;
    while i < NUM_SAMPLES_LIGHT {
        let s: Vec3 = ray.origin + ray.direction * (march_pos + 0.5 * march_step);
        let height: f32 = s.length() - EARTH_RADIUS;
        if height < 0.0 {
            return false;
        }

        *optical_depth_r += (-height / H_R).exp() * march_step;
        *optical_depth_m += (-height / H_M).exp() * march_step;

        march_pos += march_step;
        i += 1;
    }
    true
}

impl State {
    fn get_incident_light(&self, ray: Ray) -> Vec3 {
        // "pierce" the atmosphere with the viewing ray
        let mut t0: f32 = 0.0;
        let mut t1: f32 = 0.0;
        if !isect_sphere(ray, ATMOSPHERE, &mut t0, &mut t1) {
            return Vec3::zero();
        }

        let march_step: f32 = t1 / NUM_SAMPLES as f32;

        // cosine of angle between view and light directions
        let mu: f32 = ray.direction.dot(self.sun_dir);

        // Rayleigh and Mie phase functions
        // A black box indicating how light is interacting with the material
        // Similar to BRDF except
        // * it usually considers a single angle
        //   (the phase angle between 2 directions)
        // * integrates to 1 over the entire sphere of directions
        let phase_r: f32 = rayleigh_phase_func(mu);
        let phase_m: f32 = if true {
            henyey_greenstein_phase_func(mu)
        } else {
            schlick_phase_func(mu)
        };

        // optical depth (or "average density")
        // represents the accumulated extinction coefficients
        // along the path, multiplied by the length of that path
        let mut optical_depth_r: f32 = 0.0;
        let mut optical_depth_m: f32 = 0.0;

        let mut sum_r: Vec3 = Vec3::zero();
        let mut sum_m: Vec3 = Vec3::zero();
        let mut march_pos: f32 = 0.0;

        let mut i = 0;
        while i < NUM_SAMPLES {
            let s: Vec3 = ray.origin + ray.direction * (march_pos + 0.5 * march_step);
            let height: f32 = s.length() - EARTH_RADIUS;

            // integrate the height scale
            let hr: f32 = (-height / H_R).exp() * march_step;
            let hm: f32 = (-height / H_M).exp() * march_step;
            optical_depth_r += hr;
            optical_depth_m += hm;

            // gather the sunlight
            let light_ray: Ray = Ray {
                origin: s,
                direction: self.sun_dir,
            };
            let mut optical_depth_light_r: f32 = 0.0;
            let mut optical_depth_light_m: f32 = 0.0;
            let overground: bool = get_sun_light(
                light_ray,
                &mut optical_depth_light_r,
                &mut optical_depth_light_m,
            );

            if overground {
                let tau: Vec3 = BETA_R * (optical_depth_r + optical_depth_light_r)
                    + BETA_M * 1.1 * (optical_depth_m + optical_depth_light_m);
                let attenuation: Vec3 = exp(-tau);

                sum_r += hr * attenuation;
                sum_m += hm * attenuation;
            }

            march_pos += march_step;
            i += 1;
        }

        SUN_POWER * (sum_r * phase_r * BETA_R + sum_m * phase_m * BETA_M)
    }

    pub fn main_image(&mut self, frag_color: &mut Vec4, frag_coord: Vec2) {
        let aspect_ratio: Vec2 = vec2(self.inputs.resolution.x / self.inputs.resolution.y, 1.0);
        let fov: f32 = 45.0.deg_to_radians().tan();
        let point_ndc: Vec2 = frag_coord / self.inputs.resolution.xy();
        let point_cam: Vec3 = ((2.0 * point_ndc - Vec2::one()) * aspect_ratio * fov).extend(-1.0);

        let col: Vec3;

        // sun
        let rot: Mat3 = rotate_around_x(-(self.inputs.time / 2.0).sin().abs() * 90.0);
        self.sun_dir = rot.transpose() * self.sun_dir;

        if self.inputs.mouse.z < 0.1 {
            // sky dome angles
            let p: Vec3 = point_cam;
            let z2: f32 = p.x * p.x + p.y * p.y;
            let phi: f32 = p.y.atan2(p.x);
            let theta: f32 = (1.0 - z2).acos();
            let dir: Vec3 = vec3(
                theta.sin() * phi.cos(),
                theta.cos(),
                theta.sin() * phi.sin(),
            );

            let ray: Ray = Ray {
                origin: vec3(0.0, EARTH_RADIUS + 1.0, 0.0),
                direction: dir,
            };

            col = self.get_incident_light(ray);
        } else {
            let mut eye: Vec3 = vec3(0.0, EARTH_RADIUS + 1.0, 0.0);
            let mut look_at: Vec3 = vec3(0.0, EARTH_RADIUS + 1.5, -1.0);

            let ray: Ray = get_primary_ray(point_cam, &mut eye, &mut look_at);

            if ray.direction.dot(vec3(0.0, 1.0, 0.0)) > 0.0 {
                col = self.get_incident_light(ray);
            } else {
                col = Vec3::splat(0.333);
            }
        }

        *frag_color = col.extend(1.0);
    }
}
