//! Ported to Rust from <https://www.shadertoy.com/view/MlfGR4>
//!
//! Original comment:
//! ```glsl
//! ///////////////////////////////////////////////////////////////////////////////
//! //                                                                           //
//! //  GGGG IIIII  AAA  N   N TTTTT     PPPP   AAA   CCCC     M   M  AAA  N   N //
//! // G       I   A   A NN  N   T       P   P A   A C         MM MM A   A NN  N //
//! // G  GG   I   AAAAA N N N   T       PPPP  AAAAA C     --- M M M AAAAA N N N //
//! // G   G   I   A   A N  NN   T       P     A   A C         M   M A   A N  NN //
//! //  GGGG IIIII A   A N   N   T       P     A   A  CCCC     M   M A   A N   N //
//! //                                                                           //
//! ///////////////////////////////////////////////////////////////////////////////
//! */
//! ```

use shared::*;
use spirv_std::glam::{vec2, vec3, Mat3, Vec2, Vec2Swizzles, Vec3, Vec3Swizzles, Vec4};

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

    // Global variable to handle the glow effect
    glow_counter: f32,
}

impl State {
    pub fn new(inputs: Inputs) -> State {
        State {
            inputs,

            glow_counter: 0.0,
        }
    }
}

// Parameters
const VOXEL_RESOLUTION: f32 = 1.5;
const VOXEL_LIGHTING: bool = true;
const SHADOW: bool = true;
const GROUND: bool = true;
const GHOST: bool = true;
const MOUSE: bool = true;
const HSV2RGB_FAST: bool = true;
const HSV2RGB_SAFE: bool = false;

const CAMERA_FOCAL_LENGTH: f32 = 8.0;
const DELTA: f32 = 0.01;
const RAY_LENGTH_MAX: f32 = 500.0;
const RAY_STEP_MAX: f32 = 100.0;
const AMBIENT: f32 = 0.2;
const SPECULAR_POWER: f32 = 2.0;
const SPECULAR_INTENSITY: f32 = 0.3;
const SHADOW_LENGTH: f32 = 150.0;
const SHADOW_POWER: f32 = 3.0;
const FADE_POWER: f32 = 1.0;
const BACKGROUND: f32 = 0.7;
const GLOW: f32 = 0.4;
const GAMMA: f32 = 0.8;

// Math constants
const PI: f32 = 3.14159265359;
const SQRT3: f32 = 1.73205080757;

// PRNG (from https://www.shadertoy.com/view/4djSRW)
fn rand(mut seed: Vec3) -> f32 {
    seed = (seed * vec3(5.3983, 5.4427, 6.9371)).gl_fract();
    seed += Vec3::splat(seed.yzx().dot(seed + vec3(21.5351, 14.3137, 15.3219)));
    (seed.x * seed.y * seed.z * 95.4337).gl_fract()
}

impl State {
    // Distance to the voxel
    fn dist_voxel(&mut self, p: Vec3) -> f32 {
        // Update the glow counter
        self.glow_counter += 1.0;

        // Rounded box
        let voxel_radius: f32 = 0.25;
        (p.abs() + Vec3::splat(-0.5 + voxel_radius))
            .max(Vec3::zero())
            .length()
            - voxel_radius
    }

    // Distance to the scene and color of the closest point
    fn dist_scene(&mut self, mut p: Vec3, p2: &mut Vec3) -> Vec2 {
        // Update the glow counter
        self.glow_counter += 1.0;

        // Scaling
        p *= VOXEL_RESOLUTION;

        // Velocity, period of the waves, spacing of the gums
        let mut v: f32 = VOXEL_RESOLUTION * (self.inputs.time * 100.0 / VOXEL_RESOLUTION).floor();
        let k1: f32 = 0.05;
        let k2: f32 = 60.0;

        // Giant Pac-Man
        let mut body: f32 = p.length();
        body = (body - 32.0).max(27.0 - body);
        let mut eyes: f32 = 6.0 - vec3(p.x.abs() - 12.5, p.y - 19.5, p.z - 20.0).length();
        let mut mouth_angle: f32 = PI * (0.07 + 0.07 * (2.0 * v * PI / k2).cos());
        let mouth_top: f32 = p.dot(vec3(0.0, -mouth_angle.cos(), mouth_angle.sin())) - 2.0;
        mouth_angle *= 2.5;
        let mouth_bottom: f32 = p.dot(vec3(0.0, mouth_angle.cos(), mouth_angle.sin()));
        let pac_man: f32 = body.max(eyes).max(mouth_top.min(mouth_bottom));
        let mut d: Vec2 = vec2(pac_man, 0.13);
        *p2 = p;

        // Gums
        let mut q: Vec3 = p.xy().extend((p.z + v).rem_euclid(k2) - k2 * 0.5);
        let gum: f32 = (q.length() - 6.0).max(-p.z);
        if gum < d.x {
            d = vec2(gum, 0.35);
            *p2 = q;
        }

        // Ground
        if GROUND {
            q = p.xy().extend(p.z + v);
            let ground: f32 = (q.y + 50.0 + 14.0 * (q.x * k1).cos() * (q.z * k1).cos()) * 0.7;
            if ground < d.x {
                d = vec2(ground, 0.55);
                *p2 = q;
            }
        }

        // Ghost
        if GHOST {
            v = VOXEL_RESOLUTION
                * ((130.0 + 60.0 * (self.inputs.time * 3.0).cos()) / VOXEL_RESOLUTION).floor();
            q = p.xy().extend(p.z + v);
            body = vec3(q.x, (q.y - 4.0).max(0.0), q.z).length();
            body = (body - 28.0).max(22.0 - body);
            eyes = 8.0 - vec3(q.x.abs() - 12.0, q.y - 10.0, q.z - 22.0).length();
            let bottom: f32 = (q.y + 28.0 + 4.0 * (p.x * 0.4).cos() * (p.z * 0.4).cos()) * 0.7;
            let ghost: f32 = body.max(eyes).max(-bottom);
            if ghost < d.x {
                d = vec2(ghost, 0.76);
                *p2 = q;
            }
        }

        // Scaling
        d.x /= VOXEL_RESOLUTION;
        d
    }

    // Distance to the (voxelized?) scene
    fn dist(&mut self, p: &mut Vec3, ray: Vec3, voxelized: f32, ray_length_max: f32) -> Vec4 {
        let mut p2: Vec3 = *p;
        let mut d: Vec2 = vec2(1.0 / 0.0, 0.0);
        let mut ray_length: f32 = 0.0;
        let mut ray_length_in_voxel: f32 = 0.0;
        let mut ray_length_check_voxel: f32 = 0.0;
        let ray_sign: Vec3 = ray.gl_sign();
        let ray_delta_voxel: Vec3 = ray_sign / ray;
        let mut ray_step: f32 = 0.0;
        while ray_step < RAY_STEP_MAX {
            if ray_length < ray_length_in_voxel {
                d.x = self.dist_voxel((*p + Vec3::splat(0.5)).gl_fract() - Vec3::splat(0.5));
                if d.x < DELTA {
                    break;
                }
            } else if ray_length < ray_length_check_voxel {
                let mut ray_delta: Vec3 = (Vec3::splat(0.5)
                    - ray_sign * ((*p + Vec3::splat(0.5)).gl_fract() - Vec3::splat(0.5)))
                    * ray_delta_voxel;
                let d_next: f32 = ray_delta.x.min(ray_delta.y.min(ray_delta.z));
                d = self.dist_scene((*p + Vec3::splat(0.5)).floor(), &mut p2);
                if d.x < 0.0 {
                    ray_delta = ray_delta_voxel - ray_delta;
                    d.x = (ray_length_in_voxel - ray_length)
                        .max(DELTA - ray_delta.x.min(ray_delta.y.min(ray_delta.z)));
                    ray_length_in_voxel = ray_length + d_next;
                } else {
                    d.x = DELTA + d_next;
                }
            } else {
                d = self.dist_scene(*p, &mut p2);
                if voxelized > 0.5 {
                    if d.x < SQRT3 * 0.5 {
                        ray_length_check_voxel = ray_length + d.x.abs() + SQRT3 * 0.5;
                        d.x = (ray_length_in_voxel - ray_length + DELTA).max(d.x - SQRT3 * 0.5);
                    }
                } else if d.x < DELTA {
                    break;
                }
            }
            ray_length += d.x;
            if ray_length > ray_length_max {
                break;
            }
            *p += d.x * ray;
            ray_step += 1.0;
        }
        d.extend(ray_length).extend(rand(p2))
    }

    // Normal at a given point
    fn normal(&mut self, mut p: Vec3, voxelized: f32) -> Vec3 {
        let h: Vec2 = vec2(DELTA, -DELTA);
        let mut n: Vec3 = Vec3::zero();
        if voxelized > 0.5 {
            p = (p + Vec3::splat(0.5)).gl_fract() - Vec3::splat(0.5);
            n = h.xxx() * self.dist_voxel(p + h.xxx())
                + h.xyy() * self.dist_voxel(p + h.xyy())
                + h.yxy() * self.dist_voxel(p + h.yxy())
                + h.yyx() * self.dist_voxel(p + h.yyx());
        } else {
            n = h.xxx() * self.dist_scene(p + h.xxx(), &mut n).x
                + h.xyy() * self.dist_scene(p + h.xyy(), &mut n).x
                + h.yxy() * self.dist_scene(p + h.yxy(), &mut n).x
                + h.yyx() * self.dist_scene(p + h.yyx(), &mut n).x;
        }
        n.normalize()
    }
}

// HSV to RGB
fn hsv2rgb(mut hsv: Vec3) -> Vec3 {
    if HSV2RGB_SAFE {
        hsv = hsv
            .yz()
            .clamp(Vec2::zero(), Vec2::one())
            .extend(hsv.x)
            .zxy();
    }
    if HSV2RGB_FAST {
        hsv.z
            * (Vec3::one()
                + 0.5
                    * hsv.y
                    * ((2.0 * PI * (Vec3::splat(hsv.x) + vec3(0.0, 2.0 / 3.0, 1.0 / 3.0))).cos()
                        - Vec3::one()))
    } else {
        hsv.z
            * (Vec3::one()
                + Vec3::splat(hsv.y)
                    * (((Vec3::splat(hsv.x) + vec3(0.0, 2.0 / 3.0, 1.0 / 3.0)).gl_fract() * 6.0
                        - Vec3::splat(3.0))
                    .abs()
                        - Vec3::splat(2.0))
                    .clamp(-Vec3::one(), Vec3::zero()))
    }
}

impl State {
    // Main function
    pub fn main_image(&mut self, frag_color: &mut Vec4, frag_coord: Vec2) {
        // Get the fragment
        let frag: Vec2 =
            (2.0 * frag_coord - self.inputs.resolution.xy()) / self.inputs.resolution.y;

        // Define the rendering mode
        let mut mode_timing: f32 = self.inputs.time * 0.234;
        let mut mode_angle: f32 = PI * (self.inputs.time * 0.2).cos();
        mode_angle = (frag - vec2((self.inputs.time * 2.0).cos(), 0.0))
            .dot(vec2(mode_angle.cos(), mode_angle.sin()));
        let mut mode_voxel: f32 = 0.5.step((mode_timing / (4.0 * PI)).gl_fract());
        mode_timing = mode_timing.cos();
        let mode_3d: f32 = smoothstep(0.8, 0.5, mode_timing);
        let mode_switch: f32 = smoothstep(0.995, 1.0, mode_timing)
            + smoothstep(0.02, 0.0, mode_angle.abs()) * mode_voxel;
        mode_voxel = 1.0 + (0.0.step(mode_angle) - 1.0) * mode_voxel;

        // Define the ray corresponding to this fragment
        let mut ray: Vec3 = frag
            .extend(mix(8.0, CAMERA_FOCAL_LENGTH, mode_3d))
            .normalize();

        // Compute the orientation of the camera
        let mut yaw_angle: f32 = PI * (1.2 + 0.2 * (self.inputs.time * 0.5).cos());
        let mut pitch_angle: f32 = PI * (0.1 * (self.inputs.time * 0.3).cos() - 0.05);
        if MOUSE {
            yaw_angle += 4.0 * PI * self.inputs.mouse.x / self.inputs.resolution.x;
            pitch_angle += PI * 0.3 * (1.0 - self.inputs.mouse.y / self.inputs.resolution.y);
        }
        yaw_angle = mix(PI * 1.5, yaw_angle, mode_3d);
        pitch_angle *= mode_3d;

        let cos_yaw: f32 = yaw_angle.cos();
        let sin_yaw: f32 = yaw_angle.sin();
        let cos_pitch: f32 = pitch_angle.cos();
        let sin_pitch: f32 = pitch_angle.sin();

        let mut camera_orientation: Mat3 = Mat3::zero();
        camera_orientation.x_axis = vec3(cos_yaw, 0.0, -sin_yaw);
        camera_orientation.y_axis = vec3(sin_yaw * sin_pitch, cos_pitch, cos_yaw * sin_pitch);
        camera_orientation.z_axis = vec3(sin_yaw * cos_pitch, -sin_pitch, cos_yaw * cos_pitch);

        ray = camera_orientation * ray;

        // Compute the origin of the ray
        let camera_dist: f32 = mix(
            300.0,
            195.0 + 150.0 * (self.inputs.time * 0.8).cos(),
            mode_3d,
        );
        let mut origin: Vec3 = (vec3(0.0, 0.0, 40.0 * (self.inputs.time * 0.2).sin())
            - camera_orientation.z_axis * camera_dist)
            / VOXEL_RESOLUTION;

        // Compute the distance to the scene
        self.glow_counter = 0.0;
        let d: Vec4 = self.dist(
            &mut origin,
            ray,
            mode_voxel,
            RAY_LENGTH_MAX / VOXEL_RESOLUTION,
        );

        // Set the background color
        let mut final_color: Vec3 = hsv2rgb(vec3(
            0.2 * ray.y + 0.4 * mode_voxel - 0.37,
            1.0,
            mode_3d * BACKGROUND,
        ));
        let glow_color: Vec3 = GLOW * vec3(1.0, 0.3, 0.0) * self.glow_counter / RAY_STEP_MAX;
        if d.x < DELTA {
            // Set the object color
            let mut color: Vec3 = hsv2rgb(vec3(
                d.y + 0.1 * d.w * mode_voxel,
                0.5 + 0.5 * mode_voxel,
                1.0,
            ));

            // Lighting
            let l: Vec3 = mix(
                vec3(1.0, 0.0, 0.0),
                vec3(1.25 + (self.inputs.time * 0.2).cos(), 1.0, 1.0),
                mode_3d,
            )
            .normalize();
            if VOXEL_LIGHTING {
                if mode_voxel > 0.5 {
                    let n: Vec3 = self.normal((origin + Vec3::splat(0.5)).floor(), 0.0);
                    let diffuse: f32 = n.dot(l).max(0.0);
                    let specular: f32 =
                        ray.reflect(n).dot(l).max(0.0).powf(SPECULAR_POWER) * SPECULAR_INTENSITY;
                    color = (AMBIENT + diffuse) * color + Vec3::splat(specular);
                }
            }
            let n: Vec3 = self.normal(origin, mode_voxel);
            let mut diffuse: f32 = n.dot(l);
            let mut specular: f32;
            if diffuse < 0.0 {
                diffuse = 0.0;
                specular = 0.0;
            } else {
                specular = ray.reflect(n).dot(l).max(0.0).powf(SPECULAR_POWER) * SPECULAR_INTENSITY;
                if SHADOW {
                    origin += n * DELTA * 2.0;
                    let mut shadow: Vec4 =
                        self.dist(&mut origin, l, mode_voxel, SHADOW_LENGTH / VOXEL_RESOLUTION);
                    if shadow.x < DELTA {
                        shadow.z = (shadow.z * VOXEL_RESOLUTION / SHADOW_LENGTH)
                            .min(1.0)
                            .powf(SHADOW_POWER);
                        diffuse *= shadow.z;
                        specular *= shadow.z;
                    }
                }
            }
            color = (AMBIENT + diffuse) * color + Vec3::splat(specular);

            // Fading
            let fade: f32 = (1.0 - d.z * VOXEL_RESOLUTION / RAY_LENGTH_MAX)
                .max(0.0)
                .powf(FADE_POWER);
            final_color = mix(final_color, color, fade);
        }

        // Set the fragment color
        final_color = mix(
            final_color.powf(GAMMA) + glow_color,
            Vec3::one(),
            mode_switch,
        );
        *frag_color = final_color.extend(1.0);
    }
}
