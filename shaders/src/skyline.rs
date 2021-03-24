//! Ported to Rust from <https://www.shadertoy.com/view/XtsSWs>
//!
//! Original comment:
//! ```glsl
//! /*--------------------------------------------------------------------------------------
//! License CC0 - http://creativecommons.org/publicdomain/zero/1.0/
//! To the extent possible under law, the author(s) have dedicated all copyright and related and neighboring rights to this software to the public domain worldwide. This software is distributed without any warranty.
//! ----------------------------------------------------------------------------------------
//! ^ This means do ANYTHING YOU WANT with this code. Because we are programmers, not lawyers.
//! -Otavio Good
//! */
//! ```

use crate::SampleCube;
use glam::{const_vec2, vec2, vec3, Vec2, Vec2Swizzles, Vec3, Vec3Swizzles, Vec4, Vec4Swizzles};
use shared::*;

// Note: This cfg is incorrect on its surface, it really should be "are we compiling with std", but
// we tie #[no_std] above to the same condition, so it's fine.
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

pub struct Inputs<C0> {
    pub resolution: Vec3,
    pub time: f32,
    pub mouse: Vec4,
    pub channel0: C0,
}

pub struct State<C0> {
    inputs: Inputs<C0>,

    // --------------------------------------------------------
    // These variables are for the non-realtime block renderer.
    local_time: f32,
    seed: f32,

    // Animation variables
    fade: f32,
    sun_dir: Vec3,
    sun_col: Vec3,
    exposure: f32,
    sky_col: Vec3,
    horizon_col: Vec3,

    // other
    march_count: f32,
}

impl<C0> State<C0> {
    pub fn new(inputs: Inputs<C0>) -> Self {
        State {
            inputs,

            local_time: 0.0,
            seed: 1.0,

            fade: 1.0,
            sun_dir: Vec3::ZERO,
            sun_col: Vec3::ZERO,
            exposure: 1.0,
            sky_col: Vec3::ZERO,
            horizon_col: Vec3::ZERO,

            march_count: 0.0,
        }
    }
}

// ---------------- Config ----------------
// This is an option that lets you render high quality frames for screenshots. It enables
// stochastic antialiasing and motion blur automatically for any shader.
const NON_REALTIME_HQ_RENDER: bool = false;
const FRAME_TO_RENDER_HQ: f32 = 50.0; // Time in seconds of frame to render
const ANTIALIASING_SAMPLES: f32 = 16.0; // 16x antialiasing - too much might make the shader compiler angry.

const MANUAL_CAMERA: bool = false;

// ---- noise functions ----
fn _v31(a: Vec3) -> f32 {
    a.x + a.y * 37.0 + a.z * 521.0
}
fn v21(a: Vec2) -> f32 {
    a.x + a.y * 37.0
}
fn hash11(a: f32) -> f32 {
    (a.sin() * 10403.9).gl_fract()
}
fn hash21(uv: Vec2) -> f32 {
    let f: f32 = uv.x + uv.y * 37.0;
    (f.sin() * 104003.9).gl_fract()
}
fn hash22(uv: Vec2) -> Vec2 {
    let f: f32 = uv.x + uv.y * 37.0;
    (f.cos() * vec2(10003.579, 37049.7)).gl_fract()
}
fn _hash12(f: f32) -> Vec2 {
    (f.cos() * vec2(10003.579, 37049.7)).gl_fract()
}
fn _hash1d(u: f32) -> f32 {
    (u.sin() * 143.9).gl_fract() // scale this down to kill the jitters
}
fn hash2d(uv: Vec2) -> f32 {
    let f: f32 = uv.x + uv.y * 37.0;
    (f.sin() * 104003.9).gl_fract()
}
fn hash3d(uv: Vec3) -> f32 {
    let f: f32 = uv.x + uv.y * 37.0 + uv.z * 521.0;
    (f.sin() * 110003.9).gl_fract()
}
fn mix_p(f0: f32, f1: f32, a: f32) -> f32 {
    mix(f0, f1, a * a * (3.0 - 2.0 * a))
}
const ZERO_ONE: Vec2 = const_vec2!([0.0, 1.0]);
fn noise2d(uv: Vec2) -> f32 {
    let fr: Vec2 = uv.gl_fract();
    let fl: Vec2 = uv.floor();
    let h00: f32 = hash2d(fl);
    let h10: f32 = hash2d(fl + ZERO_ONE.yx());
    let h01: f32 = hash2d(fl + ZERO_ONE);
    let h11: f32 = hash2d(fl + ZERO_ONE.yy());
    mix_p(mix_p(h00, h10, fr.x), mix_p(h01, h11, fr.x), fr.y)
}
fn noise(uv: Vec3) -> f32 {
    let fr: Vec3 = uv.gl_fract();
    let fl: Vec3 = uv.floor();
    let h000: f32 = hash3d(fl);
    let h100: f32 = hash3d(fl + ZERO_ONE.yxx());
    let h010: f32 = hash3d(fl + ZERO_ONE.xyx());
    let h110: f32 = hash3d(fl + ZERO_ONE.yyx());
    let h001: f32 = hash3d(fl + ZERO_ONE.xxy());
    let h101: f32 = hash3d(fl + ZERO_ONE.yxy());
    let h011: f32 = hash3d(fl + ZERO_ONE.xyy());
    let h111: f32 = hash3d(fl + ZERO_ONE.yyy());
    mix_p(
        mix_p(mix_p(h000, h100, fr.x), mix_p(h010, h110, fr.x), fr.y),
        mix_p(mix_p(h001, h101, fr.x), mix_p(h011, h111, fr.x), fr.y),
        fr.z,
    )
}

const PI: f32 = 3.14159265;

fn saturate_vec3(a: Vec3) -> Vec3 {
    a.clamp(Vec3::ZERO, Vec3::ONE)
}
fn _saturate_vec2(a: Vec2) -> Vec2 {
    a.clamp(Vec2::ZERO, Vec2::ONE)
}
fn saturate(a: f32) -> f32 {
    a.clamp(0.0, 1.0)
}

impl<C0> State<C0> {
    // This function basically is a procedural environment map that makes the sun
    fn get_sun_color_small(&self, ray_dir: Vec3, sun_dir: Vec3) -> Vec3 {
        let local_ray: Vec3 = ray_dir.normalize();
        let dist: f32 = 1.0 - (local_ray.dot(sun_dir) * 0.5 + 0.5);
        let mut sun_intensity: f32 = 0.05 / dist;
        sun_intensity += (-dist * 150.0).exp() * 7000.0;
        sun_intensity = sun_intensity.min(40000.0);
        self.sun_col * sun_intensity * 0.025
    }

    fn get_env_map(&self, ray_dir: Vec3, sun_dir: Vec3) -> Vec3 {
        // fade the sky color, multiply sunset dimming
        let mut final_color: Vec3 = mix(
            self.horizon_col,
            self.sky_col,
            saturate(ray_dir.y).powf(0.47),
        ) * 0.95;
        // make clouds - just a horizontal plane with noise
        let mut n: f32 = noise2d(ray_dir.xz() / ray_dir.y * 1.0);
        n += noise2d(ray_dir.xz() / ray_dir.y * 2.0) * 0.5;
        n += noise2d(ray_dir.xz() / ray_dir.y * 4.0) * 0.25;
        n += noise2d(ray_dir.xz() / ray_dir.y * 8.0) * 0.125;
        n = n.abs().powf(3.0);
        n = mix(n * 0.2, n, saturate((ray_dir.y * 8.0).abs())); // fade clouds in distance
        final_color = mix(
            final_color,
            (Vec3::ONE + self.sun_col * 10.0) * 0.75 * saturate((ray_dir.y + 0.2) * 5.0),
            saturate(n * 0.125),
        );

        // add the sun
        final_color += self.get_sun_color_small(ray_dir, sun_dir);
        final_color
    }

    fn get_env_map_skyline(&self, ray_dir: Vec3, sun_dir: Vec3, height: f32) -> Vec3 {
        let mut final_color: Vec3 = self.get_env_map(ray_dir, sun_dir);

        // Make a skyscraper skyline reflection.
        let mut radial: f32 = ray_dir.z.atan2(ray_dir.x) * 4.0;
        let mut skyline: f32 =
            (((5.3456 * radial).sin() + (1.234 * radial).sin() + (2.177 * radial).sin()) * 0.6)
                .floor();
        radial *= 4.0;
        skyline += (((5.0 * radial).sin() + (1.234 * radial).sin() + (2.177 * radial).sin()) * 0.6)
            .floor()
            * 0.1;
        let mut mask: f32 = saturate((ray_dir.y * 8.0 - skyline - 2.5 + height) * 24.0);
        let vert: f32 = (radial * 32.0).sin().gl_sign() * 0.5 + 0.5;
        let hor: f32 = (ray_dir.y * 256.0).sin().gl_sign() * 0.5 + 0.5;
        mask = saturate(mask + (1.0 - hor * vert) * 0.05);
        final_color = mix(final_color * vec3(0.1, 0.07, 0.05), final_color, mask);

        final_color
    }
}

// min function that supports materials in the y component
fn matmin(a: Vec2, b: Vec2) -> Vec2 {
    if a.x < b.x {
        a
    } else {
        b
    }
}

// ---- shapes defined by distance fields ----
// See this site for a reference to more distance functions...
// http://iquilezles.org/www/articles/distfunctions/distfunctions.htm

// signed box distance field
fn sd_box(p: Vec3, radius: Vec3) -> f32 {
    let dist: Vec3 = p.abs() - radius;
    dist.x.max(dist.y.max(dist.z)).min(0.0) + dist.max(Vec3::ZERO).length()
}

// capped cylinder distance field
fn cyl_cap(p: Vec3, r: f32, len_rad: f32) -> f32 {
    let mut a: f32 = p.xy().length() - r;
    a = a.max(p.z.abs() - len_rad);
    a
}

// k should be negative. -4.0 works nicely.
// smooth blending function
fn smin(a: f32, b: f32, k: f32) -> f32 {
    ((k * a).exp2() + (k * b).exp2()).log2() / k
}

fn repeat(a: f32, len: f32) -> f32 {
    a.rem_euclid(len) - 0.5 * len
}

// Distance function that defines the car.
// Basically it's 2 boxes smooth-blended together and a mirrored cylinder for the wheels.
fn car(base_center: Vec3, unique: f32) -> Vec2 {
    // bottom box
    let mut car: f32 = sd_box(
        base_center + vec3(0.0, -0.008, 0.001),
        vec3(0.01, 0.00225, 0.0275),
    );
    // top box smooth blended
    car = smin(
        car,
        sd_box(
            base_center + vec3(0.0, -0.016, 0.008),
            vec3(0.005, 0.0005, 0.01),
        ),
        -160.0,
    );
    // mirror the z axis to duplicate the cylinders for wheels
    let mut w_mirror: Vec3 = base_center + vec3(0.0, -0.005, 0.0);
    w_mirror.z = w_mirror.z.abs() - 0.02;
    let wheels: f32 = cyl_cap((w_mirror).zyx(), 0.004, 0.0135);
    // Set materials
    let mut dist_and_mat: Vec2 = vec2(wheels, 3.0); // car wheels
                                                    // Car material is some big number that's unique to each car
                                                    // so I can have each car be a different color
    dist_and_mat = matmin(dist_and_mat, vec2(car, 100000.0 + unique)); // car
    dist_and_mat
}

// How much space between voxel borders and geometry for voxel ray march optimization
const VOXEL_PAD: f32 = 0.2;
// p should be in [0..1] range on xz plane
// pint is an integer pair saying which city block you are on
fn city_block(p: Vec3, pint: Vec2) -> Vec2 {
    // Get random numbers for this block by hashing the city block variable
    let mut rand: Vec4 = Vec4::ZERO;
    rand = hash22(pint).extend(rand.z).extend(rand.w);
    rand = hash22(rand.xy()).extend(rand.x).extend(rand.y).zwxy();
    let mut rand2: Vec2 = hash22(rand.zw());

    // Radius of the building
    let mut base_rad: f32 = 0.2 + (rand.x) * 0.1;
    base_rad = (base_rad * 20.0 + 0.5).floor() / 20.0; // try to snap this for window texture

    // make position relative to the middle of the block
    let base_center: Vec3 = p - vec3(0.5, 0.0, 0.5);
    let mut height: f32 = rand.w * rand.z + 0.1; // height of first building block
                                                 // Make the city skyline higher in the middle of the city.
    let downtown: f32 = saturate(4.0 / pint.length());
    height *= downtown;
    height *= 1.5 + (base_rad - 0.15) * 20.0;
    height += 0.1; // minimum building height
                   //height += sin(iTime + pint.x);	// animate the building heights if you're feeling silly
    height = (height * 20.0).floor() * 0.05; // height is in floor units - each floor is 0.05 high.
    let mut d: f32 = sd_box(base_center, vec3(base_rad, height, base_rad)); // large building piece

    // road
    d = d.min(p.y);

    //if (length(pint.xy) > 8.0) return vec2(d, mat);	// Hack to LOD in the distance

    // height of second building section
    let mut height2: f32 = (rand.y * 2.0 - 1.0).max(0.0) * downtown;
    height2 = (height2 * 20.0).floor() * 0.05; // floor units
    rand2 = (rand2 * 20.0).floor() * 0.05; // floor units
                                           // size pieces of building
    d = d.min(sd_box(
        base_center - vec3(0.0, height, 0.0),
        vec3(base_rad, height2 - rand2.y, base_rad * 0.4),
    ));
    d = d.min(sd_box(
        base_center - vec3(0.0, height, 0.0),
        vec3(base_rad * 0.4, height2 - rand2.x, base_rad),
    ));
    // second building section
    if rand2.y > 0.25 {
        d = d.min(sd_box(
            base_center - vec3(0.0, height, 0.0),
            vec3(base_rad * 0.8, height2, base_rad * 0.8),
        ));
        // subtract off piece from top so it looks like there's a wall around the roof.
        let mut top_width: f32 = base_rad;
        if height2 > 0.0 {
            top_width = base_rad * 0.8;
        }
        d = d.max(-sd_box(
            base_center - vec3(0.0, height + height2, 0.0),
            vec3(top_width - 0.0125, 0.015, top_width - 0.0125),
        ));
    } else {
        // Cylinder top section of building
        if height2 > 0.0 {
            d = d.min(cyl_cap(
                (base_center - vec3(0.0, height, 0.0)).xzy(),
                base_rad * 0.8,
                height2,
            ));
        }
    }
    // mini elevator shaft boxes on top of building
    d = d.min(sd_box(
        base_center
            - vec3(
                (rand.x - 0.5) * base_rad,
                height + height2,
                (rand.y - 0.5) * base_rad,
            ),
        vec3(
            base_rad * 0.3 * rand.z,
            0.1 * rand2.y,
            base_rad * 0.3 * rand2.x + 0.025,
        ),
    ));
    // mirror another box (and scale it) so we get 2 boxes for the price of 1.
    let mut box_pos: Vec3 = base_center
        - vec3(
            (rand2.x - 0.5) * base_rad,
            height + height2,
            (rand2.y - 0.5) * base_rad,
        );
    let big: f32 = box_pos.x.gl_sign();
    box_pos.x = box_pos.x.abs() - 0.02 - base_rad * 0.3 * rand.w;
    d = d.min(sd_box(
        box_pos,
        vec3(
            base_rad * 0.3 * rand.w,
            0.07 * rand.y,
            base_rad * 0.2 * rand.x + big * 0.025,
        ),
    ));

    // Put domes on some building tops for variety
    if rand.y < 0.04 {
        d = d.min((base_center - vec3(0.0, height, 0.0)).length() - base_rad * 0.8);
    }

    //d = max(d, p.y);  // flatten the city for debugging cars

    // Need to make a material variable.
    let mut dist_and_mat: Vec2 = vec2(d, 0.0);
    // sidewalk box with material
    dist_and_mat = matmin(
        dist_and_mat,
        vec2(sd_box(base_center, vec3(0.35, 0.005, 0.35)), 1.0),
    );

    dist_and_mat
}

impl<C0> State<C0> {
    // This is the distance function that defines all the scene's geometry.
    // The input is a position in space.
    // The output is the distance to the nearest surface and a material index.
    fn distance_to_object(&self, p: Vec3) -> Vec2 {
        //p.y += noise2d((p.xz)*0.0625)*8.0; // Hills
        let mut rep: Vec3 = p;
        rep = (p.xz().gl_fract()).extend(rep.y).xzy(); // [0..1] for representing the position in the city block
        let mut dist_and_mat: Vec2 = city_block(rep, p.xz().floor());

        // Set up the cars. This is doing a lot of mirroring and repeating because I
        // only want to do a single call to the car distance function for all the
        // cars in the scene. And there's a lot of traffic!
        let mut p2: Vec3 = p;
        rep = p2;
        let car_time: f32 = self.local_time * 0.2; // Speed of car driving
        let mut cross_street: f32 = 1.0; // whether we are north/south or east/west
        let mut repeat_dist: f32 = 0.25; // Car density bumper to bumper
                                         // If we are going north/south instead of east/west (?) make cars that are
                                         // stopped in the street so we don't have collisions.
        if (rep.x.gl_fract() - 0.5).abs() < 0.35 {
            p2.x += 0.05;
            p2 = (p2.zx() * vec2(-1.0, 1.0)).extend(p2.y).xzy(); // Rotate 90 degrees
            rep = p2.xz().extend(rep.y).xzy();
            cross_street = 0.0;
            repeat_dist = 0.1; // Denser traffic on cross streets
        }

        rep.z += p2.x.floor(); // shift so less repitition between parallel blocks
        rep.x = repeat(p2.x - 0.5, 1.0); // repeat every block
        rep.z = rep.z * rep.x.gl_sign(); // mirror but keep cars facing the right way
        rep.x = (rep.x * rep.x.gl_sign()) - 0.09;
        rep.z -= car_time * cross_street; // make cars move
        let unique_id: f32 = (rep.z / repeat_dist).floor(); // each car gets a unique ID that we can use for colors
        rep.z = repeat(rep.z, repeat_dist); // repeat the line of cars every quarter block
        rep.x += hash11(unique_id) * 0.075 - 0.01; // nudge cars left and right to take both lanes
        let mut front_back: f32 = hash11(unique_id * 0.987) * 0.18 - 0.09;
        front_back *= (self.local_time * 2.0 + unique_id).sin();
        rep.z += front_back * cross_street; // nudge cars forward back for variation
        let car_dist: Vec2 = car(rep, unique_id); // car distance function

        // Drop the cars in the scene with materials
        dist_and_mat = matmin(dist_and_mat, car_dist);

        dist_and_mat
    }
}

// This basically makes a procedural texture map for the sides of the buildings.
// It makes a texture, a normal for normal mapping, and a mask for window reflection.
fn calc_windows(
    block: Vec2,
    pos: Vec3,
    tex_color: &mut Vec3,
    window_ref: &mut f32,
    normal: &mut Vec3,
) {
    let hue: Vec3 = vec3(
        hash21(block) * 0.8,
        hash21(block * 7.89) * 0.4,
        hash21(block * 37.89) * 0.5,
    );
    *tex_color += hue * 0.4;
    *tex_color *= 0.75;
    let mut window: f32 = 0.0;
    window = window.max(mix(
        0.2,
        1.0,
        ((pos.y * 20.0 - 0.35).gl_fract() * 2.0 + 0.1).floor(),
    ));
    if pos.y < 0.05 {
        window = 1.0;
    }
    let mut win_width: f32 = hash21(block * 4.321) * 2.0;
    if (win_width < 1.3) && (win_width >= 1.0) {
        win_width = 1.3;
    }
    window = window.max(mix(
        0.2,
        1.0,
        ((pos.x * 40.0 + 0.05).gl_fract() * win_width).floor(),
    ));
    window = window.max(mix(
        0.2,
        1.0,
        ((pos.z * 40.0 + 0.05).gl_fract() * win_width).floor(),
    ));
    if window < 0.5 {
        *window_ref += 1.0;
    }
    window *= hash21(block * 1.123);
    *tex_color *= window;

    let wave: f32 = (((pos.y * 40.0 - 0.1) * PI).sin() * 0.505 - 0.5).floor() + 1.0;
    normal.y -= (-1.0_f32).max(1.0_f32.min(-wave * 0.5));
    let mut pits: f32 = 1.0_f32.min(((pos.z * 80.0) * PI).sin().abs() * 4.0) - 1.0;
    normal.z += pits * 0.25;
    pits = 1.0_f32.min(((pos.x * 80.0) * PI).sin().abs() * 4.0) - 1.0;
    normal.x += pits * 0.25;
}

impl<C0: SampleCube> State<C0> {
    // Input is UV coordinate of pixel to render.
    // Output is RGB color.
    fn ray_trace(&mut self, frag_coord: Vec2) -> Vec3 {
        self.march_count = 0.0;
        // -------------------------------- animate ---------------------------------------
        self.sun_col = vec3(258.0, 248.0, 200.0) / 3555.0;
        self.sun_dir = vec3(0.93, 1.0, 1.0).normalize();
        self.horizon_col = vec3(1.0, 0.95, 0.85) * 0.9;
        self.sky_col = vec3(0.3, 0.5, 0.95);
        self.exposure = 1.0;
        self.fade = 1.0;

        let mut cam_pos: Vec3 = Vec3::ZERO;
        let mut cam_up: Vec3 = Vec3::ZERO;
        let mut cam_lookat: Vec3 = Vec3::ZERO;
        // ------------------- Set up the camera rays for ray marching --------------------
        // Map uv to [-1.0..1.0]
        let mut uv: Vec2 = frag_coord / self.inputs.resolution.xy() * 2.0 - Vec2::ONE;
        uv /= 2.0; // zoom in

        if MANUAL_CAMERA {
            // Camera up vector.
            cam_up = vec3(0.0, 1.0, 0.0);

            // Camera lookat.
            cam_lookat = vec3(0.0, 0.0, 0.0);

            // debugging camera
            let mx: f32 = -self.inputs.mouse.x / self.inputs.resolution.x * PI * 2.0; // + localTime * 0.05;
            let my: f32 = self.inputs.mouse.y / self.inputs.resolution.y * 3.14 * 0.5 + PI / 2.0; // + sin(localTime * 0.3)*0.8+0.1;//*PI/2.01;
            cam_pos = vec3(my.cos() * mx.cos(), my.sin(), my.cos() * mx.sin()) * 7.35;
        //7.35
        } else {
            // Do the camera fly-by animation and different scenes.
            // Time variables for start and end of each scene
            let t0: f32 = 0.0;
            let t1: f32 = 8.0;
            let t2: f32 = 14.0;
            let t3: f32 = 24.0;
            let t4: f32 = 38.0;
            let t5: f32 = 56.0;
            let t6: f32 = 58.0;
            /*let t0: f32 = 0.0;
            let t1: f32 = 0.0;
            let t2: f32 = 0.0;
            let t3: f32 = 0.0;
            let t4: f32 = 0.0;
            let t5: f32 = 16.0;
            let t6: f32 = 18.0;*/
            // Repeat the animation after time t6
            self.local_time = (self.local_time / t6).gl_fract() * t6;
            if self.local_time < t1 {
                let time: f32 = self.local_time - t0;
                let alpha: f32 = time / (t1 - t0);
                self.fade = saturate(time);
                self.fade *= saturate(t1 - self.local_time);
                cam_pos = vec3(13.0, 3.3, -3.5);
                cam_pos.x -= smoothstep(0.0, 1.0, alpha) * 4.8;
                cam_up = vec3(0.0, 1.0, 0.0);
                cam_lookat = vec3(0.0, 1.5, 1.5);
            } else if self.local_time < t2 {
                let time: f32 = self.local_time - t1;
                let alpha: f32 = time / (t2 - t1);
                self.fade = saturate(time);
                self.fade *= saturate(t2 - self.local_time);
                cam_pos = vec3(26.0, 0.05 + smoothstep(0.0, 1.0, alpha) * 0.4, 2.0);
                cam_pos.z -= alpha * 2.8;
                cam_up = vec3(0.0, 1.0, 0.0);
                cam_lookat = vec3(cam_pos.x - 0.3, -8.15, -40.0);

                self.sun_dir = vec3(0.95, 0.6, 1.0).normalize();
                self.sun_col = vec3(258.0, 248.0, 160.0) / 3555.0;
                self.exposure *= 0.7;
                self.sky_col *= 1.5;
            } else if self.local_time < t3 {
                let time: f32 = self.local_time - t2;
                let alpha: f32 = time / (t3 - t2);
                self.fade = saturate(time);
                self.fade *= saturate(t3 - self.local_time);
                cam_pos = vec3(12.0, 6.3, -0.5);
                cam_pos.y -= alpha * 5.5;
                cam_pos.x = (alpha * 1.0).cos() * 5.2;
                cam_pos.z = (alpha * 1.0).sin() * 5.2;
                cam_up = vec3(0.0, 1.0, -0.5 + alpha * 0.5).normalize();
                cam_lookat = vec3(0.0, 1.0, -0.5);
            } else if self.local_time < t4 {
                let time: f32 = self.local_time - t3;
                let alpha: f32 = time / (t4 - t3);
                self.fade = saturate(time);
                self.fade *= saturate(t4 - self.local_time);
                cam_pos = vec3(2.15 - alpha * 0.5, 0.02, -1.0 - alpha * 0.2);
                cam_pos.y += smoothstep(0.0, 1.0, alpha * alpha) * 3.4;
                cam_up = vec3(0.0, 1.0, 0.0).normalize();
                cam_lookat = vec3(0.0, 0.5 + alpha, alpha * 5.0);
            } else if self.local_time < t5 {
                let time: f32 = self.local_time - t4;
                let alpha: f32 = time / (t5 - t4);
                self.fade = saturate(time);
                self.fade *= saturate(t5 - self.local_time);
                cam_pos = vec3(-2.0, 1.3 - alpha * 1.2, -10.5 - alpha * 0.5);
                cam_up = vec3(0.0, 1.0, 0.0).normalize();
                cam_lookat = vec3(-2.0, 0.3 + alpha, -0.0);
                self.sun_dir = vec3(0.5 - alpha * 0.6, 0.3 - alpha * 0.3, 1.0).normalize();
                self.sun_col = vec3(258.0, 148.0, 60.0) / 3555.0;
                self.local_time *= 16.0;
                self.exposure *= 0.4;
                self.horizon_col = vec3(1.0, 0.5, 0.35) * 2.0;
                self.sky_col = vec3(0.75, 0.5, 0.95);
            } else if self.local_time < t6 {
                self.fade = 0.0;
                cam_pos = vec3(26.0, 100.0, 2.0);
                cam_up = vec3(0.0, 1.0, 0.0);
                cam_lookat = vec3(0.3, 0.15, 0.0);
            }
        }

        // Camera setup for ray tracing / marching
        let cam_vec: Vec3 = (cam_lookat - cam_pos).normalize();
        let side_norm: Vec3 = cam_up.cross(cam_vec).normalize();
        let up_norm: Vec3 = cam_vec.cross(side_norm);
        let world_facing: Vec3 = cam_pos + cam_vec;
        let world_pix: Vec3 = world_facing
            + uv.x * side_norm * (self.inputs.resolution.x / self.inputs.resolution.y)
            + uv.y * up_norm;
        let ray_vec: Vec3 = (world_pix - cam_pos).normalize();

        // ----------------------------- Ray march the scene ------------------------------
        let mut dist_and_mat: Vec2 = Vec2::ZERO; // Distance and material
        let mut t: f32 = 0.05; // + Hash2d(uv)*0.1;	// random dither-fade things close to the camera
        let max_depth: f32 = 45.0; // farthest distance rays will travel
        let mut pos: Vec3 = Vec3::ZERO;
        let small_val: f32 = 0.000625;
        // ray marching time
        let mut i = 0;

        while i < 250 {
            // This is the count of the max times the ray actually marches.
            self.march_count += 1.0;
            // Step along the ray.
            pos = cam_pos + ray_vec * t;
            // This is _the_ function that defines the "distance field".
            // It's really what makes the scene geometry. The idea is that the
            // distance field returns the distance to the closest object, and then
            // we know we are safe to "march" along the ray by that much distance
            // without hitting anything. We repeat this until we get really close
            // and then break because we have effectively hit the object.
            dist_and_mat = self.distance_to_object(pos);

            // 2d voxel walk through the city blocks.
            // The distance function is not continuous at city block boundaries,
            // so we have to pause our ray march at each voxel boundary.
            let mut walk: f32 = dist_and_mat.x;
            let mut dx: f32 = -pos.x.gl_fract();
            if ray_vec.x > 0.0 {
                dx = (-pos.x).gl_fract();
            }
            let mut dz: f32 = -pos.z.gl_fract();
            if ray_vec.z > 0.0 {
                dz = (-pos.z).gl_fract();
            }
            let mut nearest_voxel: f32 =
                (dx / ray_vec.x).gl_fract().min((dz / ray_vec.z).gl_fract()) + VOXEL_PAD;
            nearest_voxel = VOXEL_PAD.max(nearest_voxel); // hack that assumes streets and sidewalks are this wide.
                                                          //nearestVoxel = nearestVoxel.max(t * 0.02); // hack to stop voxel walking in the distance.
            walk = walk.min(nearest_voxel);

            // move down the ray a safe amount
            t += walk;
            // If we are very close to the object, let's call it a hit and exit this loop.
            if (t > max_depth) || (dist_and_mat.x.abs() < small_val) {
                break;
            }
            i += 1;
        }

        // Ray trace a ground plane to infinity
        let alpha: f32 = -cam_pos.y / ray_vec.y;
        if (t > max_depth) && (ray_vec.y < -0.0) {
            pos = (cam_pos.xz() + ray_vec.xz() * alpha).extend(pos.y).xzy();
            pos.y = -0.0;
            t = alpha;
            dist_and_mat.y = 0.0;
            dist_and_mat.x = 0.0;
        }
        // --------------------------------------------------------------------------------
        // Now that we have done our ray marching, let's put some color on this geometry.
        let mut final_color: Vec3;

        // If a ray actually hit the object, let's light it.
        if (t <= max_depth) || (t == alpha) {
            let dist: f32 = dist_and_mat.x;
            // calculate the normal from the distance field. The distance field is a volume, so if you
            // sample the current point and neighboring points, you can use the difference to get
            // the normal.
            let small_vec: Vec3 = vec3(small_val, 0.0, 0.0);
            let normal_u: Vec3 = vec3(
                dist - self.distance_to_object(pos - small_vec.xyy()).x,
                dist - self.distance_to_object(pos - small_vec.yxy()).x,
                dist - self.distance_to_object(pos - small_vec.yyx()).x,
            );
            let mut normal: Vec3 = normal_u.normalize();

            // calculate 2 ambient occlusion values. One for global stuff and one
            // for local stuff
            let mut ambient_s: f32 = 1.0;
            ambient_s *= saturate(self.distance_to_object(pos + normal * 0.0125).x * 80.0);
            ambient_s *= saturate(self.distance_to_object(pos + normal * 0.025).x * 40.0);
            ambient_s *= saturate(self.distance_to_object(pos + normal * 0.05).x * 20.0);
            ambient_s *= saturate(self.distance_to_object(pos + normal * 0.1).x * 10.0);
            ambient_s *= saturate(self.distance_to_object(pos + normal * 0.2).x * 5.0);
            ambient_s *= saturate(self.distance_to_object(pos + normal * 0.4).x * 2.5);
            //ambientS *= saturate_f32(DistanceToObject(pos + normal * 0.8).x*1.25);
            let mut ambient: f32 = ambient_s; // * saturate_f32(DistanceToObject(pos + normal * 1.6).x*1.25*0.5);
                                              //ambient *= saturate_f32(DistanceToObject(pos + normal * 3.2)*1.25*0.25);
                                              //ambient *= saturate_f32(DistanceToObject(pos + normal * 6.4)*1.25*0.125);
            ambient = ambient.powf(0.5).max(0.025); // tone down ambient with a pow and min clamp it.
            ambient = saturate(ambient);

            // calculate the reflection vector for highlights
            let ref_: Vec3 = ray_vec.reflect(normal);

            // Trace a ray toward the sun for sun shadows
            let mut sun_shadow: f32 = 1.0;
            let mut iter: f32 = 0.01;
            let nudge_pos: Vec3 = pos + normal * 0.002; // don't start tracing too close or inside the object
            let mut i = 0;
            while i < 40 {
                let shadow_pos: Vec3 = nudge_pos + self.sun_dir * iter;
                let temp_dist: f32 = self.distance_to_object(shadow_pos).x;
                sun_shadow *= saturate(temp_dist * 150.0); // Shadow hardness
                if temp_dist <= 0.0 {
                    break;
                }

                let mut walk: f32 = temp_dist;
                let mut dx: f32 = -shadow_pos.x.gl_fract();
                if self.sun_dir.x > 0.0 {
                    dx = (-shadow_pos.x).gl_fract();
                }
                let mut dz: f32 = -shadow_pos.z.gl_fract();
                if self.sun_dir.z > 0.0 {
                    dz = (-shadow_pos.z).gl_fract();
                }
                let mut nearest_voxel: f32 = (dx / self.sun_dir.x)
                    .gl_fract()
                    .min((dz / self.sun_dir.z).gl_fract())
                    + small_val;
                nearest_voxel = nearest_voxel.max(0.2); // hack that assumes streets and sidewalks are this wide.
                walk = walk.min(nearest_voxel);

                iter += walk.max(0.01);
                if iter > 4.5 {
                    break;
                }
                i += 1;
            }
            sun_shadow = saturate(sun_shadow);

            // make a few frequencies of noise to give it some texture
            let mut n: f32 = 0.0;
            n += noise(pos * 32.0);
            n += noise(pos * 64.0);
            n += noise(pos * 128.0);
            n += noise(pos * 256.0);
            n += noise(pos * 512.0);
            n = mix(0.7, 0.95, n);

            // ------ Calculate texture color  ------
            let block: Vec2 = pos.xz().floor();
            let mut tex_color: Vec3 = vec3(0.95, 1.0, 1.0);
            tex_color *= 0.8;
            let mut window_ref: f32 = 0.0;
            // texture map the sides of buildings
            if (normal.y < 0.1) && (dist_and_mat.y == 0.0) {
                let posdx: Vec3 = pos.ddx();
                let posdy: Vec3 = pos.ddy();
                let _pos_grad: Vec3 = posdx * hash21(uv) + posdy * hash21(uv * 7.6543);

                // Quincunx antialias the building texture and normal map.
                // I guess procedural textures are hard to mipmap.
                let mut col_total: Vec3;
                let mut col_temp: Vec3 = tex_color;
                let mut n_temp: Vec3 = Vec3::ZERO;
                calc_windows(block, pos, &mut col_temp, &mut window_ref, &mut n_temp);
                col_total = col_temp;

                col_temp = tex_color;
                calc_windows(
                    block,
                    pos + posdx * 0.666,
                    &mut col_temp,
                    &mut window_ref,
                    &mut n_temp,
                );
                col_total += col_temp;

                col_temp = tex_color;
                calc_windows(
                    block,
                    pos + posdx * 0.666 + posdy * 0.666,
                    &mut col_temp,
                    &mut window_ref,
                    &mut n_temp,
                );
                col_total += col_temp;

                col_temp = tex_color;
                calc_windows(
                    block,
                    pos + posdy * 0.666,
                    &mut col_temp,
                    &mut window_ref,
                    &mut n_temp,
                );
                col_total += col_temp;

                col_temp = tex_color;
                calc_windows(
                    block,
                    pos + posdx * 0.333 + posdy * 0.333,
                    &mut col_temp,
                    &mut window_ref,
                    &mut n_temp,
                );
                col_total += col_temp;

                tex_color = col_total * 0.2;
                window_ref *= 0.2;

                normal = (normal + n_temp * 0.2).normalize();
            } else {
                // Draw the road
                let xroad: f32 = ((pos.x + 0.5).gl_fract() - 0.5).abs();
                let zroad: f32 = ((pos.z + 0.5).gl_fract() - 0.5).abs();
                let road: f32 = saturate((xroad.min(zroad) - 0.143) * 480.0);
                tex_color *= 1.0 - normal.y * 0.95 * hash21(block * 9.87) * road; // change rooftop color
                tex_color *= mix(0.1, 1.0, road);

                // double yellow line in middle of road
                let mut yellow_line: f32 = saturate(1.0 - (xroad.min(zroad) - 0.002) * 480.0);
                yellow_line *= saturate((xroad.min(zroad) - 0.0005) * 480.0);
                yellow_line *= saturate((xroad * xroad + zroad * zroad - 0.05) * 880.0);
                tex_color = mix(tex_color, vec3(1.0, 0.8, 0.3), yellow_line);

                // white dashed lines on road
                let mut white_line: f32 = saturate(1.0 - (xroad.min(zroad) - 0.06) * 480.0);
                white_line *= saturate((xroad.min(zroad) - 0.056) * 480.0);
                white_line *= saturate((xroad * xroad + zroad * zroad - 0.05) * 880.0);
                white_line *= saturate(1.0 - ((zroad * 8.0).gl_fract() - 0.5) * 280.0); // dotted line
                white_line *= saturate(1.0 - ((xroad * 8.0).gl_fract() - 0.5) * 280.0);
                tex_color = mix(tex_color, Vec3::splat(0.5), white_line);

                white_line = saturate(1.0 - (xroad.min(zroad) - 0.11) * 480.0);
                white_line *= saturate((xroad.min(zroad) - 0.106) * 480.0);
                white_line *= saturate((xroad * xroad + zroad * zroad - 0.06) * 880.0);
                tex_color = mix(tex_color, Vec3::splat(0.5), white_line);

                // crosswalk
                let mut cross_walk: f32 = saturate(1.0 - ((xroad * 40.0).gl_fract() - 0.5) * 280.0);
                cross_walk *= saturate((zroad - 0.15) * 880.0);
                cross_walk *= saturate((-zroad + 0.21) * 880.0) * (1.0 - road);
                cross_walk *= n * n;
                tex_color = mix(tex_color, Vec3::splat(0.25), cross_walk);
                cross_walk = saturate(1.0 - ((zroad * 40.0).gl_fract() - 0.5) * 280.0);
                cross_walk *= saturate((xroad - 0.15) * 880.0);
                cross_walk *= saturate((-xroad + 0.21) * 880.0) * (1.0 - road);
                cross_walk *= n * n;
                tex_color = mix(tex_color, Vec3::splat(0.25), cross_walk);

                {
                    // sidewalk cracks
                    let mut sidewalk: f32 = 1.0;
                    let mut block_size: Vec2 = Vec2::splat(100.0);
                    if pos.y > 0.1 {
                        block_size = vec2(10.0, 50.0);
                    }
                    //sidewalk *= (pos.x*block_size).sin()abs().pow(0.025);
                    //sidewalk *= (pos.z*block_size).sin()abs().pow(0.025);
                    sidewalk *=
                        saturate(((pos.z * block_size.x).sin() * 800.0 / block_size.x).abs());
                    sidewalk *=
                        saturate(((pos.x * block_size.y).sin() * 800.0 / block_size.y).abs());
                    sidewalk = saturate(mix(0.7, 1.0, sidewalk));
                    sidewalk = saturate((1.0 - road) + sidewalk);
                    tex_color *= sidewalk;
                }
            }
            // Car tires are almost black to not call attention to their ugly.
            if dist_and_mat.y == 3.0 {
                tex_color = Vec3::splat(0.05);
            }

            // apply noise
            tex_color *= Vec3::ONE * n * 0.05;
            tex_color *= 0.7;
            tex_color = saturate_vec3(tex_color);

            let mut window_mask: f32 = 0.0;
            if dist_and_mat.y >= 100.0 {
                // car texture and windows
                tex_color = vec3(
                    hash11(dist_and_mat.y) * 1.0,
                    hash11(dist_and_mat.y * 8.765),
                    hash11(dist_and_mat.y * 17.731),
                ) * 0.1;
                tex_color = tex_color.abs().powf(0.2); // bias toward white
                tex_color = Vec3::splat(0.25).max(tex_color); // not too saturated color.
                tex_color.z = tex_color.y.min(tex_color.z); // no purple cars. just not realistic. :)
                tex_color *= hash11(dist_and_mat.y * 0.789) * 0.15;
                window_mask = saturate(((pos.y - 0.0175).abs() * 3800.0).max(0.0) - 10.0);
                let dir_norm: Vec2 = normal.xz().normalize().abs();
                let mut pillars: f32 = saturate(1.0 - dir_norm.x.max(dir_norm.y));
                pillars = (pillars - 0.15).max(0.0).powf(0.125);
                window_mask = window_mask.max(pillars);
                tex_color *= window_mask;
            }

            // ------ Calculate lighting color ------
            // Start with sun color, standard lighting equation, and shadow
            let mut light_color: Vec3 =
                Vec3::splat(100.0) * self.sun_col * saturate(self.sun_dir.dot(normal)) * sun_shadow;
            // weighted average the near ambient occlusion with the far for just the right look
            let ambient_avg: f32 = (ambient * 3.0 + ambient_s) * 0.25;
            // Add sky color with ambient acclusion
            light_color +=
                (self.sky_col * saturate(normal.y * 0.5 + 0.5)) * ambient_avg.powf(0.35) * 2.5;
            light_color *= 4.0;

            // finally, apply the light to the texture.
            final_color = tex_color * light_color;
            // Reflections for cars
            if dist_and_mat.y >= 100.0 {
                let mut yfade: f32 = 0.01_f32.max(1.0_f32.min(ref_.y * 100.0));
                // low-res way of making lines at the edges of car windows. Not sure I like it.
                yfade *= saturate(1.0 - (window_mask.ddx() * window_mask.ddy()).abs() * 250.995);
                final_color += self.get_env_map_skyline(ref_, self.sun_dir, pos.y - 1.5)
                    * 0.3
                    * yfade
                    * sun_shadow.max(0.4);
                final_color +=
                    saturate_vec3(self.inputs.channel0.sample_cube(ref_).xyz() - Vec3::splat(0.35))
                        * 0.15
                        * sun_shadow.max(0.2);
            }
            // reflections for building windows
            if window_ref != 0.0 {
                final_color *= mix(1.0, 0.6, window_ref);
                let yfade: f32 = 0.01_f32.max(1.0_f32.min(ref_.y * 100.0));
                final_color += self.get_env_map_skyline(ref_, self.sun_dir, pos.y - 0.5)
                    * 0.6
                    * yfade
                    * sun_shadow.max(0.6)
                    * window_ref; //*(windowMask*0.5+0.5);
                final_color +=
                    saturate_vec3(self.inputs.channel0.sample_cube(ref_).xyz() - Vec3::splat(0.35))
                        * 0.15
                        * sun_shadow.max(0.25)
                        * window_ref;
            }
            final_color *= 0.9;
            // fog that fades to reddish plus the sun color so that fog is brightest towards sun
            let mut rv2: Vec3 = ray_vec;
            rv2.y *= saturate(rv2.y.gl_sign());
            let mut fog_color: Vec3 = self.get_env_map(rv2, self.sun_dir);
            fog_color = Vec3::splat(9.0).min(fog_color);
            final_color = mix(fog_color, final_color, (-t * 0.02).exp());

            // visualize length of gradient of distance field to check distance field correctness
            //final_color = Vec3::splat(0.5) * (normalU.length() / smallVec.x);
            //final_color = Vec3::splat(marchCount)/255.0;
        } else {
            // Our ray trace hit nothing, so draw sky.
            final_color = self.get_env_map(ray_vec, self.sun_dir);
        }

        // vignette?
        final_color *= Vec3::ONE * saturate(1.0 - (uv / 2.5).length());
        final_color *= 1.3 * self.exposure;

        // output the final color without gamma correction - will do gamma later.
        final_color.clamp(Vec3::ZERO, Vec3::ONE) * saturate(self.fade + 0.2)
    }
}

impl<C0: SampleCube> State<C0> {
    // This function breaks the image down into blocks and scans
    // through them, rendering 1 block at a time. It's for non-
    // realtime things that take a long time to render.

    // This is the frame rate to render at. Too fast and you will
    // miss some blocks.
    const BLOCK_RATE: f32 = 20.0;

    fn block_render(&self, frag_coord: Vec2) {
        // blockSize is how much it will try to render in 1 frame.
        // adjust this smaller for more complex scenes, bigger for
        // faster render times.
        let block_size: f32 = 64.0;
        // Make the block repeatedly scan across the image based on time.
        let frame: f32 = (self.inputs.time * Self::BLOCK_RATE).floor();
        let block_res: Vec2 = (self.inputs.resolution.xy() / block_size).floor() + Vec2::ONE;
        // ugly bug with mod.
        //float blockX = mod(frame, blockRes.x);
        let block_x: f32 = (frame / block_res.x).gl_fract() * block_res.x;
        //float blockY = mod(floor(frame / blockRes.x), blockRes.y);
        let block_y: f32 = ((frame / block_res.x).floor() / block_res.y).gl_fract() * block_res.y;
        // Don't draw anything outside the current block.
        if (frag_coord.x - block_x * block_size >= block_size)
            || (frag_coord.x - (block_x - 1.0) * block_size < block_size)
            || (frag_coord.y - block_y * block_size >= block_size)
            || (frag_coord.y - (block_y - 1.0) * block_size < block_size)
        {
            discard();
        }
    }

    pub fn main_image(&mut self, frag_color: &mut Vec4, frag_coord: Vec2) {
        if NON_REALTIME_HQ_RENDER {
            // Optionally render a non-realtime scene with high quality
            self.block_render(frag_coord);
        }

        // Do a multi-pass render
        let mut final_color: Vec3 = Vec3::ZERO;
        if NON_REALTIME_HQ_RENDER {
            let mut i = 0.0;
            while i < ANTIALIASING_SAMPLES {
                let motion_blur_length_in_seconds: f32 = 1.0 / 60.0;
                // Set this to the time in seconds of the frame to render.
                self.local_time = FRAME_TO_RENDER_HQ;
                // This line will motion-blur the renders
                self.local_time += hash11(v21(frag_coord + Vec2::splat(self.seed)))
                    * motion_blur_length_in_seconds;
                // Jitter the pixel position so we get antialiasing when we do multiple passes.
                let mut jittered: Vec2 = frag_coord
                    + vec2(
                        hash21(frag_coord + Vec2::splat(self.seed)),
                        hash21(frag_coord * 7.234567 + Vec2::splat(self.seed)),
                    );
                // don't antialias if only 1 sample.
                if ANTIALIASING_SAMPLES == 1.0 {
                    jittered = frag_coord
                };
                // Accumulate one pass of raytracing into our pixel value
                final_color += self.ray_trace(jittered);
                // Change the random seed for each pass.
                self.seed *= 1.01234567;
                i += 1.0;
            }
            // Average all accumulated pixel intensities
            final_color /= ANTIALIASING_SAMPLES;
        } else {
            // Regular real-time rendering
            self.local_time = self.inputs.time;
            final_color = self.ray_trace(frag_coord);
        }

        *frag_color = final_color.clamp(Vec3::ZERO, Vec3::ONE).sqrt().extend(1.0);
    }
}
