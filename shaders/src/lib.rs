#![cfg_attr(target_arch = "spirv", no_std)]
#![feature(lang_items)]
#![feature(register_attr)]
#![register_attr(spirv)]

use shared::*;
use spirv_std::glam::{vec2, vec3, vec4, Vec2, Vec3, Vec4};
use spirv_std::storage_class::{Input, Output, PushConstant};

pub mod a_lot_of_spheres;
pub mod a_question_of_time;
pub mod apollonian;
pub mod atmosphere_system_test;
pub mod bubble_buckey_balls;
pub mod clouds;
pub mod galaxy_of_universes;
pub mod heart;
pub mod mandelbrot_smooth;
pub mod miracle_snowflakes;
pub mod morphing;
pub mod phantom_star;
pub mod playing_marble;
pub mod protean_clouds;
pub mod raymarching_primitives;
pub mod seascape;
pub mod soft_shadow_variation;
pub mod tileable_water_caustic;
pub mod two_tweets;

pub trait Channel: Copy {
    fn sample_cube(self, p: Vec3) -> Vec4;
}

#[derive(Copy, Clone)]
struct ConstantColor {
    color: Vec4,
}

impl Channel for ConstantColor {
    fn sample_cube(self, _: Vec3) -> Vec4 {
        self.color
    }
}

#[derive(Copy, Clone)]
struct RgbCube {
    alpha: f32,
    intensity: f32,
}

impl Channel for RgbCube {
    fn sample_cube(self, p: Vec3) -> Vec4 {
        (p.abs() * self.intensity).extend(self.alpha)
    }
}

pub fn fs(constants: &ShaderConstants, mut frag_coord: Vec2) -> Vec4 {
    const COLS: usize = 4;
    const ROWS: usize = 5;

    let resolution = vec3(
        constants.width as f32 / COLS as f32,
        constants.height as f32 / ROWS as f32,
        0.0,
    );
    let time = constants.time;
    let mut mouse = vec4(
        constants.drag_end_x / COLS as f32,
        constants.drag_end_y / ROWS as f32,
        constants.drag_start_x / COLS as f32,
        constants.drag_start_y / ROWS as f32,
    );
    if mouse != Vec4::zero() {
        mouse.y = resolution.y - mouse.y;
        mouse.w = resolution.y - mouse.w;
    }
    if !constants.mouse_left_pressed {
        mouse.z *= -1.0;
    }
    if !constants.mouse_left_clicked {
        mouse.w *= -1.0;
    }

    let col = (frag_coord.x / resolution.x) as usize;
    let row = (frag_coord.y / resolution.y) as usize;
    let i = row * COLS + col;

    frag_coord.x %= resolution.x;
    frag_coord.y = resolution.y - frag_coord.y % resolution.y;

    let mut color = Vec4::zero();
    match i {
        0 => two_tweets::Inputs { resolution, time }.main_image(&mut color, frag_coord),
        1 => heart::Inputs { resolution, time }.main_image(&mut color, frag_coord),
        2 => clouds::Inputs { resolution, time }.main_image(&mut color, frag_coord),
        3 => mandelbrot_smooth::Inputs { resolution, time }.main_image(&mut color, frag_coord),
        4 => protean_clouds::State::new(protean_clouds::Inputs {
            resolution,
            time,
            mouse,
        })
        .main_image(&mut color, frag_coord),
        5 => tileable_water_caustic::Inputs { resolution, time }.main_image(&mut color, frag_coord),
        6 => apollonian::State::new(apollonian::Inputs {
            resolution,
            time,
            mouse,
        })
        .main_image(&mut color, frag_coord),
        7 => phantom_star::Inputs { resolution, time }.main_image(&mut color, frag_coord),
        8 => seascape::Inputs {
            resolution,
            time,
            mouse,
        }
        .main_image(&mut color, frag_coord),
        9 => playing_marble::Inputs {
            resolution,
            time,
            mouse,
            channel0: RgbCube {
                alpha: 1.0,
                intensity: 1.0,
            },
        }
        .main_image(&mut color, frag_coord),
        10 => a_lot_of_spheres::Inputs { resolution, time }.main_image(&mut color, frag_coord),
        11 => a_question_of_time::Inputs {
            resolution,
            time,
            mouse,
        }
        .main_image(&mut color, frag_coord),
        12 => galaxy_of_universes::Inputs { resolution, time }.main_image(&mut color, frag_coord),
        13 => atmosphere_system_test::State::new(atmosphere_system_test::Inputs {
            resolution,
            time,
            mouse,
        })
        .main_image(&mut color, frag_coord),
        14 => soft_shadow_variation::Inputs { resolution, time }.main_image(&mut color, frag_coord),
        15 => miracle_snowflakes::State::new(miracle_snowflakes::Inputs {
            resolution,
            time,
            mouse,
        })
        .main_image(&mut color, frag_coord),
        16 => morphing::State::new(morphing::Inputs {
            resolution,
            time,
            mouse,
        })
        .main_image(&mut color, frag_coord),
        17 => bubble_buckey_balls::State::new(bubble_buckey_balls::Inputs {
            resolution,
            time,
            mouse,
            channel0: RgbCube {
                alpha: 1.0,
                intensity: 0.5,
            },
            channel1: ConstantColor { color: Vec4::one() },
        })
        .main_image(&mut color, frag_coord),
        18 => raymarching_primitives::Inputs {
            resolution,
            frame: (time * 60.0) as i32,
            time,
            mouse,
        }
        .main_image(&mut color, frag_coord),
        _ => {}
    }
    pow(color.truncate(), 2.2).extend(color.w)
}

#[allow(unused_attributes)]
#[spirv(fragment)]
pub fn main_fs(
    #[spirv(frag_coord)] in_frag_coord: Input<Vec4>,
    constants: PushConstant<ShaderConstants>,
    mut output: Output<Vec4>,
) {
    let constants = constants.load();

    let frag_coord = vec2(in_frag_coord.load().x, in_frag_coord.load().y);
    let color = fs(&constants, frag_coord);
    output.store(color);
}

#[allow(unused_attributes)]
#[spirv(vertex)]
pub fn main_vs(
    #[spirv(vertex_index)] vert_idx: Input<i32>,
    #[spirv(position)] mut builtin_pos: Output<Vec4>,
) {
    let vert_idx = vert_idx.load();

    // Create a "full screen triangle" by mapping the vertex index.
    // ported from https://www.saschawillems.de/blog/2016/08/13/vulkan-tutorial-on-rendering-a-fullscreen-quad-without-buffers/
    let uv = vec2(((vert_idx << 1) & 2) as f32, (vert_idx & 2) as f32);
    let pos = 2.0 * uv - Vec2::one();

    builtin_pos.store(pos.extend(0.0).extend(1.0));
}
