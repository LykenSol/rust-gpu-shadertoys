//! Ported to Rust from <https://www.shadertoy.com/view/ldjGzt>
//!
//! Original comment:
//! ```glsl
//! // FlappyBird by Ben Raziel. Feb 2014
//!
//! // Based on the "Super Mario Bros" shader by HLorenzi
//! // https://www.shadertoy.com/view/Msj3zD
//! ```

use spirv_std::glam::{const_vec4, vec2, vec4, Vec2, Vec3, Vec4};

// Note: This cfg is incorrect on its surface, it really should be "are we compiling with std", but
// we tie #[no_std] above to the same condition, so it's fine.
#[cfg(target_arch = "spirv")]
use {shared::*, spirv_std::num_traits::Float};

pub struct Inputs {
    pub resolution: Vec3,
    pub time: f32,
}

pub struct State {
    inputs: Inputs,

    frag_color: Vec4,
}

impl State {
    pub fn new(inputs: Inputs) -> State {
        State {
            inputs,

            frag_color: Vec4::zero(),
        }
    }
}

// Helper functions for drawing sprites
fn rgb(r: i32, g: i32, b: i32) -> Vec4 {
    vec4(r as f32 / 255.0, g as f32 / 255.0, b as f32 / 255.0, 1.0)
}
fn sprrow(
    x: i32,
    a: f32,
    b: f32,
    c: f32,
    d: f32,
    e: f32,
    f: f32,
    g: f32,
    h: f32,
    i: f32,
    j: f32,
    k: f32,
    l: f32,
    m: f32,
    n: f32,
    o: f32,
    p: f32,
) -> f32 {
    if x <= 7 {
        sprrow_h(a, b, c, d, e, f, g, h)
    } else {
        sprrow_h(i, j, k, l, m, n, o, p)
    }
}
fn sprrow_h(a: f32, b: f32, c: f32, d: f32, e: f32, f: f32, g: f32, h: f32) -> f32 {
    a + 4.0 * (b + 4.0 * (c + 4.0 * (d + 4.0 * (e + 4.0 * (f + 4.0 * (g + 4.0 * (h)))))))
}
fn _secrow(x: i32, a: f32, b: f32, c: f32, d: f32, e: f32, f: f32, g: f32, h: f32) -> f32 {
    if x <= 3 {
        _secrow_h(a, b, c, d)
    } else {
        _secrow_h(e, f, g, h)
    }
}
fn _secrow_h(a: f32, b: f32, c: f32, d: f32) -> f32 {
    a + 8.0 * (b + 8.0 * (c + 8.0 * (d)))
}
fn select(x: i32, i: f32) -> f32 {
    (i / 4.0_f32.powf(x as f32)).floor().rem_euclid(4.0)
}
fn _selectsec(x: i32, i: f32) -> f32 {
    (i / 8.0_f32.powf(x as f32)).floor().rem_euclid(8.0)
}

// drawing consts
const PIPE_WIDTH: f32 = 26.0; // px
const PIPE_BOTTOM: f32 = 39.0; // px
const PIPE_HOLE_HEIGHT: f32 = 12.0; // px

// const PIPE_OUTLINE_COLOR: Vec4 = RGB(84, 56, 71);
const PIPE_OUTLINE_COLOR: Vec4 =
    const_vec4!([84 as f32 / 255.0, 56 as f32 / 255.0, 71 as f32 / 255.0, 1.0]);

// gameplay consts
const HORZ_PIPE_DISTANCE: f32 = 100.0; // px;
const VERT_PIPE_DISTANCE: f32 = 55.0; // px;
const PIPE_MIN: f32 = 20.0;
const PIPE_MAX: f32 = 70.0;
const PIPE_PER_CYCLE: f32 = 8.0;

impl State {
    fn draw_horz_rect(&mut self, y_coord: f32, min_y: f32, max_y: f32, color: Vec4) {
        if (y_coord >= min_y) && (y_coord < max_y) {
            self.frag_color = color;
        }
    }

    fn draw_low_bush(&mut self, x: i32, y: i32) {
        if y < 0 || y > 3 || x < 0 || x > 15 {
            return;
        }

        let mut col: f32 = 0.0; // 0 = transparent

        if y == 3 {
            col = sprrow(
                x, 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0.,
            );
        }
        if y == 2 {
            col = sprrow(
                x, 0., 0., 0., 0., 1., 1., 2., 2., 2., 2., 1., 1., 0., 0., 0., 0.,
            );
        }
        if y == 1 {
            col = sprrow(
                x, 0., 0., 0., 1., 1., 2., 2., 2., 2., 2., 2., 1., 1., 0., 0., 0.,
            );
        }
        if y == 0 {
            col = sprrow(
                x, 0., 0., 1., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 1., 0., 0.,
            );
        }

        // i made this tho a i32 cast bc it seems as this thing(SELECT(i32, f32)) uses a i 32
        col = select((x as f32).rem_euclid(8.0) as i32, col);
        if col == 1.0 {
            self.frag_color = rgb(87, 201, 111);
        } else if col == 2.0 {
            self.frag_color = rgb(100, 224, 117);
        }
    }

    fn draw_high_bush(&mut self, x: i32, y: i32) {
        if y < 0 || y > 6 || x < 0 || x > 15 {
            return;
        }

        let mut col: f32 = 0.0; // 0 = transparent

        if y == 6 {
            col = sprrow(
                x, 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0.,
            );
        }
        if y == 5 {
            col = sprrow(
                x, 0., 0., 0., 0., 1., 1., 2., 2., 2., 2., 1., 1., 0., 0., 0., 0.,
            );
        }
        if y == 4 {
            col = sprrow(
                x, 0., 0., 1., 1., 2., 2., 2., 2., 2., 2., 2., 2., 1., 1., 0., 0.,
            );
        }
        if y == 3 {
            col = sprrow(
                x, 0., 1., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 1., 0.,
            );
        }
        if y == 2 {
            col = sprrow(
                x, 0., 1., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 1., 0.,
            );
        }
        if y == 1 {
            col = sprrow(
                x, 1., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 1.,
            );
        }
        if y == 0 {
            col = sprrow(
                x, 1., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 1.,
            );
        }

        col = select((x as f32).rem_euclid(8.0) as i32, col);
        if col == 1.0 {
            self.frag_color = rgb(87, 201, 111);
        } else if col == 2.0 {
            self.frag_color = rgb(100, 224, 117);
        }
    }

    fn draw_cloud(&mut self, x: i32, y: i32) {
        if y < 0 || y > 6 || x < 0 || x > 15 {
            return;
        }

        let mut col: f32 = 0.0; // 0 = transparent

        if y == 6 {
            col = sprrow(
                x, 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0.,
            );
        }
        if y == 5 {
            col = sprrow(
                x, 0., 0., 0., 0., 1., 1., 2., 2., 2., 2., 1., 1., 0., 0., 0., 0.,
            );
        }
        if y == 4 {
            col = sprrow(
                x, 0., 0., 1., 1., 2., 2., 2., 2., 2., 2., 2., 2., 1., 1., 0., 0.,
            );
        }
        if y == 3 {
            col = sprrow(
                x, 0., 1., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 1., 0.,
            );
        }
        if y == 2 {
            col = sprrow(
                x, 0., 1., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 1., 0.,
            );
        }
        if y == 1 {
            col = sprrow(
                x, 1., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 1.,
            );
        }
        if y == 0 {
            col = sprrow(
                x, 1., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 2., 1.,
            );
        }

        col = select((x as f32).rem_euclid(8.0) as i32, col);
        if col == 1.0 {
            self.frag_color = rgb(218, 246, 216);
        } else if col == 2.0 {
            self.frag_color = rgb(233, 251, 218);
        }
    }

    fn draw_bird_f0(&mut self, x: i32, y: i32) {
        if y < 0 || y > 11 || x < 0 || x > 15 {
            return;
        }

        // pass 0 - draw black, white and yellow
        let mut col: f32 = 0.0; // 0 = transparent
        if y == 11 {
            col = sprrow(
                x, 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0.,
            );
        }
        if y == 10 {
            col = sprrow(
                x, 0., 0., 0., 0., 1., 1., 3., 3., 3., 1., 2., 2., 1., 0., 0., 0.,
            );
        }
        if y == 9 {
            col = sprrow(
                x, 0., 0., 0., 1., 3., 3., 3., 3., 1., 2., 2., 2., 2., 1., 0., 0.,
            );
        }
        if y == 8 {
            col = sprrow(
                x, 0., 0., 1., 3., 3., 3., 3., 3., 1., 2., 2., 2., 1., 2., 1., 0.,
            );
        }
        if y == 7 {
            col = sprrow(
                x, 0., 1., 3., 3., 3., 3., 3., 3., 1., 2., 2., 2., 1., 2., 1., 0.,
            );
        }
        if y == 6 {
            col = sprrow(
                x, 0., 1., 3., 3., 3., 3., 3., 3., 3., 1., 2., 2., 2., 2., 1., 0.,
            );
        }
        if y == 5 {
            col = sprrow(
                x, 0., 1., 1., 1., 1., 1., 3., 3., 3., 3., 1., 1., 1., 1., 1., 1.,
            );
        }
        if y == 4 {
            col = sprrow(
                x, 1., 2., 2., 2., 2., 2., 1., 3., 3., 1., 2., 2., 2., 2., 2., 1.,
            );
        }
        if y == 3 {
            col = sprrow(
                x, 1., 2., 2., 2., 2., 1., 3., 3., 1., 2., 1., 1., 1., 1., 1., 1.,
            );
        }
        if y == 2 {
            col = sprrow(
                x, 1., 2., 2., 2., 1., 3., 3., 3., 3., 1., 2., 2., 2., 2., 1., 0.,
            );
        }
        if y == 1 {
            col = sprrow(
                x, 0., 1., 1., 1., 1., 3., 3., 3., 3., 3., 1., 1., 1., 1., 1., 0.,
            );
        }
        if y == 0 {
            col = sprrow(
                x, 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0.,
            );
        }

        col = select((x as f32).rem_euclid(8.0) as i32, col);
        if col == 1.0 {
            self.frag_color = rgb(82, 56, 70); // outline color (black)
        } else if col == 2.0 {
            self.frag_color = rgb(250, 250, 250); // eye color (white)
        } else if col == 3.0 {
            self.frag_color = rgb(247, 182, 67); // normal yellow color
        }

        // pass 1 - draw red, light yellow and dark yellow
        col = 0.0; // 0 = transparent
        if y == 11 {
            col = sprrow(
                x, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            );
        }
        if y == 10 {
            col = sprrow(
                x, 0., 0., 0., 0., 0., 0., 3., 3., 3., 0., 0., 0., 0., 0., 0., 0.,
            );
        }
        if y == 9 {
            col = sprrow(
                x, 0., 0., 0., 0., 3., 3., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            );
        }
        if y == 8 {
            col = sprrow(
                x, 0., 0., 0., 3., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            );
        }
        if y == 7 {
            col = sprrow(
                x, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            );
        }
        if y == 6 {
            col = sprrow(
                x, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            );
        }
        if y == 5 {
            col = sprrow(
                x, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            );
        }
        if y == 4 {
            col = sprrow(
                x, 0., 3., 0., 0., 0., 3., 0., 0., 0., 0., 1., 1., 1., 1., 1., 0.,
            );
        }
        if y == 3 {
            col = sprrow(
                x, 0., 0., 0., 0., 0., 0., 2., 2., 0., 1., 0., 0., 0., 0., 0., 0.,
            );
        }
        if y == 2 {
            col = sprrow(
                x, 0., 0., 0., 3., 0., 2., 2., 2., 2., 0., 1., 1., 1., 1., 0., 0.,
            );
        }
        if y == 1 {
            col = sprrow(
                x, 0., 0., 0., 0., 0., 2., 2., 2., 2., 2., 0., 0., 0., 0., 0., 0.,
            );
        }
        if y == 0 {
            col = sprrow(
                x, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            );
        }

        col = select((x as f32).rem_euclid(8.0) as i32, col);
        if col == 1.0 {
            self.frag_color = rgb(249, 58, 28); // mouth color (red)
        } else if col == 2.0 {
            self.frag_color = rgb(222, 128, 55); // brown
        } else if col == 3.0 {
            self.frag_color = rgb(249, 214, 145); // light yellow
        }
    }

    fn draw_bird_f1(&mut self, x: i32, y: i32) {
        if y < 0 || y > 11 || x < 0 || x > 15 {
            return;
        }

        // pass 0 - draw black, white and yellow
        let mut col: f32 = 0.0; // 0 = transparent
        if y == 11 {
            col = sprrow(
                x, 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0.,
            );
        }
        if y == 10 {
            col = sprrow(
                x, 0., 0., 0., 0., 1., 1., 3., 3., 3., 1., 2., 2., 1., 0., 0., 0.,
            );
        }
        if y == 9 {
            col = sprrow(
                x, 0., 0., 0., 1., 3., 3., 3., 3., 1., 2., 2., 2., 2., 1., 0., 0.,
            );
        }
        if y == 8 {
            col = sprrow(
                x, 0., 0., 1., 3., 3., 3., 3., 3., 1., 2., 2., 2., 1., 2., 1., 0.,
            );
        }
        if y == 7 {
            col = sprrow(
                x, 0., 1., 3., 3., 3., 3., 3., 3., 1., 2., 2., 2., 1., 2., 1., 0.,
            );
        }
        if y == 6 {
            col = sprrow(
                x, 0., 1., 1., 1., 1., 1., 3., 3., 3., 1., 2., 2., 2., 2., 1., 0.,
            );
        }
        if y == 5 {
            col = sprrow(
                x, 1., 2., 2., 2., 2., 2., 1., 3., 3., 3., 1., 1., 1., 1., 1., 1.,
            );
        }
        if y == 4 {
            col = sprrow(
                x, 1., 2., 2., 2., 2., 2., 1., 3., 3., 1., 2., 2., 2., 2., 2., 1.,
            );
        }
        if y == 3 {
            col = sprrow(
                x, 0., 1., 1., 1., 1., 1., 3., 3., 1., 2., 1., 1., 1., 1., 1., 1.,
            );
        }
        if y == 2 {
            col = sprrow(
                x, 0., 0., 1., 3., 3., 3., 3., 3., 3., 1., 2., 2., 2., 2., 1., 0.,
            );
        }
        if y == 1 {
            col = sprrow(
                x, 0., 0., 0., 1., 1., 3., 3., 3., 3., 3., 1., 1., 1., 1., 1., 0.,
            );
        }
        if y == 0 {
            col = sprrow(
                x, 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0.,
            );
        }

        col = select((x as f32).rem_euclid(8.0) as i32, col);
        if col == 1.0 {
            self.frag_color = rgb(82, 56, 70); // outline color (black)
        } else if col == 2.0 {
            self.frag_color = rgb(250, 250, 250); // eye color (white)
        } else if col == 3.0 {
            self.frag_color = rgb(247, 182, 67); // normal yellow color
        }

        // pass 1 - draw red, light yellow and dark yellow
        col = 0.0; // 0 = transparent
        if y == 11 {
            col = sprrow(
                x, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            );
        }
        if y == 10 {
            col = sprrow(
                x, 0., 0., 0., 0., 0., 0., 3., 3., 3., 0., 0., 0., 0., 0., 0., 0.,
            );
        }
        if y == 9 {
            col = sprrow(
                x, 0., 0., 0., 0., 3., 3., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            );
        }
        if y == 8 {
            col = sprrow(
                x, 0., 0., 0., 3., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            );
        }
        if y == 7 {
            col = sprrow(
                x, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            );
        }
        if y == 6 {
            col = sprrow(
                x, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            );
        }
        if y == 5 {
            col = sprrow(
                x, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            );
        }
        if y == 4 {
            col = sprrow(
                x, 0., 3., 0., 0., 0., 3., 0., 0., 0., 0., 1., 1., 1., 1., 1., 0.,
            );
        }
        if y == 3 {
            col = sprrow(
                x, 0., 0., 0., 0., 0., 0., 2., 2., 0., 1., 0., 0., 0., 0., 0., 0.,
            );
        }
        if y == 2 {
            col = sprrow(
                x, 0., 0., 0., 2., 2., 2., 2., 2., 2., 0., 1., 1., 1., 1., 0., 0.,
            );
        }
        if y == 1 {
            col = sprrow(
                x, 0., 0., 0., 0., 0., 2., 2., 2., 2., 2., 0., 0., 0., 0., 0., 0.,
            );
        }
        if y == 0 {
            col = sprrow(
                x, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            );
        }

        col = select((x as f32).rem_euclid(8.0) as i32, col);
        if col == 1.0 {
            self.frag_color = rgb(249, 58, 28); // mouth color (red)
        } else if col == 2.0 {
            self.frag_color = rgb(222, 128, 55); // brown
        } else if col == 3.0 {
            self.frag_color = rgb(249, 214, 145); // light yellow
        }
    }

    fn draw_bird_f2(&mut self, x: i32, y: i32) {
        if y < 0 || y > 11 || x < 0 || x > 15 {
            return;
        }

        // pass 0 - draw black, white and yellow
        let mut col: f32 = 0.0; // 0 = transparent
        if y == 11 {
            col = sprrow(
                x, 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0.,
            );
        }
        if y == 10 {
            col = sprrow(
                x, 0., 0., 0., 0., 1., 1., 3., 3., 3., 1., 2., 2., 1., 0., 0., 0.,
            );
        }
        if y == 9 {
            col = sprrow(
                x, 0., 0., 0., 1., 3., 3., 3., 3., 1., 2., 2., 2., 2., 1., 0., 0.,
            );
        }
        if y == 8 {
            col = sprrow(
                x, 0., 1., 1., 1., 3., 3., 3., 3., 1., 2., 2., 2., 1., 2., 1., 0.,
            );
        }
        if y == 7 {
            col = sprrow(
                x, 1., 2., 2., 2., 1., 3., 3., 3., 1., 2., 2., 2., 1., 2., 1., 0.,
            );
        }
        if y == 6 {
            col = sprrow(
                x, 1., 2., 2., 2., 2., 1., 3., 3., 3., 1., 2., 2., 2., 2., 1., 0.,
            );
        }
        if y == 5 {
            col = sprrow(
                x, 1., 2., 2., 2., 2., 1., 3., 3., 3., 3., 1., 1., 1., 1., 1., 1.,
            );
        }
        if y == 4 {
            col = sprrow(
                x, 0., 1., 2., 2., 2., 1., 3., 3., 3., 1., 2., 2., 2., 2., 2., 1.,
            );
        }
        if y == 3 {
            col = sprrow(
                x, 0., 1., 1., 1., 1., 3., 3., 3., 1., 2., 1., 1., 1., 1., 1., 1.,
            );
        }
        if y == 2 {
            col = sprrow(
                x, 0., 0., 1., 3., 3., 3., 3., 3., 3., 1., 2., 2., 2., 2., 1., 0.,
            );
        }
        if y == 1 {
            col = sprrow(
                x, 0., 0., 0., 1., 1., 3., 3., 3., 3., 3., 1., 1., 1., 1., 1., 0.,
            );
        }
        if y == 0 {
            col = sprrow(
                x, 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0.,
            );
        }

        col = select((x as f32).rem_euclid(8.0) as i32, col);
        if col == 1.0 {
            self.frag_color = rgb(82, 56, 70); // outline color (black)
        } else if col == 2.0 {
            self.frag_color = rgb(250, 250, 250); // eye color (white)
        } else if col == 3.0 {
            self.frag_color = rgb(247, 182, 67); // normal yellow color
        }

        // pass 1 - draw red, light yellow and dark yellow
        col = 0.0; // 0 = transparent
        if y == 11 {
            col = sprrow(
                x, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            );
        }
        if y == 10 {
            col = sprrow(
                x, 0., 0., 0., 0., 0., 0., 3., 3., 3., 0., 0., 0., 0., 0., 0., 0.,
            );
        }
        if y == 9 {
            col = sprrow(
                x, 0., 0., 0., 0., 3., 3., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            );
        }
        if y == 8 {
            col = sprrow(
                x, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            );
        }
        if y == 7 {
            col = sprrow(
                x, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            );
        }
        if y == 6 {
            col = sprrow(
                x, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            );
        }
        if y == 5 {
            col = sprrow(
                x, 0., 3., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            );
        }
        if y == 4 {
            col = sprrow(
                x, 0., 0., 3., 3., 3., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 0.,
            );
        }
        if y == 3 {
            col = sprrow(
                x, 0., 0., 0., 0., 0., 2., 2., 2., 0., 1., 0., 0., 0., 0., 0., 0.,
            );
        }
        if y == 2 {
            col = sprrow(
                x, 0., 0., 0., 2., 2., 2., 2., 2., 2., 0., 1., 1., 1., 1., 0., 0.,
            );
        }
        if y == 1 {
            col = sprrow(
                x, 0., 0., 0., 0., 0., 2., 2., 2., 2., 2., 0., 0., 0., 0., 0., 0.,
            );
        }
        if y == 0 {
            col = sprrow(
                x, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            );
        }

        col = select((x as f32).rem_euclid(8.0) as i32, col);
        if col == 1.0 {
            self.frag_color = rgb(249, 58, 28); // mouth color (red)
        } else if col == 2.0 {
            self.frag_color = rgb(222, 128, 55); // brown
        } else if col == 3.0 {
            self.frag_color = rgb(249, 214, 145); // light yellow
        }
    }

    fn get_level_pixel(&self, frag_coord: Vec2) -> Vec2 {
        // Get the current game pixel
        // (Each game pixel is two screen pixels)
        //  (or four, if the screen is larger)
        let mut x: f32 = frag_coord.x / 2.0;
        let mut y: f32 = frag_coord.y / 2.0;

        if self.inputs.resolution.y >= 640.0 {
            x /= 2.0;
            y /= 2.0;
        }

        if self.inputs.resolution.y < 200.0 {
            x *= 2.0;
            y *= 2.0;
        }

        vec2(x, y)
    }

    fn get_level_bounds(&self) -> Vec2 {
        // same logic as getLevelPixel, but returns the boundaries of the screen

        let mut x: f32 = self.inputs.resolution.x / 2.0;
        let mut y: f32 = self.inputs.resolution.y / 2.0;

        if self.inputs.resolution.y >= 640.0 {
            x /= 2.0;
            y /= 2.0;
        }

        if self.inputs.resolution.y < 200.0 {
            x *= 2.0;
            y *= 2.0;
        }

        vec2(x, y)
    }

    fn draw_ground(&mut self, co: Vec2) {
        self.draw_horz_rect(co.y, 0.0, 31.0, rgb(221, 216, 148));
        self.draw_horz_rect(co.y, 31.0, 32.0, rgb(208, 167, 84)); // shadow below the green sprites
    }

    fn draw_green_stripes(&mut self, co: Vec2) {
        let f: i32 = (self.inputs.time * 60.0).rem_euclid(6.0) as i32;

        self.draw_horz_rect(co.y, 32.0, 33.0, rgb(86, 126, 41)); // shadow blow

        let min_y: f32 = 33.0;
        let height: f32 = 6.0;

        let dark_green: Vec4 = rgb(117, 189, 58);
        let light_green: Vec4 = rgb(158, 228, 97);

        // draw diagonal stripes, and animate them
        if (co.y >= min_y) && (co.y < min_y + height) {
            let y_pos: f32 = co.y - min_y - f as f32;
            let x_pos: f32 = (co.x - y_pos).rem_euclid(height);

            if x_pos >= height / 2.0 {
                self.frag_color = dark_green;
            } else {
                self.frag_color = light_green;
            }
        }

        self.draw_horz_rect(co.y, 37.0, 38.0, rgb(228, 250, 145)); // shadow highlight above
        self.draw_horz_rect(co.y, 38.0, 39.0, rgb(84, 56, 71)); // black separator
    }

    fn draw_tile(&mut self, type_: i32, tile_corner: Vec2, co: Vec2) {
        if (co.x < tile_corner.x)
            || (co.x > (tile_corner.x + 16.0))
            || (co.y < tile_corner.y)
            || (co.y > (tile_corner.y + 16.0))
        {
            return;
        }

        let mod_x: i32 = (co.x - tile_corner.x).rem_euclid(16.0) as i32;
        let mod_y: i32 = (co.y - tile_corner.y).rem_euclid(16.0) as i32;

        if type_ == 0 {
            self.draw_low_bush(mod_x, mod_y);
        } else if type_ == 1 {
            self.draw_high_bush(mod_x, mod_y);
        } else if type_ == 2 {
            self.draw_cloud(mod_x, mod_y);
        } else if type_ == 3 {
            self.draw_bird_f0(mod_x, mod_y);
        } else if type_ == 4 {
            self.draw_bird_f1(mod_x, mod_y);
        } else if type_ == 5 {
            self.draw_bird_f2(mod_x, mod_y);
        }
    }

    fn draw_vert_line(&mut self, co: Vec2, x_pos: f32, y_start: f32, y_end: f32, color: Vec4) {
        if (co.x >= x_pos) && (co.x < (x_pos + 1.0)) && (co.y >= y_start) && (co.y < y_end) {
            self.frag_color = color;
        }
    }

    fn draw_horz_line(&mut self, co: Vec2, y_pos: f32, x_start: f32, x_end: f32, color: Vec4) {
        if (co.y >= y_pos) && (co.y < (y_pos + 1.0)) && (co.x >= x_start) && (co.x < x_end) {
            self.frag_color = color;
        }
    }

    fn draw_horz_gradient_rect(
        &mut self,
        co: Vec2,
        bottom_left: Vec2,
        top_right: Vec2,
        left_color: Vec4,
        right_color: Vec4,
    ) {
        if (co.x < bottom_left.x)
            || (co.y < bottom_left.y)
            || (co.x > top_right.x)
            || (co.y > top_right.y)
        {
            return;
        }

        let distance_ratio: f32 = (co.x - bottom_left.x) / (top_right.x - bottom_left.x);

        self.frag_color = (1.0 - distance_ratio) * left_color + distance_ratio * right_color;
    }

    fn draw_bottom_pipe(&mut self, co: Vec2, x_pos: f32, height: f32) {
        if (co.x < x_pos)
            || (co.x > (x_pos + PIPE_WIDTH))
            || (co.y < PIPE_BOTTOM)
            || (co.y > (PIPE_BOTTOM + height))
        {
            return;
        }

        // draw the bottom part of the pipe
        // outlines
        let bottom_part_end: f32 = PIPE_BOTTOM - PIPE_HOLE_HEIGHT + height;
        self.draw_vert_line(
            co,
            x_pos + 1.0,
            PIPE_BOTTOM,
            bottom_part_end,
            PIPE_OUTLINE_COLOR,
        );
        self.draw_vert_line(
            co,
            x_pos + PIPE_WIDTH - 2.0,
            PIPE_WIDTH,
            bottom_part_end,
            PIPE_OUTLINE_COLOR,
        );

        // gradient fills
        self.draw_horz_gradient_rect(
            co,
            vec2(x_pos + 2.0, PIPE_BOTTOM),
            vec2(x_pos + 10.0, bottom_part_end),
            rgb(133, 168, 75),
            rgb(228, 250, 145),
        );
        self.draw_horz_gradient_rect(
            co,
            vec2(x_pos + 10.0, PIPE_BOTTOM),
            vec2(x_pos + 20.0, bottom_part_end),
            rgb(228, 250, 145),
            rgb(86, 126, 41),
        );
        self.draw_horz_gradient_rect(
            co,
            vec2(x_pos + 20.0, PIPE_BOTTOM),
            vec2(x_pos + 24.0, bottom_part_end),
            rgb(86, 126, 41),
            rgb(86, 126, 41),
        );

        // shadows
        self.draw_horz_line(
            co,
            bottom_part_end - 1.0,
            x_pos + 2.0,
            x_pos + PIPE_WIDTH - 2.0,
            rgb(86, 126, 41),
        );

        // draw the pipe opening
        // outlines
        self.draw_vert_line(
            co,
            x_pos,
            bottom_part_end,
            bottom_part_end + PIPE_HOLE_HEIGHT,
            PIPE_OUTLINE_COLOR,
        );
        self.draw_vert_line(
            co,
            x_pos + PIPE_WIDTH - 1.0,
            bottom_part_end,
            bottom_part_end + PIPE_HOLE_HEIGHT,
            PIPE_OUTLINE_COLOR,
        );
        self.draw_horz_line(
            co,
            bottom_part_end,
            x_pos,
            x_pos + PIPE_WIDTH - 1.0,
            PIPE_OUTLINE_COLOR,
        );
        self.draw_horz_line(
            co,
            bottom_part_end + PIPE_HOLE_HEIGHT - 1.0,
            x_pos,
            x_pos + PIPE_WIDTH - 1.0,
            PIPE_OUTLINE_COLOR,
        );

        // gradient fills
        let gradient_bottom: f32 = bottom_part_end + 1.0;
        let gradient_top: f32 = bottom_part_end + PIPE_HOLE_HEIGHT - 1.0;
        self.draw_horz_gradient_rect(
            co,
            vec2(x_pos + 1.0, gradient_bottom),
            vec2(x_pos + 5.0, gradient_top),
            rgb(221, 234, 131),
            rgb(228, 250, 145),
        );
        self.draw_horz_gradient_rect(
            co,
            vec2(x_pos + 5.0, gradient_bottom),
            vec2(x_pos + 22.0, gradient_top),
            rgb(228, 250, 145),
            rgb(86, 126, 41),
        );
        self.draw_horz_gradient_rect(
            co,
            vec2(x_pos + 22.0, gradient_bottom),
            vec2(x_pos + 25.0, gradient_top),
            rgb(86, 126, 41),
            rgb(86, 126, 41),
        );

        // shadows
        self.draw_horz_line(
            co,
            gradient_bottom,
            x_pos + 1.0,
            x_pos + 25.0,
            rgb(86, 126, 41),
        );
        self.draw_horz_line(
            co,
            gradient_top - 1.0,
            x_pos + 1.0,
            x_pos + 25.0,
            rgb(122, 158, 67),
        );
    }

    fn draw_top_pipe(&mut self, co: Vec2, x_pos: f32, height: f32) {
        let bounds: Vec2 = self.get_level_bounds();

        if (co.x < x_pos)
            || (co.x > (x_pos + PIPE_WIDTH))
            || (co.y < (bounds.y - height))
            || (co.y > bounds.y)
        {
            return;
        }

        // draw the bottom part of the pipe
        // outlines
        let bottom_part_end: f32 = bounds.y + PIPE_HOLE_HEIGHT - height;
        self.draw_vert_line(
            co,
            x_pos + 1.0,
            bottom_part_end,
            bounds.y,
            PIPE_OUTLINE_COLOR,
        );
        self.draw_vert_line(
            co,
            x_pos + PIPE_WIDTH - 2.0,
            bottom_part_end,
            bounds.y,
            PIPE_OUTLINE_COLOR,
        );

        // gradient fills
        self.draw_horz_gradient_rect(
            co,
            vec2(x_pos + 2.0, bottom_part_end),
            vec2(x_pos + 10.0, bounds.y),
            rgb(133, 168, 75),
            rgb(228, 250, 145),
        );
        self.draw_horz_gradient_rect(
            co,
            vec2(x_pos + 10.0, bottom_part_end),
            vec2(x_pos + 20.0, bounds.y),
            rgb(228, 250, 145),
            rgb(86, 126, 41),
        );
        self.draw_horz_gradient_rect(
            co,
            vec2(x_pos + 20.0, bottom_part_end),
            vec2(x_pos + 24.0, bounds.y),
            rgb(86, 126, 41),
            rgb(86, 126, 41),
        );

        // shadows
        self.draw_horz_line(
            co,
            bottom_part_end + 1.0,
            x_pos + 2.0,
            x_pos + PIPE_WIDTH - 2.0,
            rgb(86, 126, 41),
        );

        // draw the pipe opening
        // outlines
        self.draw_vert_line(
            co,
            x_pos,
            bottom_part_end - PIPE_HOLE_HEIGHT,
            bottom_part_end,
            PIPE_OUTLINE_COLOR,
        );
        self.draw_vert_line(
            co,
            x_pos + PIPE_WIDTH - 1.0,
            bottom_part_end - PIPE_HOLE_HEIGHT,
            bottom_part_end,
            PIPE_OUTLINE_COLOR,
        );
        self.draw_horz_line(
            co,
            bottom_part_end,
            x_pos,
            x_pos + PIPE_WIDTH,
            PIPE_OUTLINE_COLOR,
        );
        self.draw_horz_line(
            co,
            bottom_part_end - PIPE_HOLE_HEIGHT,
            x_pos,
            x_pos + PIPE_WIDTH - 1.0,
            PIPE_OUTLINE_COLOR,
        );

        // gradient fills
        let gradient_bottom: f32 = bottom_part_end - PIPE_HOLE_HEIGHT + 1.0;
        let gradient_top: f32 = bottom_part_end;
        self.draw_horz_gradient_rect(
            co,
            vec2(x_pos + 1.0, gradient_bottom),
            vec2(x_pos + 5.0, gradient_top),
            rgb(221, 234, 131),
            rgb(228, 250, 145),
        );
        self.draw_horz_gradient_rect(
            co,
            vec2(x_pos + 5.0, gradient_bottom),
            vec2(x_pos + 22.0, gradient_top),
            rgb(228, 250, 145),
            rgb(86, 126, 41),
        );
        self.draw_horz_gradient_rect(
            co,
            vec2(x_pos + 22.0, gradient_bottom),
            vec2(x_pos + 25.0, gradient_top),
            rgb(86, 126, 41),
            rgb(86, 126, 41),
        );

        // shadows
        self.draw_horz_line(
            co,
            gradient_bottom,
            x_pos + 1.0,
            x_pos + 25.0,
            rgb(122, 158, 67),
        );
        self.draw_horz_line(
            co,
            gradient_top - 1.0,
            x_pos + 1.0,
            x_pos + 25.0,
            rgb(86, 126, 41),
        );
    }

    fn draw_bush_group(&mut self, mut bottom_corner: Vec2, co: Vec2) {
        self.draw_tile(0, bottom_corner, co);
        bottom_corner.x += 13.0;

        self.draw_tile(1, bottom_corner, co);
        bottom_corner.x += 13.0;

        self.draw_tile(0, bottom_corner, co);
    }

    fn draw_bushes(&mut self, co: Vec2) {
        self.draw_horz_rect(co.y, 39.0, 70.0, rgb(100, 224, 117));

        let mut i = 0;
        while i < 20 {
            let x_offset: f32 = i as f32 * 45.0;
            self.draw_bush_group(vec2(x_offset, 70.0), co);
            self.draw_bush_group(vec2(x_offset + 7.0, 68.0), co);
            self.draw_bush_group(vec2(x_offset - 16.0, 65.0), co);
            i += 1;
        }
    }

    fn draw_clouds(&mut self, co: Vec2) {
        let mut i = 0;
        while i < 20 {
            let x_offset: f32 = i as f32 * 40.0;
            self.draw_tile(2, vec2(x_offset, 95.0), co);
            self.draw_tile(2, vec2(x_offset + 14.0, 91.0), co);
            self.draw_tile(2, vec2(x_offset + 28.0, 93.0), co);
            i += 1;
        }

        self.draw_horz_rect(co.y, 70.0, 95.0, rgb(233, 251, 218));
    }

    fn draw_pipe_pair(&mut self, co: Vec2, x_pos: f32, bottom_pipe_height: f32) {
        let bounds: Vec2 = self.get_level_bounds();
        let top_pipe_height: f32 =
            bounds.y - (VERT_PIPE_DISTANCE + PIPE_BOTTOM + bottom_pipe_height);

        self.draw_bottom_pipe(co, x_pos, bottom_pipe_height);
        self.draw_top_pipe(co, x_pos, top_pipe_height);
    }

    fn draw_pipes(&mut self, co: Vec2) {
        // calculate the starting position of the pipes according to the current frame
        let animation_cycle_length: f32 = HORZ_PIPE_DISTANCE * PIPE_PER_CYCLE; // the number of frames after which the animation should repeat itself
        let f: i32 = (self.inputs.time * 60.0).rem_euclid(animation_cycle_length) as i32;
        let mut x_pos: f32 = -f as f32;

        let center: f32 = (PIPE_MAX + PIPE_MIN) / 2.0;
        let half_top: f32 = (center + PIPE_MAX) / 2.0;
        let half_bottom: f32 = (center + PIPE_MIN) / 2.0;

        let mut i = 0;
        while i < 12 {
            let mut y_pos: f32 = center;
            let cycle: i32 = (i as f32).rem_euclid(8.0) as i32;

            if (cycle == 1) || (cycle == 3) {
                y_pos = half_top;
            } else if cycle == 2 {
                y_pos = PIPE_MAX;
            } else if (cycle == 5) || (cycle == 7) {
                y_pos = half_bottom;
            } else if cycle == 6 {
                y_pos = PIPE_MIN;
            }

            self.draw_pipe_pair(co, x_pos, y_pos);
            x_pos += HORZ_PIPE_DISTANCE;
            i += 1;
        }
    }

    fn draw_bird(&mut self, co: Vec2) {
        let animation_cycle_length: f32 = HORZ_PIPE_DISTANCE * PIPE_PER_CYCLE; // the number of frames after which the animation should repeat itself
        let cycle_frame: i32 = (self.inputs.time * 60.0).rem_euclid(animation_cycle_length) as i32;
        let f_cycle_frame: f32 = cycle_frame as f32;

        let start_pos: f32 = 110.0;
        let speed: f32 = 2.88;
        let updown_delta: f32 = 0.16;
        let acceleration: f32 = -0.0975;
        let jump_frame: f32 = (self.inputs.time * 60.0).rem_euclid(30.0) as i32 as f32;
        let horz_dist: i32 = HORZ_PIPE_DISTANCE as i32;

        // calculate the "jumping" effect on the Y axis.
        // Using equations of motion, const acceleration: x = x0 + v0*t + 1/2at^2
        let mut y_pos: f32 = start_pos + speed * jump_frame + acceleration * jump_frame.powf(2.0);

        let speed_delta: f32 = updown_delta * f_cycle_frame.rem_euclid(HORZ_PIPE_DISTANCE);
        let mut prev_up_cycles: i32 = 0;
        let mut prev_down_cycles: i32 = 0;

        // count the number of pipes we've already passed.
        // for each such pipe, we deduce if we went "up" or "down" in Y
        let cycle_count: i32 = (f_cycle_frame / HORZ_PIPE_DISTANCE) as i32;

        let mut i = 0;
        while i < 10 {
            if i <= cycle_count {
                if i == 1 {
                    prev_up_cycles += 1;
                }

                if (i >= 2) && (i < 6) {
                    prev_down_cycles += 1;
                }
                if i >= 6 {
                    prev_up_cycles += 1;
                }
            }
            i += 1;
        }

        // add up/down delta from all the previous pipes
        y_pos += ((prev_up_cycles - prev_down_cycles) as f32) * HORZ_PIPE_DISTANCE * updown_delta;

        // calculate the up/down delta for the current two pipes, and add it to the previous result
        if ((cycle_frame >= 0) && (cycle_frame < horz_dist))
            || ((cycle_frame >= 5 * horz_dist) && (cycle_frame < 9 * horz_dist))
        {
            y_pos += speed_delta;
        } else {
            y_pos -= speed_delta;
        }

        let anim_frame: i32 = (self.inputs.time * 7.0).rem_euclid(3.0) as i32;
        if anim_frame == 0 {
            self.draw_tile(3, vec2(105.0, y_pos as i32 as f32), co);
        }
        if anim_frame == 1 {
            self.draw_tile(4, vec2(105.0, y_pos as i32 as f32), co);
        }
        if anim_frame == 2 {
            self.draw_tile(5, vec2(105.0, y_pos as i32 as f32), co);
        }
    }

    pub fn main_image(&mut self, frag_color: &mut Vec4, frag_coord: Vec2) {
        let level_pixel: Vec2 = self.get_level_pixel(frag_coord);

        self.frag_color = rgb(113, 197, 207); // draw the blue sky background

        self.draw_ground(level_pixel);
        self.draw_green_stripes(level_pixel);
        self.draw_clouds(level_pixel);
        self.draw_bushes(level_pixel);
        self.draw_pipes(level_pixel);
        self.draw_bird(level_pixel);

        *frag_color = self.frag_color;
    }
}
