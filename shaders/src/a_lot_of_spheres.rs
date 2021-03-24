//! Ported to Rust from <https://www.shadertoy.com/view/lsX3WH>
//!
//! Original comment:
//! ```glsl
//! // A lot of spheres. Created by Reinder Nijhoff 2013
//! // Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.
//! // @reindernijhoff
//! //
//! // https://www.shadertoy.com/view/lsX3WH
//! //
//! */
//! ```

use shared::*;
use glam::{const_mat2, vec2, vec3, Mat2, Vec2, Vec3, Vec3Swizzles, Vec4};

// Note: This cfg is incorrect on its surface, it really should be "are we compiling with std", but
// we tie #[no_std] above to the same condition, so it's fine.
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

pub struct Inputs {
    pub resolution: Vec3,
    pub time: f32,
}

const SHADOW: bool = true;
const REFLECTION: bool = true;

const RAYCASTSTEPS: usize = 40;

const EXPOSURE: f32 = 0.9;
const EPSILON: f32 = 0.0001;
const MAXDISTANCE: f32 = 400.0;
const GRIDSIZE: f32 = 8.0;
const GRIDSIZESMALL: f32 = 5.0;
const MAXHEIGHT: f32 = 30.0;
const SPEED: f32 = 0.5;

//
// math functions
//

const _MR: Mat2 = const_mat2!([0.84147, 0.54030, 0.54030, -0.84147]);
fn hash(n: f32) -> f32 {
    (n.sin() * 43758.5453).gl_fract()
}
fn _hash2(n: f32) -> Vec2 {
    (vec2(n, n + 1.0).sin() * vec2(2.1459123, 3.3490423)).gl_fract()
}
fn hash2_vec(n: Vec2) -> Vec2 {
    (vec2(n.x * n.y, n.x + n.y).sin() * vec2(2.1459123, 3.3490423)).gl_fract()
}
fn _hash3(n: f32) -> Vec3 {
    (vec3(n, n + 1.0, n + 2.0).sin() * vec3(3.5453123, 4.1459123, 1.3490423)).gl_fract()
}
fn hash3_vec(n: Vec2) -> Vec3 {
    (vec3(n.x, n.y, n.x + 2.0).sin() * vec3(3.5453123, 4.1459123, 1.3490423)).gl_fract()
}

//
// intersection functions
//

fn intersect_plane(ro: Vec3, rd: Vec3, height: f32, dist: &mut f32) -> bool {
    if rd.y == 0.0 {
        return false;
    }

    let mut d: f32 = -(ro.y - height) / rd.y;
    d = d.min(100000.0);
    if d > 0.0 {
        *dist = d;
        return true;
    }
    false
}

fn intersect_unit_sphere(ro: Vec3, rd: Vec3, sph: Vec3, dist: &mut f32, normal: &mut Vec3) -> bool {
    let ds: Vec3 = ro - sph;
    let bs: f32 = rd.dot(ds);
    let cs: f32 = ds.dot(ds) - 1.0;
    let mut ts: f32 = bs * bs - cs;

    if ts > 0.0 {
        ts = -bs - ts.sqrt();
        if ts > 0.0 {
            *normal = ((ro + ts * rd) - sph).normalize();
            *dist = ts;
            return true;
        }
    }
    false
}

//
// Scene
//

fn get_sphere_offset(grid: Vec2, center: &mut Vec2) {
    *center = (hash2_vec(grid + vec2(43.12, 1.23)) - Vec2::splat(0.5)) * (GRIDSIZESMALL);
}

impl Inputs {
    fn get_moving_sphere_position(&self, grid: Vec2, sphere_offset: Vec2, center: &mut Vec3) {
        // falling?
        let s: f32 = 0.1 + hash(grid.x * 1.23114 + 5.342 + 74.324231 * grid.y);
        let t: f32 = (14. * s + self.time / s * 0.3).gl_fract();

        let y: f32 = s * MAXHEIGHT * (4.0 * t * (1. - t)).abs();
        let offset: Vec2 = grid + sphere_offset;

        *center = vec3(offset.x, y, offset.y) + 0.5 * vec3(GRIDSIZE, 2.0, GRIDSIZE);
    }
}
fn get_sphere_position(grid: Vec2, sphere_offset: Vec2, center: &mut Vec3) {
    let offset: Vec2 = grid + sphere_offset;
    *center = vec3(offset.x, 0.0, offset.y) + 0.5 * vec3(GRIDSIZE, 2.0, GRIDSIZE);
}
fn get_sphere_color(grid: Vec2) -> Vec3 {
    hash3_vec(grid + vec2(43.12 * grid.y, 12.23 * grid.x)).normalize()
}

impl Inputs {
    fn trace(
        &self,
        ro: Vec3,
        rd: Vec3,
        intersection: &mut Vec3,
        normal: &mut Vec3,
        dist: &mut f32,
        material: &mut i32,
    ) -> Vec3 {
        *material = 0; // sky
        *dist = MAXDISTANCE;
        let mut distcheck: f32 = 0.0;

        let mut sphere_center: Vec3 = Vec3::zero();
        let mut col: Vec3;
        let mut normalcheck: Vec3 = Vec3::zero();
        if intersect_plane(ro, rd, 0.0, &mut distcheck) && distcheck < MAXDISTANCE {
            *dist = distcheck;
            *material = 1;
            *normal = vec3(0.0, 1.0, 0.0);
            col = Vec3::one();
        } else {
            col = Vec3::zero();
        }

        // trace grid
        let mut pos: Vec3 = (ro / GRIDSIZE).floor() * GRIDSIZE;
        let ri: Vec3 = 1.0 / rd;
        let rs: Vec3 = rd.gl_sign() * GRIDSIZE;
        let mut dis: Vec3 = (pos - ro + 0.5 * Vec3::splat(GRIDSIZE) + rs * 0.5) * ri;
        let mut mm: Vec3;

        let mut i = 0;
        while i < RAYCASTSTEPS {
            if *material > 1 || ro.xz().distance(pos.xz()) > *dist + GRIDSIZE {
                break;
            }
            let mut offset: Vec2 = Vec2::zero();
            get_sphere_offset(pos.xz(), &mut offset);

            self.get_moving_sphere_position(pos.xz(), -offset, &mut sphere_center);

            if intersect_unit_sphere(ro, rd, sphere_center, &mut distcheck, &mut normalcheck)
                && distcheck < *dist
            {
                *dist = distcheck;
                *normal = normalcheck;
                *material = 2;
            }

            get_sphere_position(pos.xz(), offset, &mut sphere_center);
            if intersect_unit_sphere(ro, rd, sphere_center, &mut distcheck, &mut normalcheck)
                && distcheck < *dist
            {
                *dist = distcheck;
                *normal = normalcheck;
                col = Vec3::splat(2.0);
                *material = 3;
            }
            mm = dis.step(dis.zyx());
            dis += mm * rs * ri;
            pos += mm * rs;
            i += 1;
        }

        let mut color: Vec3 = Vec3::zero();
        if *material > 0 {
            *intersection = ro + rd * *dist;
            let map: Vec2 = (intersection.xz() / GRIDSIZE).floor() * GRIDSIZE;

            if *material == 1 || *material == 3 {
                // lightning
                let c: Vec3 = vec3(-GRIDSIZE, 0.0, GRIDSIZE);
                let mut x = 0;
                while x < 3 {
                    let mut y = 0;
                    while y < 3 {
                        let mapoffset: Vec2 = map + vec2([c.x, c.y, c.z][x], [c.x, c.y, c.z][y]);
                        let mut offset: Vec2 = Vec2::zero();
                        get_sphere_offset(mapoffset, &mut offset);
                        let lcolor: Vec3 = get_sphere_color(mapoffset);
                        let mut lpos: Vec3 = Vec3::zero();
                        self.get_moving_sphere_position(mapoffset, -offset, &mut lpos);

                        let mut shadow: f32 = 1.0;

                        if SHADOW && *material == 1 {
                            let mut sx = 0;
                            while sx < 3 {
                                let mut sy = 0;
                                while sy < 3 {
                                    if shadow < 1.0 {
                                        sy += 1;
                                        continue;
                                    }

                                    let smapoffset: Vec2 =
                                        map + vec2([c.x, c.y, c.z][x], [c.x, c.y, c.z][y]);
                                    let mut soffset: Vec2 = Vec2::zero();
                                    get_sphere_offset(smapoffset, &mut soffset);
                                    let mut slpos: Vec3 = Vec3::zero();
                                    let mut sn: Vec3 = Vec3::zero();
                                    get_sphere_position(smapoffset, soffset, &mut slpos);
                                    let mut sd: f32 = 0.0;
                                    if intersect_unit_sphere(
                                        *intersection,
                                        (lpos - *intersection).normalize(),
                                        slpos,
                                        &mut sd,
                                        &mut sn,
                                    ) {
                                        shadow = 0.0;
                                    }
                                    sy += 1;
                                }
                                sx += 1;
                            }
                        }
                        color += col
                            * lcolor
                            * (shadow
                                * ((lpos - *intersection).normalize().dot(*normal)).max(0.0)
                                * (1. - (lpos.distance(*intersection) / GRIDSIZE).clamp(0.0, 1.)));
                        y += 1;
                    }
                    x += 1;
                }
            } else {
                // emitter
                color = (1.5 + normal.dot(vec3(0.5, 0.5, -0.5))) * get_sphere_color(map);
            }
        }
        color
    }

    pub fn main_image(&mut self, frag_color: &mut Vec4, frag_coord: Vec2) {
        let q: Vec2 = frag_coord / self.resolution.xy();
        let mut p: Vec2 = Vec2::splat(-1.0) + 2.0 * q;
        p.x *= self.resolution.x / self.resolution.y;

        // camera
        let ce: Vec3 = vec3(
            (0.232 * self.time).cos() * 10.0,
            6. + 3.0 * (0.3 * self.time).cos(),
            GRIDSIZE * (self.time / SPEED),
        );
        let ro: Vec3 = ce;
        let ta: Vec3 = ro
            + vec3(
                -(0.232 * self.time).sin() * 10.,
                -2.0 + (0.23 * self.time).cos(),
                10.0,
            );

        let roll: f32 = -0.15 * (0.5 * self.time).sin();
        // camera tx
        let cw: Vec3 = (ta - ro).normalize();
        let cp: Vec3 = vec3(roll.sin(), roll.cos(), 0.0);
        let cu: Vec3 = (cw.cross(cp)).normalize();
        let cv: Vec3 = (cu.cross(cw)).normalize();
        let mut rd: Vec3 = (p.x * cu + p.y * cv + 1.5 * cw).normalize();
        // raytrace
        let mut material: i32 = 0;
        let mut normal: Vec3 = Vec3::zero();
        let mut intersection: Vec3 = Vec3::zero();
        let mut dist: f32 = 0.0;

        let mut col: Vec3 = self.trace(
            ro,
            rd,
            &mut intersection,
            &mut normal,
            &mut dist,
            &mut material,
        );
        if material > 0 && REFLECTION {
            let ro: Vec3 = intersection + EPSILON * normal;
            rd = rd.reflect(normal);
            col += 0.05
                * self.trace(
                    ro,
                    rd,
                    &mut intersection,
                    &mut normal,
                    &mut dist,
                    &mut material,
                );
        }

        col = col.powf_vec(vec3(EXPOSURE, EXPOSURE, EXPOSURE));
        col = col.clamp(Vec3::zero(), Vec3::one());
        // vigneting
        col *= 0.25 + 0.75 * (16.0 * q.x * q.y * (1.0 - q.x) * (1.0 - q.y)).powf(0.15);

        *frag_color = col.extend(1.0);
    }
}
