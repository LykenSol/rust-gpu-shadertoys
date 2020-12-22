//! Ported to Rust from <https://www.shadertoy.com/view/MdjGR1>
//!
//! Original comment:
//! ```glsl
//! // The MIT License
//! // Copyright © 2013 Inigo Quilez
//! // Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//!
//! // A test on using ray differentials (only primary rays for now) to choose texture filtering
//! // footprint, and adaptively supersample/filter the procedural texture/patter (up to a rate
//! // of 10x10).
//!
//! // This solves texture aliasing without resorting to full-screen 10x10 supersampling, which would
//! // involve doing raytracing and lighting 10x10 times (not realtime at all).
//!
//! // The tecnique should be used to filter every texture independently. The ratio of the supersampling
//! // could be inveresely proportional to the screen/lighing supersampling rate such that the cost
//! // of texturing would be constant no matter the final image quality settings.
//! */
//! ```

use shared::*;
use spirv_std::glam::{const_vec4, vec2, vec3, Vec2, Vec3, Vec3Swizzles, Vec4, Vec4Swizzles};

// Note: This cfg is incorrect on its surface, it really should be "are we compiling with std", but
// we tie #[no_std] above to the same condition, so it's fine.
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

pub struct Inputs {
    pub resolution: Vec3,
    pub time: f32,
    pub mouse: Vec4,
}

//===============================================================================================
//===============================================================================================

const MAX_SAMPLES: i32 = 10; // 10*10

//===============================================================================================
//===============================================================================================
// noise implementation
//===============================================================================================
//===============================================================================================

fn hash3(mut p: Vec3) -> Vec3 {
    p = vec3(
        p.dot(vec3(127.1, 311.7, 74.7)),
        p.dot(vec3(269.5, 183.3, 246.1)),
        p.dot(vec3(113.5, 271.9, 124.6)),
    );

    -Vec3::one() + 2.0 * (p.sin() * 13.5453123).gl_fract()
}

fn noise(p: Vec3) -> f32 {
    let i: Vec3 = p.floor();
    let f: Vec3 = p.gl_fract();

    let u: Vec3 = f * f * (Vec3::splat(3.0) - 2.0 * f);

    mix(
        mix(
            mix(
                hash3(i + vec3(0.0, 0.0, 0.0)).dot(f - vec3(0.0, 0.0, 0.0)),
                hash3(i + vec3(1.0, 0.0, 0.0)).dot(f - vec3(1.0, 0.0, 0.0)),
                u.x,
            ),
            mix(
                hash3(i + vec3(0.0, 1.0, 0.0)).dot(f - vec3(0.0, 1.0, 0.0)),
                hash3(i + vec3(1.0, 1.0, 0.0)).dot(f - vec3(1.0, 1.0, 0.0)),
                u.x,
            ),
            u.y,
        ),
        mix(
            mix(
                hash3(i + vec3(0.0, 0.0, 1.0)).dot(f - vec3(0.0, 0.0, 1.0)),
                hash3(i + vec3(1.0, 0.0, 1.0)).dot(f - vec3(1.0, 0.0, 1.0)),
                u.x,
            ),
            mix(
                hash3(i + vec3(0.0, 1.0, 1.0)).dot(f - vec3(0.0, 1.0, 1.0)),
                hash3(i + vec3(1.0, 1.0, 1.0)).dot(f - vec3(1.0, 1.0, 1.0)),
                u.x,
            ),
            u.y,
        ),
        u.z,
    )
}

//===============================================================================================
//===============================================================================================
// sphere implementation
//===============================================================================================
//===============================================================================================

fn soft_shadow_sphere(ro: Vec3, rd: Vec3, sph: Vec4) -> f32 {
    let oc: Vec3 = sph.xyz() - ro;
    let b: f32 = oc.dot(rd);

    let mut res: f32 = 1.0;
    if b > 0.0 {
        let h: f32 = oc.dot(oc) - b * b - sph.w * sph.w;
        res = smoothstep(0.0, 1.0, 2.0 * h / b);
    }
    res
}

fn occ_sphere(sph: Vec4, pos: Vec3, nor: Vec3) -> f32 {
    let di: Vec3 = sph.xyz() - pos;
    let l: f32 = di.length();
    1.0 - nor.dot(di / l) * sph.w * sph.w / (l * l)
}

fn i_sphere(ro: Vec3, rd: Vec3, sph: Vec4) -> f32 {
    let mut t: f32 = -1.0;
    let ce: Vec3 = ro - sph.xyz();
    let b: f32 = rd.dot(ce);
    let c: f32 = ce.dot(ce) - sph.w * sph.w;
    let h: f32 = b * b - c;
    if h > 0.0 {
        t = -b - h.sqrt();
    }

    t
}

//===============================================================================================
//===============================================================================================
// scene
//===============================================================================================
//===============================================================================================

// spheres
const SC0: Vec4 = const_vec4!([0.0, 1.0, 0.0, 1.0]);
const SC1: Vec4 = const_vec4!([0.0, 1.0, 14.0, 4.0]);
const SC2: Vec4 = const_vec4!([-11.0, 1.0, 12.0, 4.0]);
const SC3: Vec4 = const_vec4!([13.0, 1.0, -10.0, 4.0]);

fn intersect(
    ro: Vec3,
    rd: Vec3,
    pos: &mut Vec3,
    nor: &mut Vec3,
    occ: &mut f32,
    matid: &mut f32,
) -> f32 {
    // raytrace
    let mut tmin: f32 = 10000.0;
    *nor = Vec3::zero();
    *occ = 1.0;
    *pos = Vec3::zero();

    // raytrace-plane
    let mut h: f32 = (0.0 - ro.y) / rd.y;
    if h > 0.0 {
        tmin = h;
        *nor = vec3(0.0, 1.0, 0.0);
        *pos = ro + h * rd;
        *matid = 0.0;
        *occ = occ_sphere(SC0, *pos, *nor)
            * occ_sphere(SC1, *pos, *nor)
            * occ_sphere(SC2, *pos, *nor)
            * occ_sphere(SC3, *pos, *nor);
    }

    // raytrace-sphere
    h = i_sphere(ro, rd, SC0);
    if h > 0.0 && h < tmin {
        tmin = h;
        *pos = ro + h * rd;
        *nor = (*pos - SC0.xyz()).normalize();
        *matid = 1.0;
        *occ = 0.5 + 0.5 * nor.y;
    }

    h = i_sphere(ro, rd, SC1);
    if h > 0.0 && h < tmin {
        tmin = h;
        *pos = ro + tmin * rd;
        *nor = (ro + h * rd - SC1.xyz()).normalize();
        *matid = 1.0;
        *occ = 0.5 + 0.5 * nor.y;
    }

    h = i_sphere(ro, rd, SC2);
    if h > 0.0 && h < tmin {
        tmin = h;
        *pos = ro + tmin * rd;
        *nor = (ro + h * rd - SC2.xyz()).normalize();
        *matid = 1.0;
        *occ = 0.5 + 0.5 * nor.y;
    }

    h = i_sphere(ro, rd, SC3);
    if h > 0.0 && h < tmin {
        tmin = h;
        *pos = ro + tmin * rd;
        *nor = (ro + h * rd - SC3.xyz()).normalize();
        *matid = 1.0;
        *occ = 0.5 + 0.5 * nor.y;
    }

    tmin
}

fn tex_coords(p: Vec3) -> Vec3 {
    64.0 * p
}

fn mytexture(mut p: Vec3, _n: Vec3, matid: f32) -> Vec3 {
    p += Vec3::splat(0.1);
    let ip: Vec3 = (p / 20.0).floor();
    let fp: Vec3 = (Vec3::splat(0.5) + p / 20.0).gl_fract();

    let mut id: f32 = ((ip.dot(vec3(127.1, 311.7, 74.7))).sin() * 58.5453123).gl_fract();
    id = mix(id, 0.3, matid);

    let f: f32 = (ip.x + (ip.y + ip.z.rem_euclid(2.0)).rem_euclid(2.0)).rem_euclid(2.0);

    let mut g: f32 =
        0.5 + 1.0 * noise(p * mix(vec3(0.2 + 0.8 * f, 1.0, 1.0 - 0.8 * f), Vec3::one(), matid));

    g *= mix(
        smoothstep(0.03, 0.04, (fp.x - 0.5).abs() / 0.5)
            * smoothstep(0.03, 0.04, (fp.z - 0.5).abs() / 0.5),
        1.0,
        matid,
    );

    let col: Vec3 =
        Vec3::splat(0.5) + 0.5 * (Vec3::splat(1.0 + 2.0 * id) + vec3(0.0, 1.0, 2.0)).sin();

    col * g
}

impl Inputs {
    fn calc_camera(&self, ro: &mut Vec3, ta: &mut Vec3) {
        let an: f32 = 0.1 * self.time;
        *ro = vec3(5.5 * an.cos(), 1.0, 5.5 * an.sin());
        *ta = vec3(0.0, 1.0, 0.0);
    }
}

fn do_lighting(pos: Vec3, nor: Vec3, occ: f32, rd: Vec3) -> Vec3 {
    let sh: f32 = soft_shadow_sphere(pos, Vec3::splat(0.57703), SC0)
        .min(soft_shadow_sphere(pos, Vec3::splat(0.57703), SC1))
        .min(soft_shadow_sphere(pos, Vec3::splat(0.57703), SC2))
        .min(soft_shadow_sphere(pos, Vec3::splat(0.57703), SC3));
    let dif: f32 = nor.dot(Vec3::splat(0.57703)).clamp(0.0, 1.0);
    let bac: f32 = nor.dot(vec3(-0.707, 0.0, -0.707)).clamp(0.0, 1.0);
    let mut lin: Vec3 = dif * vec3(1.50, 1.40, 1.30) * sh;
    lin += occ * vec3(0.15, 0.20, 0.30);
    lin += bac * vec3(0.20, 0.20, 0.20);
    lin += Vec3::splat(
        sh * 0.8
            * rd.reflect(nor)
                .dot(Vec3::splat(0.57703))
                .clamp(0.0, 1.0)
                .powf(12.0),
    );

    lin
}
//===============================================================================================
//===============================================================================================
// render
//===============================================================================================
//===============================================================================================
impl Inputs {
    fn calc_ray_for_pixel(&self, pix: Vec2, res_ro: &mut Vec3, res_rd: &mut Vec3) {
        let p: Vec2 = (-self.resolution.xy() + 2.0 * pix) / self.resolution.y;
        // camera movement
        let mut ro: Vec3 = Vec3::zero();
        let mut ta: Vec3 = Vec3::zero();
        self.calc_camera(&mut ro, &mut ta);
        // camera matrix
        let ww: Vec3 = (ta - ro).normalize();
        let uu: Vec3 = ww.cross(vec3(0.0, 1.0, 0.0)).normalize();
        let vv: Vec3 = uu.cross(ww).normalize();
        // create view ray
        let rd: Vec3 = (p.x * uu + p.y * vv + 1.5 * ww).normalize();

        *res_ro = ro;
        *res_rd = rd;
    }
}

// sample a procedural texture with filtering
fn sample_texture_with_filter(
    uvw: Vec3,
    ddx_uvw: Vec3,
    ddy_uvw: Vec3,
    nor: Vec3,
    mid: f32,
) -> Vec3 {
    let sx: i32 = 1 + (4.0 * (ddx_uvw - uvw).length()).clamp(0.0, (MAX_SAMPLES - 1) as f32) as i32;
    let sy: i32 = 1 + (4.0 * (ddy_uvw - uvw).length()).clamp(0.0, (MAX_SAMPLES - 1) as f32) as i32;

    let mut no: Vec3 = Vec3::zero();

    if true {
        let mut j = 0;
        while j < MAX_SAMPLES {
            let mut i = 0;
            while i < MAX_SAMPLES {
                if j < sy && i < sx {
                    let st: Vec2 = vec2(i as f32, j as f32) / vec2(sx as f32, sy as f32);
                    no += mytexture(
                        uvw + st.x * (ddx_uvw - uvw) + st.y * (ddy_uvw - uvw),
                        nor,
                        mid,
                    );
                }
                i += 1;
            }
            j += 1;
        }
    } else {
        let mut j = 0;
        while j < sy {
            let mut i = 0;
            while i < sx {
                let st: Vec2 = vec2(i as f32, j as f32) / vec2(sx as f32, sy as f32);
                no += mytexture(
                    uvw + st.x * (ddx_uvw - uvw) + st.y * (ddy_uvw - uvw),
                    nor,
                    mid,
                );
                i += 1;
            }
            j += 1;
        }
    }

    no / (sx * sy) as f32
}

fn sample_texture(uvw: Vec3, nor: Vec3, mid: f32) -> Vec3 {
    mytexture(uvw, nor, mid)
}

impl Inputs {
    pub fn main_image(&mut self, frag_color: &mut Vec4, frag_coord: Vec2) {
        let p: Vec2 = (-self.resolution.xy() + 2.0 * frag_coord) / self.resolution.y;
        let mut th: f32 = (-self.resolution.x + 2.0 * self.mouse.x) / self.resolution.y;

        if self.mouse.z < 0.01 {
            th = 0.5 / self.resolution.y;
        }

        let mut ro: Vec3 = Vec3::zero();
        let mut rd: Vec3 = Vec3::zero();
        let mut ddx_ro: Vec3 = Vec3::zero();
        let mut ddx_rd: Vec3 = Vec3::zero();
        let mut ddy_ro: Vec3 = Vec3::zero();
        let mut ddy_rd: Vec3 = Vec3::zero();
        self.calc_ray_for_pixel(frag_coord + vec2(0.0, 0.0), &mut ro, &mut rd);
        self.calc_ray_for_pixel(frag_coord + vec2(1.0, 0.0), &mut ddx_ro, &mut ddx_rd);
        self.calc_ray_for_pixel(frag_coord + vec2(0.0, 1.0), &mut ddy_ro, &mut ddy_rd);

        // trace
        let mut pos: Vec3 = Vec3::zero();
        let mut nor: Vec3 = Vec3::zero();
        let mut occ: f32 = 0.0;
        let mut mid: f32 = 0.0;
        let t: f32 = intersect(ro, rd, &mut pos, &mut nor, &mut occ, &mut mid);

        let mut col: Vec3 = Vec3::splat(0.9);

        let uvw: Vec3;
        let ddx_uvw: Vec3;
        let ddy_uvw: Vec3;

        if t < 100.0 {
            if true {
                // -----------------------------------------------------------------------
                // compute ray differentials by intersecting the tangent plane to the
                // surface.
                // -----------------------------------------------------------------------

                // computer ray differentials
                let ddx_pos: Vec3 = ddx_ro - ddx_rd * (ddx_ro - pos).dot(nor) / ddx_rd.dot(nor);
                let ddy_pos: Vec3 = ddy_ro - ddy_rd * (ddy_ro - pos).dot(nor) / ddy_rd.dot(nor);

                // calc texture sampling footprint
                uvw = tex_coords(pos);
                ddx_uvw = tex_coords(ddx_pos);
                ddy_uvw = tex_coords(ddy_pos);
            } else {
                // -----------------------------------------------------------------------
                // Because we are in the GPU, we do have access to differentials directly
                // This wouldn't be the case in a regular raytrace.
                // It wouldn't work as well in shaders doing interleaved calculations in
                // pixels (such as some of the 3D/stereo shaders here in Shadertoy)
                // -----------------------------------------------------------------------
                uvw = tex_coords(pos);

                // calc texture sampling footprint
                ddx_uvw = uvw + uvw.ddx();
                ddy_uvw = uvw + uvw.ddy();
            }
            // shading
            let mate: Vec3;

            if p.x > th {
                mate = sample_texture(uvw, nor, mid);
            } else {
                mate = sample_texture_with_filter(uvw, ddx_uvw, ddy_uvw, nor, mid);
            }

            // lighting
            let lin: Vec3 = do_lighting(pos, nor, occ, rd);

            // combine lighting with material
            col = mate * lin;

            // fog
            col = mix(col, Vec3::splat(0.9), 1.0 - (-0.0002 * t * t).exp());
        }

        // gamma correction
        col = col.powf(0.4545);

        col *= smoothstep(0.006, 0.008, (p.x - th).abs());

        *frag_color = col.extend(1.0);
    }
}
