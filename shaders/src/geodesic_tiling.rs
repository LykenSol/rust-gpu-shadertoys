//! Ported to Rust from <https://www.shadertoy.com/view/llVXRd>

use shared::*;
use spirv_std::glam::{
    const_mat2, const_vec2, const_vec3, vec2, vec3, Mat2, Mat3, Vec2, Vec3, Vec3Swizzles, Vec4,
    Vec4Swizzles,
};

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

    face_plane: Vec3,
    u_plane: Vec3,
    v_plane: Vec3,

    nc: Vec3,
    pab: Vec3,
    pbc: Vec3,
    pca: Vec3,

    time: f32,
}

impl State {
    pub fn new(inputs: Inputs) -> State {
        State {
            inputs,
            face_plane: Vec3::zero(),
            u_plane: Vec3::zero(),
            v_plane: Vec3::zero(),
            nc: Vec3::zero(),
            pab: Vec3::zero(),
            pbc: Vec3::zero(),
            pca: Vec3::zero(),
            time: 0.0,
        }
    }
}

const MODEL_ROTATION: Vec2 = const_vec2!([0.3, 0.25]);
const CAMERA_ROTATION: Vec2 = const_vec2!([0.5, 0.5]);

// 0: Defaults
// 1: Model
// 2: Camera
const MOUSE_CONTROL: i32 = 1;

const DEBUG: bool = false;

// 1, 2, or 3
const LOOP: usize = 0;

// --------------------------------------------------------
// HG_SDF
// https://www.shadertoy.com/view/Xs3GRB
// --------------------------------------------------------

fn p_r(p: &mut Vec2, a: f32) {
    *p = a.cos() * *p + a.sin() * vec2(p.y, -p.x);
}

fn p_reflect(p: &mut Vec3, plane_normal: Vec3, offset: f32) -> f32 {
    let t: f32 = p.dot(plane_normal) + offset;
    if t < 0.0 {
        *p = *p - (2. * t) * plane_normal;
    }
    t.gl_sign()
}

fn smax(a: f32, b: f32, r: f32) -> f32 {
    let m: f32 = a.max(b);
    if (-a < r) && (-b < r) {
        m.max(-(r - ((r + a) * (r + a) + (r + b) * (r + b)).sqrt()))
    } else {
        m
    }
}

// --------------------------------------------------------
// Icosahedron domain mirroring
// Adapted from knighty https://www.shadertoy.com/view/MsKGzw
// --------------------------------------------------------

const PI: f32 = 3.14159265359;

const TYPE: i32 = 5;

impl State {
    fn init_icosahedron(&mut self) {
        //setup folding planes and vertex
        let cospin: f32 = (PI / TYPE as f32).cos();
        let scospin: f32 = (0.75 - cospin * cospin).sqrt();
        self.nc = vec3(-0.5, -cospin, scospin); //3rd folding plane. The two others are xz and yz planes
        self.pbc = vec3(scospin, 0., 0.5); //No normalization in order to have 'barycentric' coordinates work evenly
        self.pca = vec3(0., scospin, cospin);
        self.pbc = self.pbc.normalize();
        self.pca = self.pca.normalize(); //for slightly better DE. In reality it's not necesary to apply normalization :)
        self.pab = vec3(0.0, 0.0, 1.0);

        self.face_plane = self.pca;
        self.u_plane = vec3(1.0, 0.0, 0.0).cross(self.face_plane);
        self.v_plane = vec3(1.0, 0.0, 0.0);
    }

    fn p_mod_icosahedron(&self, p: &mut Vec3) {
        *p = p.abs();
        p_reflect(p, self.nc, 0.0);
        *p = p.xy().abs().extend(p.z);
        p_reflect(p, self.nc, 0.0);
        *p = p.xy().abs().extend(p.z);
        p_reflect(p, self.nc, 0.0);
    }
}

// --------------------------------------------------------
// Triangle tiling
// Adapted from mattz https://www.shadertoy.com/view/4d2GzV
// --------------------------------------------------------

const SQRT3: f32 = 1.7320508075688772;
const I3: f32 = 0.5773502691896258;

const CART2HEX: Mat2 = const_mat2!([1.0, 0.0, I3, 2.0 * I3]);
const HEX2CART: Mat2 = const_mat2!([1.0, 0.0, -0.5, 0.5 * SQRT3]);

const _PHI: f32 = 1.618033988749895;
const _TAU: f32 = 6.283185307179586;

struct TriPoints {
    a: Vec2,
    b: Vec2,
    c: Vec2,
    center: Vec2,
    ab: Vec2,
    bc: Vec2,
    ca: Vec2,
}

fn closest_tri_points(p: Vec2) -> TriPoints {
    let p_tri: Vec2 = CART2HEX * p;
    let pi: Vec2 = p_tri.floor();
    let pf: Vec2 = p_tri.gl_fract();

    let split1: f32 = pf.y.step(pf.x);
    let split2: f32 = pf.x.step(pf.y);

    let mut a: Vec2 = vec2(split1, 1.0);
    let mut b: Vec2 = vec2(1.0, split2);
    let mut c: Vec2 = vec2(0.0, 0.0);

    a += pi;
    b += pi;
    c += pi;

    a = HEX2CART * a;
    b = HEX2CART * b;
    c = HEX2CART * c;

    let center: Vec2 = (a + b + c) / 3.;

    let ab: Vec2 = (a + b) / 2.;
    let bc: Vec2 = (b + c) / 2.;
    let ca: Vec2 = (c + a) / 2.;

    TriPoints {
        a,
        b,
        c,
        center,
        ab,
        bc,
        ca,
    }
}

// --------------------------------------------------------
// Geodesic tiling
// --------------------------------------------------------

struct TriPoints3D {
    a: Vec3,
    b: Vec3,
    c: Vec3,
    center: Vec3,
    ab: Vec3,
    bc: Vec3,
    ca: Vec3,
}

fn intersection(n: Vec3, plane_normal: Vec3, plane_offset: f32) -> Vec3 {
    let denominator: f32 = plane_normal.dot(n);
    let t: f32 = (Vec3::zero().dot(plane_normal) + plane_offset) / -denominator;
    n * t
}

//// Edge length of an icosahedron with an inscribed sphere of radius of 1
//float edgeLength = 1. / ((sqrt(3.) / 12.) * (3. + sqrt(5.)));
//// Inner radius of the icosahedron's face
//float faceRadius = (1./6.) * sqrt(3.) * edgeLength;
const FACE_RADIUS: f32 = 0.3819660112501051;

impl State {
    // 2D coordinates on the icosahedron face
    fn icosahedron_face_coordinates(&self, p: Vec3) -> Vec2 {
        let pn: Vec3 = p.normalize();
        let i: Vec3 = intersection(pn, self.face_plane, -1.0);
        vec2(i.dot(self.u_plane), i.dot(self.v_plane))
    }

    // Project 2D icosahedron face coordinates onto a sphere
    fn face_to_sphere(&self, face_point: Vec2) -> Vec3 {
        (self.face_plane + (self.u_plane * face_point.x) + (self.v_plane * face_point.y))
            .normalize()
    }

    fn geodesic_tri_points(&self, p: Vec3, subdivisions: f32) -> TriPoints3D {
        // Get 2D cartesian coordiantes on that face
        let uv: Vec2 = self.icosahedron_face_coordinates(p);

        // Get points on the nearest triangle tile
        let uv_scale: f32 = subdivisions / FACE_RADIUS / 2.0;
        let points: TriPoints = closest_tri_points(uv * uv_scale);

        // Project 2D triangle coordinates onto a sphere
        let a: Vec3 = self.face_to_sphere(points.a / uv_scale);
        let b: Vec3 = self.face_to_sphere(points.b / uv_scale);
        let c: Vec3 = self.face_to_sphere(points.c / uv_scale);
        let center: Vec3 = self.face_to_sphere(points.center / uv_scale);
        let ab: Vec3 = self.face_to_sphere(points.ab / uv_scale);
        let bc: Vec3 = self.face_to_sphere(points.bc / uv_scale);
        let ca: Vec3 = self.face_to_sphere(points.ca / uv_scale);

        TriPoints3D {
            a,
            b,
            c,
            center,
            ab,
            bc,
            ca,
        }
    }
}

// --------------------------------------------------------
// Spectrum colour palette
// IQ https://www.shadertoy.com/view/ll2GD3
// --------------------------------------------------------

fn pal(t: f32, a: Vec3, b: Vec3, c: Vec3, d: Vec3) -> Vec3 {
    a + b * (6.28318 * (c * t + d)).cos()
}

fn spectrum(n: f32) -> Vec3 {
    pal(
        n,
        vec3(0.5, 0.5, 0.5),
        vec3(0.5, 0.5, 0.5),
        vec3(1.0, 1.0, 1.0),
        vec3(0.0, 0.33, 0.67),
    )
}

// --------------------------------------------------------
// Model/Camera Rotation
// --------------------------------------------------------

fn spherical_matrix(theta: f32, phi: f32) -> Mat3 {
    let cx: f32 = theta.cos();
    let cy: f32 = phi.cos();
    let sx: f32 = theta.sin();
    let sy: f32 = phi.sin();
    Mat3::from_cols_array(&[cy, -sy * -sx, -sy * cx, 0.0, cx, sx, sy, cy * -sx, cy * cx])
}

impl State {
    fn mouse_rotation(&self, enable: bool, mut xy: Vec2) -> Mat3 {
        if enable {
            let mouse: Vec2 = self.inputs.mouse.xy() / self.inputs.resolution.xy();

            if mouse.x != 0. && mouse.y != 0. {
                xy.x = mouse.x;
                xy.y = mouse.y;
            }
        }
        let rx: f32;
        let ry: f32;

        rx = (xy.y + 0.5) * PI;
        ry = (-xy.x) * 2.0 * PI;

        spherical_matrix(rx, ry)
    }

    fn model_rotation(&self) -> Mat3 {
        self.mouse_rotation(MOUSE_CONTROL == 1, MODEL_ROTATION)
    }

    fn camera_rotation(&self) -> Mat3 {
        self.mouse_rotation(MOUSE_CONTROL == 2, CAMERA_ROTATION)
    }
}

// --------------------------------------------------------
// Animation
// --------------------------------------------------------

const SCENE_DURATION: f32 = 6.0;
const CROSSFADE_DURATION: f32 = 2.0;

struct HexSpec {
    round_top: f32,
    round_corner: f32,
    height: f32,
    thickness: f32,
    gap: f32,
}

fn new_hex_spec(subdivisions: f32) -> HexSpec {
    HexSpec {
        round_top: 0.05 / subdivisions,
        round_corner: 0.1 / subdivisions,
        height: 2.0,
        thickness: 2.0,
        gap: 0.005,
    }
}

impl State {
    // Animation 1

    fn anim_subdivisions1(&self) -> f32 {
        mix(2.4, 3.4, (self.time * PI).cos() * 0.5 + 0.5)
    }

    fn anim_hex1(&self, hex_center: Vec3, subdivisions: f32) -> HexSpec {
        let mut spec: HexSpec = new_hex_spec(subdivisions);

        let mut offset: f32 = self.time * 3. * PI;
        offset -= subdivisions;
        let mut blend: f32 = hex_center.dot(self.pca);
        blend = (blend * 30. + offset).cos() * 0.5 + 0.5;
        spec.height = mix(1.75, 2., blend);

        spec.thickness = spec.height;

        spec
    }
    // Animation 2

    fn anim_subdivisions2(&self) -> f32 {
        mix(1., 2.3, (self.time * PI / 2.).sin() * 0.5 + 0.5)
    }

    fn anim_hex2(&self, hex_center: Vec3, subdivisions: f32) -> HexSpec {
        let mut spec: HexSpec = new_hex_spec(subdivisions);

        let blend: f32 = hex_center.y;
        spec.height = mix(1.6, 2., (blend * 10. + self.time * PI).sin() * 0.5 + 0.5);

        spec.round_top = 0.02 / subdivisions;
        spec.round_corner = 0.09 / subdivisions;
        spec.thickness = spec.round_top * 4.0;
        spec.gap = 0.01;

        spec
    }

    // Animation 3

    fn anim_subdivisions3(&self) -> f32 {
        5.0
    }

    fn anim_hex3(&self, hex_center: Vec3, subdivisions: f32) -> HexSpec {
        let mut spec: HexSpec = new_hex_spec(subdivisions);

        let mut blend: f32 = hex_center.dot(self.pab).acos() * 10.0;
        blend = (blend + self.time * PI).cos() * 0.5 + 0.5;
        spec.gap = mix(0.01, 0.4, blend) / subdivisions;

        spec.thickness = spec.round_top * 2.;

        spec
    }
}

// Transition between animations

fn sine_in_out(t: f32) -> f32 {
    -0.5 * ((PI * t).cos() - 1.0)
}

impl State {
    fn transition_values(&self, a: f32, b: f32, c: f32) -> f32 {
        if LOOP != 0 {
            if LOOP == 1 {
                return a;
            }
            if LOOP == 2 {
                return b;
            }
            if LOOP == 3 {
                return c;
            }
        }
        let t: f32 = self.time / SCENE_DURATION;
        let scene: f32 = t.rem_euclid(3.0).floor();
        let mut blend: f32 = t.gl_fract();
        let delay: f32 = (SCENE_DURATION - CROSSFADE_DURATION) / SCENE_DURATION;
        blend = (blend - delay).max(0.0) / (1.0 - delay);
        blend = sine_in_out(blend);
        let ab: f32 = mix(a, b, blend);
        let bc: f32 = mix(b, c, blend);
        let cd: f32 = mix(c, a, blend);
        let mut result: f32 = mix(ab, bc, scene.min(1.0));
        result = mix(result, cd, (scene - 1.0).max(0.0));
        result
    }

    fn transition_hex_specs(&self, a: HexSpec, b: HexSpec, c: HexSpec) -> HexSpec {
        let round_top: f32 = self.transition_values(a.round_top, b.round_top, c.round_top);
        let round_corner: f32 =
            self.transition_values(a.round_corner, b.round_corner, c.round_corner);
        let height: f32 = self.transition_values(a.height, b.height, c.height);
        let thickness: f32 = self.transition_values(a.thickness, b.thickness, c.thickness);
        let gap: f32 = self.transition_values(a.gap, b.gap, c.gap);
        HexSpec {
            round_top,
            round_corner,
            height,
            thickness,
            gap,
        }
    }
}

// --------------------------------------------------------
// Modelling
// --------------------------------------------------------

const FACE_COLOR: Vec3 = const_vec3!([0.9, 0.9, 1.0]);
const BACK_COLOR: Vec3 = const_vec3!([0.1, 0.1, 0.15]);
const BACKGROUND_COLOR: Vec3 = const_vec3!([0.0, 0.005, 0.03]);

#[derive(Clone, Copy, Default)]
struct Model {
    dist: f32,
    albedo: Vec3,
    glow: f32,
}

impl State {
    fn hex_model(
        &self,
        p: Vec3,
        hex_center: Vec3,
        edge_a: Vec3,
        edge_b: Vec3,
        spec: HexSpec,
    ) -> Model {
        let mut d: f32;

        let edge_a_dist: f32 = p.dot(edge_a) + spec.gap;
        let edge_b_dist: f32 = p.dot(edge_b) - spec.gap;
        let edge_dist: f32 = smax(edge_a_dist, -edge_b_dist, spec.round_corner);

        let outer_dist: f32 = p.length() - spec.height;
        d = smax(edge_dist, outer_dist, spec.round_top);

        let inner_dist: f32 = p.length() - spec.height + spec.thickness;
        d = smax(d, -inner_dist, spec.round_top);

        let mut color: Vec3;

        let mut face_blend: f32 = (spec.height - p.length()) / spec.thickness;
        face_blend = face_blend.clamp(0.0, 1.0);
        color = mix(FACE_COLOR, BACK_COLOR, 0.5.step(face_blend));

        let edge_color: Vec3 = spectrum(hex_center.dot(self.pca) * 5.0 + p.length() + 0.8);
        let edge_blend: f32 = smoothstep(-0.04, -0.005, edge_dist);
        color = mix(color, edge_color, edge_blend);

        Model {
            dist: d,
            albedo: color,
            glow: edge_blend,
        }
    }
}

// checks to see which intersection is closer
fn op_u(m1: Model, m2: Model) -> Model {
    if m1.dist < m2.dist {
        m1
    } else {
        m2
    }
}

impl State {
    fn geodesic_model(&self, mut p: Vec3) -> Model {
        self.p_mod_icosahedron(&mut p);

        let subdivisions: f32 = self.transition_values(
            self.anim_subdivisions1(),
            self.anim_subdivisions2(),
            self.anim_subdivisions3(),
        );
        let points: TriPoints3D = self.geodesic_tri_points(p, subdivisions);

        let edge_ab: Vec3 = points.center.cross(points.ab).normalize();
        let edge_bc: Vec3 = points.center.cross(points.bc).normalize();
        let edge_ca: Vec3 = points.center.cross(points.ca).normalize();

        let mut model: Model;
        let mut part: Model;
        let mut spec: HexSpec;

        spec = self.transition_hex_specs(
            self.anim_hex1(points.b, subdivisions),
            self.anim_hex2(points.b, subdivisions),
            self.anim_hex3(points.b, subdivisions),
        );
        part = self.hex_model(p, points.b, edge_ab, edge_bc, spec);
        model = part;

        spec = self.transition_hex_specs(
            self.anim_hex1(points.c, subdivisions),
            self.anim_hex2(points.c, subdivisions),
            self.anim_hex3(points.c, subdivisions),
        );
        part = self.hex_model(p, points.c, edge_bc, edge_ca, spec);
        model = op_u(model, part);

        spec = self.transition_hex_specs(
            self.anim_hex1(points.a, subdivisions),
            self.anim_hex2(points.a, subdivisions),
            self.anim_hex3(points.a, subdivisions),
        );
        part = self.hex_model(p, points.a, edge_ca, edge_ab, spec);
        model = op_u(model, part);

        model
    }

    fn map(&self, mut p: Vec3) -> Model {
        let m: Mat3 = self.model_rotation();
        p = m.transpose() * p;
        if LOOP == 0 {
            p_r(&mut p.xz(), self.time * PI / 16.);
        }
        let model: Model = self.geodesic_model(p);
        model
    }
}

// --------------------------------------------------------
// LIGHTING
// Adapted from IQ https://www.shadertoy.com/view/Xds3zN
// --------------------------------------------------------

fn do_lighting(model: Model, _pos: Vec3, nor: Vec3, _ref: Vec3, rd: Vec3) -> Vec3 {
    let light_pos: Vec3 = vec3(0.5, 0.5, -1.0).normalize();
    let back_light_pos: Vec3 = vec3(-0.5, -0.3, 1.0).normalize();
    let ambient_pos: Vec3 = vec3(0.0, 1.0, 0.0);

    let lig: Vec3 = light_pos;
    let amb: f32 = ((nor.dot(ambient_pos) + 1.0) / 2.0).clamp(0.0, 1.0);
    let dif: f32 = nor.dot(lig).clamp(0.0, 1.0);
    let bac: f32 = nor.dot(back_light_pos).clamp(0.0, 1.0).powf(1.5);
    let fre: f32 = (1.0 + nor.dot(rd)).clamp(0.0, 1.0).powf(2.0);

    let mut lin: Vec3 = Vec3::zero();
    lin += 1.20 * dif * Vec3::splat(0.9);
    lin += 0.80 * amb * vec3(0.5, 0.7, 0.8);
    lin += 0.30 * bac * Vec3::splat(0.25);
    lin += 0.20 * fre * Vec3::one();

    let albedo: Vec3 = model.albedo;
    let col: Vec3 = mix(albedo * lin, albedo, model.glow);

    col
}

// --------------------------------------------------------
// Ray Marching
// Adapted from cabbibo https://www.shadertoy.com/view/Xl2XWt
// --------------------------------------------------------

const MAX_TRACE_DISTANCE: f32 = 8.0; // max trace distance
const INTERSECTION_PRECISION: f32 = 0.001; // precision of the intersection
const NUM_OF_TRACE_STEPS: i32 = 100;
const FUDGE_FACTOR: f32 = 0.9; // Default is 1, reduce to fix overshoots

struct CastRay {
    origin: Vec3,
    direction: Vec3,
}

struct Ray {
    origin: Vec3,
    direction: Vec3,
    len: f32,
}

struct Hit {
    ray: Ray,
    model: Model,
    pos: Vec3,
    is_background: bool,
    normal: Vec3,
    color: Vec3,
}

impl State {
    fn calc_normal(&self, pos: Vec3) -> Vec3 {
        let eps: Vec3 = vec3(0.001, 0.0, 0.0);
        let nor: Vec3 = vec3(
            self.map(pos + eps.xyy()).dist - self.map(pos - eps.xyy()).dist,
            self.map(pos + eps.yxy()).dist - self.map(pos - eps.yxy()).dist,
            self.map(pos + eps.yyx()).dist - self.map(pos - eps.yyx()).dist,
        );
        nor.normalize()
    }

    fn raymarch(&self, cast_ray: CastRay) -> Hit {
        let mut current_dist: f32 = INTERSECTION_PRECISION * 2.0;
        let mut model: Model = Model::default();

        let mut ray: Ray = Ray {
            origin: cast_ray.origin,
            direction: cast_ray.direction,
            len: 0.0,
        };

        let mut i = 0;
        while i < NUM_OF_TRACE_STEPS {
            if current_dist < INTERSECTION_PRECISION || ray.len > MAX_TRACE_DISTANCE {
                break;
            }
            model = self.map(ray.origin + ray.direction * ray.len);
            current_dist = model.dist;
            ray.len += current_dist * FUDGE_FACTOR;
            i += 1;
        }

        let mut is_background: bool = false;
        let mut pos: Vec3 = Vec3::zero();
        let mut normal: Vec3 = Vec3::zero();
        let color: Vec3 = Vec3::zero();

        if ray.len > MAX_TRACE_DISTANCE {
            is_background = true;
        } else {
            pos = ray.origin + ray.direction * ray.len;
            normal = self.calc_normal(pos);
        }

        Hit {
            ray,
            model,
            pos,
            is_background,
            normal,
            color,
        }
    }
}

// --------------------------------------------------------
// Rendering
// --------------------------------------------------------

fn shade_surface(hit: &mut Hit) {
    let mut color: Vec3 = BACKGROUND_COLOR;

    if hit.is_background {
        hit.color = color;
        return;
    }

    let _ref: Vec3 = hit.ray.direction.reflect(hit.normal);

    if DEBUG {
        color = hit.normal * 0.5 + Vec3::splat(0.5);
    } else {
        color = do_lighting(hit.model, hit.pos, hit.normal, _ref, hit.ray.direction);
    }
    hit.color = color;
}

fn render(mut hit: Hit) -> Vec3 {
    shade_surface(&mut hit);
    hit.color
}

// --------------------------------------------------------
// Camera
// https://www.shadertoy.com/view/Xl2XWt
// --------------------------------------------------------

fn calc_look_at_matrix(ro: Vec3, ta: Vec3, roll: f32) -> Mat3 {
    let ww: Vec3 = (ta - ro).normalize();
    let uu: Vec3 = ww.cross(vec3(roll.sin(), roll.cos(), 0.0)).normalize();
    let vv: Vec3 = uu.cross(ww).normalize();
    Mat3::from_cols(uu, vv, ww)
}

impl State {
    fn do_camera(
        &self,
        cam_pos: &mut Vec3,
        cam_tar: &mut Vec3,
        cam_roll: &mut f32,
        _time: f32,
        _mouse: Vec2,
    ) {
        let dist: f32 = 5.5;
        *cam_roll = 0.0;
        *cam_tar = vec3(0.0, 0.0, 0.0);
        *cam_pos = vec3(0.0, 0.0, -dist);
        *cam_pos = self.camera_rotation().transpose() * *cam_pos;
        *cam_pos += *cam_tar;
    }
}

// --------------------------------------------------------
// Gamma
// https://www.shadertoy.com/view/Xds3zN
// --------------------------------------------------------

const GAMMA: f32 = 2.2;

fn gamma(color: Vec3, g: f32) -> Vec3 {
    color.powf(g)
}

fn linear_to_screen(linear_rgb: Vec3) -> Vec3 {
    gamma(linear_rgb, 1.0 / GAMMA)
}

impl State {
    pub fn main_image(&mut self, frag_color: &mut Vec4, frag_coord: Vec2) {
        self.time = self.inputs.time;

        if LOOP != 0 {
            if LOOP == 1 {
                self.time = self.time.rem_euclid(2.0);
            }

            if LOOP == 2 {
                self.time = self.time.rem_euclid(4.0);
            }

            if LOOP == 3 {
                self.time = self.time.rem_euclid(2.0);
            }
        }

        self.init_icosahedron();

        let p: Vec2 = (-self.inputs.resolution.xy() + 2.0 * frag_coord) / self.inputs.resolution.y;
        let m: Vec2 = self.inputs.mouse.xy() / self.inputs.resolution.xy();

        let mut cam_pos: Vec3 = vec3(0.0, 0.0, 2.0);
        let mut cam_tar: Vec3 = vec3(0.0, 0.0, 0.0);
        let mut cam_roll: f32 = 0.0;

        // camera movement
        self.do_camera(&mut cam_pos, &mut cam_tar, &mut cam_roll, self.time, m);

        // camera matrix
        let cam_mat: Mat3 = calc_look_at_matrix(cam_pos, cam_tar, cam_roll); // 0.0 is the camera roll

        // create view ray
        let rd: Vec3 = (cam_mat * p.extend(2.0)).normalize(); // 2.0 is the lens length

        let hit: Hit = self.raymarch(CastRay {
            origin: cam_pos,
            direction: rd,
        });

        let mut color: Vec3 = render(hit);

        if !DEBUG {
            color = linear_to_screen(color);
        }

        *frag_color = color.extend(1.0);
    }
}
