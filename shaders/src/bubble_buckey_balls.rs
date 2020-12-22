//! Ported to Rust from <https://www.shadertoy.com/view/lslSRf>
//!
//! Original comment:
//! ```glsl
//! // mplanck
//! // Tested on 13-inch Powerbook
//! // Tested on Late 2013 iMac
//! // Tested on Nvidia GTX 780 Windows 7
//! ```

use crate::SampleCube;
use shared::*;
use spirv_std::glam::{vec2, vec3, Vec2, Vec3, Vec3Swizzles, Vec4, Vec4Swizzles};

// Note: This cfg is incorrect on its surface, it really should be "are we compiling with std", but
// we tie #[no_std] above to the same condition, so it's fine.
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

pub struct Inputs<C0, C1> {
    pub resolution: Vec3,
    pub time: f32,
    pub mouse: Vec4,
    pub channel0: C0,
    pub channel1: C1,
}

pub struct State<C0, C1> {
    inputs: Inputs<C0, C1>,

    cam_point_at: Vec3,
    cam_origin: Vec3,
    time: f32,
    ldir: Vec3,
}

impl<C0, C1> State<C0, C1> {
    pub fn new(inputs: Inputs<C0, C1>) -> Self {
        State {
            inputs,

            cam_point_at: Vec3::zero(),
            cam_origin: Vec3::zero(),
            time: 0.0,
            ldir: vec3(0.8, 1.0, 0.0),
        }
    }
}

// **************************************************************************
// CONSTANTS

const PI: f32 = 3.14159;
const TWO_PI: f32 = 6.28318;
const _PI_OVER_TWO: f32 = 1.570796;
const ONE_OVER_PI: f32 = 0.318310;
const GR: f32 = 1.61803398;

const SMALL_FLOAT: f32 = 0.0001;
const BIG_FLOAT: f32 = 1000000.0;

// **************************************************************************
// MATERIAL DEFINES

const SPHERE_MATL: f32 = 1.0;
const CHAMBER_MATL: f32 = 2.0;
const BOND_MATL: f32 = 3.0;

// **************************************************************************
// UTILITIES

// Rotate the input point around the y-axis by the angle given as a
// cos(angle) and sin(angle) argument.  There are many times where  I want to
// reuse the same angle on different points, so why do the heavy trig twice.
// Range of outputs := ([-1.,-1.,-1.] -> [1.,1.,1.])

fn rotate_around_y_axis(point: Vec3, cosangle: f32, sinangle: f32) -> Vec3 {
    return vec3(
        point.x * cosangle + point.z * sinangle,
        point.y,
        point.x * -sinangle + point.z * cosangle,
    );
}

// Rotate the input point around the x-axis by the angle given as a
// cos(angle) and sin(angle) argument.  There are many times where  I want to
// reuse the same angle on different points, so why do the  heavy trig twice.
// Range of outputs := ([-1.,-1.,-1.] -> [1.,1.,1.])

fn rotate_around_x_axis(point: Vec3, cosangle: f32, sinangle: f32) -> Vec3 {
    vec3(
        point.x,
        point.y * cosangle - point.z * sinangle,
        point.y * sinangle + point.z * cosangle,
    )
}

fn pow5(v: f32) -> f32 {
    let tmp: f32 = v * v;
    tmp * tmp * v
}

// convert a 3d point to two polar coordinates.
// First coordinate is elevation angle (angle from the plane going through x+z)
// Second coordinate is azimuth (rotation around the y axis)
// Range of outputs - ([PI/2, -PI/2], [-PI, PI])
fn _cartesian_to_polar(p: Vec3) -> Vec2 {
    vec2(PI / 2. - (p.y / p.length()).acos(), p.z.atan2(p.x))
}

fn mergeobjs(a: Vec2, b: Vec2) -> Vec2 {
    if a.x < b.x {
        return a;
    } else {
        return b;
    }

    // XXX: Some architectures have bad optimization paths
    // that will cause inappropriate branching if you DON'T
    // use an if statement here.

    //return mix(b, a, step(a.x, b.x));
}

// **************************************************************************
// DISTANCE FIELDS

fn spheredf(pos: Vec3, r: f32) -> f32 {
    pos.length() - r
}

fn segmentdf(p: Vec3, a: Vec3, b: Vec3, r: f32) -> f32 {
    let ba: Vec3 = b - a;
    let mut t: f32 = ba.dot(p - a) / SMALL_FLOAT.max(ba.dot(ba));
    t = t.clamp(0., 1.);
    return (ba * t + a - p).length() - r;
}

// **************************************************************************
// SCENE MARCHING

fn buckeyballsobj(p: Vec3, mr: f32) -> Vec2 {
    let mut ballsobj: Vec2 = vec2(BIG_FLOAT, SPHERE_MATL);
    let ap: Vec3 = p.abs();
    //let ap: Vec3 = p;

    // vertices
    // fully positive hexagon
    let p1: Vec3 = vec3(0.66, 0.33 + 0.66 * GR, 0.33 * GR);
    let p2: Vec3 = vec3(0.33, 0.66 + 0.33 * GR, 0.66 * GR);
    let p3: Vec3 = vec3(0.33 * GR, 0.66, 0.33 + 0.66 * GR);
    let p4: Vec3 = vec3(0.66 * GR, 0.33, 0.66 + 0.33 * GR);
    let p5: Vec3 = vec3(0.33 + 0.66 * GR, 0.33 * GR, 0.66);
    let p6: Vec3 = vec3(0.66 + 0.33 * GR, 0.66 * GR, 0.33);

    // fully positive connectors
    let p7: Vec3 = vec3(0.33, GR, 0.0);
    let p8: Vec3 = vec3(GR, 0.0, 0.33);
    let p9: Vec3 = vec3(0.0, 0.33, GR);

    ballsobj.x = ballsobj.x.min(spheredf(ap - p1, mr));
    ballsobj.x = ballsobj.x.min(spheredf(ap - p2, mr));
    ballsobj.x = ballsobj.x.min(spheredf(ap - p3, mr));
    ballsobj.x = ballsobj.x.min(spheredf(ap - p4, mr));
    ballsobj.x = ballsobj.x.min(spheredf(ap - p5, mr));
    ballsobj.x = ballsobj.x.min(spheredf(ap - p6, mr));
    ballsobj.x = ballsobj.x.min(spheredf(ap - p7, mr));
    ballsobj.x = ballsobj.x.min(spheredf(ap - p8, mr));
    ballsobj.x = ballsobj.x.min(spheredf(ap - p9, mr));

    let mut bondsobj: Vec2 = vec2(BIG_FLOAT, BOND_MATL);

    let br: f32 = 0.2 * mr;
    bondsobj.x = bondsobj.x.min(segmentdf(ap, p1, p2, br));
    bondsobj.x = bondsobj.x.min(segmentdf(ap, p2, p3, br));
    bondsobj.x = bondsobj.x.min(segmentdf(ap, p3, p4, br));
    bondsobj.x = bondsobj.x.min(segmentdf(ap, p4, p5, br));
    bondsobj.x = bondsobj.x.min(segmentdf(ap, p5, p6, br));
    bondsobj.x = bondsobj.x.min(segmentdf(ap, p6, p1, br));

    bondsobj.x = bondsobj.x.min(segmentdf(ap, p1, p7, br));
    bondsobj.x = bondsobj.x.min(segmentdf(ap, p5, p8, br));
    bondsobj.x = bondsobj.x.min(segmentdf(ap, p3, p9, br));

    // bond neighbors
    let p10: Vec3 = vec3(-0.33, 0.66 + 0.33 * GR, 0.66 * GR);

    let p11: Vec3 = vec3(0.66 * GR, -0.33, 0.66 + 0.33 * GR);

    let p12: Vec3 = vec3(0.66 + 0.33 * GR, 0.66 * GR, -0.33);

    let p13: Vec3 = vec3(-0.33, GR, 0.0);
    let p14: Vec3 = vec3(0.66, 0.33 + 0.66 * GR, -0.33 * GR);

    let p15: Vec3 = vec3(GR, 0.0, -0.33);
    let p16: Vec3 = vec3(0.33 + 0.66 * GR, -0.33 * GR, 0.66);

    let p17: Vec3 = vec3(0.0, -0.33, GR);
    let p18: Vec3 = vec3(-0.33 * GR, 0.66, 0.33 + 0.66 * GR);

    bondsobj.x = bondsobj.x.min(segmentdf(ap, p2, p10, br));
    bondsobj.x = bondsobj.x.min(segmentdf(ap, p4, p11, br));
    bondsobj.x = bondsobj.x.min(segmentdf(ap, p6, p12, br));

    bondsobj.x = bondsobj.x.min(segmentdf(ap, p7, p13, br));
    bondsobj.x = bondsobj.x.min(segmentdf(ap, p7, p14, br));

    bondsobj.x = bondsobj.x.min(segmentdf(ap, p8, p15, br));
    bondsobj.x = bondsobj.x.min(segmentdf(ap, p8, p16, br));

    bondsobj.x = bondsobj.x.min(segmentdf(ap, p9, p17, br));
    bondsobj.x = bondsobj.x.min(segmentdf(ap, p9, p18, br));

    mergeobjs(ballsobj, bondsobj)
}

fn chamberobj(p: Vec3) -> Vec2 {
    vec2(20.0 - p.length(), CHAMBER_MATL)
}

impl<C0, C1> State<C0, C1> {
    fn scenedf(&self, p: Vec3) -> Vec2 {
        //let mp: Vec3 = p;
        //let bbi: f32 = 0.0;

        let mut mp: Vec3 = p + Vec3::splat(3.0);
        let bbi: f32 = Vec3::one().dot((mp / 6.).floor());
        let mr: f32 = 0.4 * (0.7 + 0.5 * (2.0 * self.time - 1.0 * p.y + 6281.0 * bbi).sin());

        mp = mp.rem_euclid(6.0) - Vec3::splat(3.0);

        let obj: Vec2 = buckeyballsobj(mp, mr);

        mergeobjs(chamberobj(p), obj)
    }
}

const DISTMARCH_STEPS: usize = 60;
const DISTMARCH_MAXDIST: f32 = 50.0;

impl<C0, C1> State<C0, C1> {
    fn distmarch(&self, ro: Vec3, rd: Vec3, maxd: f32) -> Vec2 {
        let epsilon: f32 = 0.001;
        let mut dist: f32 = 10.0 * epsilon;
        let mut t: f32 = 0.0;
        let mut material: f32 = 0.0;
        let mut i = 0;
        while i < DISTMARCH_STEPS {
            if dist.abs() < epsilon || t > maxd {
                break;
            }
            // advance the distance of the last lookup
            t += dist;
            let dfresult: Vec2 = self.scenedf(ro + t * rd);
            dist = dfresult.x;
            material = dfresult.y;
            i += 1;
        }

        if t > maxd {
            material = -1.0;
        }
        return vec2(t, material);
    }
}

// **************************************************************************
// SHADOWING & NORMALS

const SOFTSHADOW_STEPS: usize = 40;
const SOFTSHADOW_STEPSIZE: f32 = 0.1;

impl<C0, C1> State<C0, C1> {
    fn calc_soft_shadow(&self, ro: Vec3, rd: Vec3, mint: f32, maxt: f32, k: f32) -> f32 {
        let mut shadow: f32 = 1.0;
        let mut t: f32 = mint;

        let mut i = 0;
        while i < SOFTSHADOW_STEPS {
            if t < maxt {
                let h: f32 = self.scenedf(ro + rd * t).x;
                shadow = shadow.min(k * h / t);
                t += SOFTSHADOW_STEPSIZE;
            }
            i += 1;
        }
        shadow.clamp(0.0, 1.0)
    }
}

const AO_NUMSAMPLES: usize = 6;
const AO_STEPSIZE: f32 = 0.1;
const AO_STEPSCALE: f32 = 0.4;

impl<C0, C1> State<C0, C1> {
    fn calc_ao(&self, p: Vec3, n: Vec3) -> f32 {
        let mut ao: f32 = 0.0;
        let mut aoscale: f32 = 1.0;

        let mut aoi = 0;
        while aoi < AO_NUMSAMPLES {
            let step: f32 = 0.01 + AO_STEPSIZE * aoi as f32;
            let aop: Vec3 = n * step + p;

            let d: f32 = self.scenedf(aop).x;
            ao += -(d - step) * aoscale;
            aoscale *= AO_STEPSCALE;
            aoi += 1;
        }

        ao.clamp(0.0, 1.0)
    }

    // **************************************************************************
    // CAMERA & GLOBALS

    fn animate_globals(&mut self) {
        // remap the mouse click ([-1, 1], [-1/ar, 1/ar])
        let mut click: Vec2 = self.inputs.mouse.xy() / self.inputs.resolution.xx();
        click = 2.0 * click - Vec2::one();

        self.time = 0.8 * self.inputs.time - 10.0;

        // camera position
        self.cam_origin = vec3(4.5, 0.0, 4.5);

        let rotx: f32 = -1. * PI * (0.5 * click.y + 0.45) + 0.05 * self.time;
        let cosrotx: f32 = rotx.cos();
        let sinrotx: f32 = rotx.sin();

        let roty: f32 = TWO_PI * click.x + 0.05 * self.time;
        let cosroty: f32 = roty.cos();
        let sinroty: f32 = roty.sin();

        // Rotate the camera around the origin
        self.cam_origin = rotate_around_x_axis(self.cam_origin, cosrotx, sinrotx);
        self.cam_origin = rotate_around_y_axis(self.cam_origin, cosroty, sinroty);

        self.cam_point_at = Vec3::zero();

        let lroty: f32 = 0.9 * self.time;
        let coslroty: f32 = lroty.cos();
        let sinlroty: f32 = lroty.sin();

        // Rotate the light around the origin
        self.ldir = rotate_around_y_axis(self.ldir, coslroty, sinlroty);
    }
}

struct CameraData {
    origin: Vec3,
    dir: Vec3,
    _st: Vec2,
}

impl<C0, C1> State<C0, C1> {
    fn setup_camera(&self, frag_coord: Vec2) -> CameraData {
        // aspect ratio
        let invar: f32 = self.inputs.resolution.y / self.inputs.resolution.x;
        let mut st: Vec2 = frag_coord / self.inputs.resolution.xy() - Vec2::splat(0.5);
        st.y *= invar;

        // calculate the ray origin and ray direction that represents
        // mapping the image plane towards the scene
        let iu: Vec3 = vec3(0., 1., 0.);

        let iz: Vec3 = (self.cam_point_at - self.cam_origin).normalize();
        let ix: Vec3 = (iz.cross(iu)).normalize();
        let iy: Vec3 = ix.cross(iz);

        let dir: Vec3 = (st.x * ix + st.y * iy + 0.7 * iz).normalize();

        CameraData {
            origin: self.cam_origin,
            dir,
            _st: st,
        }
    }
}
// **************************************************************************
// SHADING

#[derive(Clone, Copy)]
struct SurfaceData {
    point: Vec3,
    normal: Vec3,
    basecolor: Vec3,
    roughness: f32,
    metallic: f32,
}

impl SurfaceData {
    fn init_surf(p: Vec3, n: Vec3) -> Self {
        SurfaceData {
            point: p,
            normal: n,
            basecolor: Vec3::zero(),
            roughness: 0.0,
            metallic: 0.0,
        }
    }
}

impl<C0, C1> State<C0, C1> {
    fn calc_normal(&self, p: Vec3) -> Vec3 {
        let epsilon: Vec3 = vec3(0.001, 0.0, 0.0);
        let n: Vec3 = vec3(
            self.scenedf(p + epsilon.xyy()).x - self.scenedf(p - epsilon.xyy()).x,
            self.scenedf(p + epsilon.yxy()).x - self.scenedf(p - epsilon.yxy()).x,
            self.scenedf(p + epsilon.yyx()).x - self.scenedf(p - epsilon.yyx()).x,
        );
        n.normalize()
    }
}
fn material(surfid: f32, surf: &mut SurfaceData) {
    let _surfcol: Vec3 = Vec3::one();
    if surfid - 0.5 < SPHERE_MATL {
        surf.basecolor = vec3(0.8, 0.2, 0.5);
        surf.roughness = 0.5;
        surf.metallic = 0.8;
    } else if surfid - 0.5 < CHAMBER_MATL {
        surf.basecolor = Vec3::zero();
        surf.roughness = 1.0;
    } else if surfid - 0.5 < BOND_MATL {
        surf.basecolor = vec3(0.02, 0.02, 0.05);
        surf.roughness = 0.2;
        surf.metallic = 0.0;
    }
}

impl<C0: SampleCube, C1: SampleCube> State<C0, C1> {
    fn integrate_dir_light(&self, ldir: Vec3, lcolor: Vec3, surf: SurfaceData) -> Vec3 {
        let vdir: Vec3 = (self.cam_origin - surf.point).normalize();

        // The half vector of a microfacet model
        let hdir: Vec3 = (ldir + vdir).normalize();

        // cos(theta_h) - theta_h is angle between half vector and normal
        let costh: f32 = -SMALL_FLOAT.max(surf.normal.dot(hdir));
        // cos(theta_d) - theta_d is angle between half vector and light dir/view dir
        let costd: f32 = -SMALL_FLOAT.max(ldir.dot(hdir));
        // cos(theta_l) - theta_l is angle between the light vector and normal
        let costl: f32 = -SMALL_FLOAT.max(surf.normal.dot(ldir));
        // cos(theta_v) - theta_v is angle between the viewing vector and normal
        let costv: f32 = -SMALL_FLOAT.max(surf.normal.dot(vdir));

        let ndl: f32 = costl.clamp(0.0, 1.);

        let mut cout: Vec3 = Vec3::zero();

        if ndl > 0. {
            let frk: f32 = 0.5 + 2.0 * costd * costd * surf.roughness;
            let diff: Vec3 = surf.basecolor
                * ONE_OVER_PI
                * (1. + (frk - 1.) * pow5(1. - costl))
                * (1. + (frk - 1.) * pow5(1. - costv));
            //let diff: Vec3 = surf.basecolor * ONE_OVER_PI; // lambert

            // D(h) factor
            // using the GGX approximation where the gamma factor is 2.

            // Clamping roughness so that a directional light has a specular
            // response.  A roughness of perfectly 0 will create light
            // singularities.
            let r: f32 = surf.roughness.max(0.05);
            let alpha: f32 = r * r;
            let denom: f32 = costh * costh * (alpha * alpha - 1.) + 1.;
            let d: f32 = (alpha * alpha) / (PI * denom * denom);

            // using the GTR approximation where the gamma factor is generalized
            // let alpha: f32 = surf.roughness * surf.roughness;
            // let gamma: f32 = 2.0;
            // let sinth: f32 = length(cross(surf.normal, hdir));
            // let D: f32 = 1.0/(alpha*alpha*costh*costh + sinth*sinth).powf(gamma);

            // G(h,l,v) factor
            let k: f32 = ((r + 1.) * (r + 1.)) / 8.;
            let gl: f32 = costv / (costv * (1. - k) + k);
            let gv: f32 = costl / (costl * (1. - k) + k);
            let g: f32 = gl * gv;

            // F(h,l) factor
            let f0: Vec3 = mix(Vec3::splat(0.5), surf.basecolor, surf.metallic);
            let f: Vec3 = f0 + (Vec3::one() - f0) * pow5(1.0 - costd);

            let spec: Vec3 = d * f * g / (4.0 * costl * costv);

            let shd: f32 = self.calc_soft_shadow(surf.point, ldir, 0.1, 20.0, 5.0);

            cout += diff * ndl * shd * lcolor;
            cout += spec * ndl * shd * lcolor;
        }

        cout
    }

    fn sample_env_light(&self, ldir: Vec3, lcolor: Vec3, surf: SurfaceData) -> Vec3 {
        let vdir: Vec3 = (self.cam_origin - surf.point).normalize();

        // The half vector of a microfacet model
        let hdir: Vec3 = (ldir + vdir).normalize();

        // cos(theta_h) - theta_h is angle between half vector and normal
        let costh: f32 = surf.normal.dot(hdir);
        // cos(theta_d) - theta_d is angle between half vector and light dir/view dir
        let costd: f32 = ldir.dot(hdir);
        // cos(theta_l) - theta_l is angle between the light vector and normal
        let costl: f32 = surf.normal.dot(ldir);
        // cos(theta_v) - theta_v is angle between the viewing vector and normal
        let costv: f32 = surf.normal.dot(vdir);

        let ndl: f32 = costl.clamp(0.0, 1.0);
        let mut cout: Vec3 = Vec3::zero();
        if ndl > 0. {
            let r: f32 = surf.roughness;
            // G(h,l,v) factor
            let k: f32 = r * r / 2.;
            let gl: f32 = costv / (costv * (1. - k) + k);
            let gv: f32 = costl / (costl * (1. - k) + k);
            let g: f32 = gl * gv;

            // F(h,l) factor
            let f0: Vec3 = mix(Vec3::splat(0.5), surf.basecolor, surf.metallic);
            let f: Vec3 = f0 + (Vec3::one() - f0) * pow5(1. - costd);

            // Combines the BRDF as well as the pdf of this particular
            // sample direction.
            let spec: Vec3 = lcolor * g * f * costd / (costh * costv);

            let shd: f32 = self.calc_soft_shadow(surf.point, ldir, 0.02, 20.0, 7.0);

            cout = spec * shd * lcolor;
        }

        cout
    }

    fn integrate_env_light(&self, surf: SurfaceData) -> Vec3 {
        let vdir: Vec3 = (surf.point - self.cam_origin).normalize();
        let envdir: Vec3 = vdir.reflect(surf.normal);
        let specolor: Vec4 = Vec4::splat(0.4)
            * mix(
                self.inputs.channel0.sample_cube(envdir),
                self.inputs.channel1.sample_cube(envdir),
                surf.roughness,
            );

        self.sample_env_light(envdir, specolor.xyz(), surf)
    }

    fn shade_surface(&self, surf: SurfaceData) -> Vec3 {
        let amb: Vec3 = surf.basecolor * 0.04;
        // ambient occlusion is amount of occlusion.  So 1 is fully occluded
        // and 0 is not occluded at all.  Makes math easier when mixing
        // shadowing effects.
        let ao: f32 = self.calc_ao(surf.point, surf.normal);

        let centerldir: Vec3 = (-surf.point).normalize();

        let mut cout: Vec3 = Vec3::zero();
        if surf.basecolor.dot(Vec3::one()) > SMALL_FLOAT {
            cout += self.integrate_dir_light(self.ldir, Vec3::splat(0.3), surf);
            cout += self.integrate_dir_light(centerldir, vec3(0.3, 0.5, 1.0), surf);
            cout += self.integrate_env_light(surf) * (1.0 - 3.5 * ao);
            cout += amb * (1.0 - 5.5 * ao);
        }
        cout
    }

    // **************************************************************************
    // MAIN

    pub fn main_image(&mut self, frag_color: &mut Vec4, frag_coord: Vec2) {
        // ----------------------------------------------------------------------
        // Animate globals

        self.animate_globals();

        // ----------------------------------------------------------------------
        // Setup Camera

        let cam: CameraData = self.setup_camera(frag_coord);

        // ----------------------------------------------------------------------
        // SCENE MARCHING

        let scenemarch: Vec2 = self.distmarch(cam.origin, cam.dir, DISTMARCH_MAXDIST);

        // ----------------------------------------------------------------------
        // SHADING

        let mut scenecol: Vec3 = Vec3::zero();
        if scenemarch.y > SMALL_FLOAT {
            let mp: Vec3 = cam.origin + scenemarch.x * cam.dir;
            let mn: Vec3 = self.calc_normal(mp);

            let mut curr_surf: SurfaceData = SurfaceData::init_surf(mp, mn);

            material(scenemarch.y, &mut curr_surf);
            scenecol = self.shade_surface(curr_surf);
        }

        // ----------------------------------------------------------------------
        // POST PROCESSING

        // fall off exponentially into the distance (as if there is a spot light
        // on the point of interest).
        scenecol *= (-0.01 * (scenemarch.x * scenemarch.x - 300.0)).exp();

        // brighten
        scenecol *= 1.3;

        // distance fog
        scenecol = mix(
            scenecol,
            0.02 * vec3(1.0, 0.2, 0.8),
            smoothstep(10.0, 30.0, scenemarch.x),
        );

        // Gamma correct
        scenecol = scenecol.powf_vec(Vec3::splat(0.45));

        // Contrast adjust - cute trick learned from iq
        scenecol = mix(
            scenecol,
            Vec3::splat(scenecol.dot(Vec3::splat(0.333))),
            -0.6,
        );

        // color tint
        scenecol = 0.5 * scenecol + 0.5 * scenecol * vec3(1.0, 1.0, 0.9);

        *frag_color = scenecol.extend(1.0);
    }
}
