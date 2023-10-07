use crate::ray::Ray;
use glam::DVec3;
use rand::Rng;
use std::{ops::{Range, Neg}, f64::consts::PI};

pub trait Hittable {
    fn hit(&self, ray: &Ray, interval: Range<f64>) -> Option<HitRecord>;
}

#[derive(Clone)]
pub struct HitRecord {
    pub point: DVec3,
    pub normal: DVec3,
    pub t: f64,
    pub front_face: bool,
    pub material: Material,
    pub u: f64,
    pub v: f64,
}

impl HitRecord {
    pub fn with_face_normal(
        material: Material,
        point: DVec3,
        normal: DVec3,
        t: f64,
        front_face: bool,
        u: f64,
        v: f64,
    ) -> Self {
        let normal: DVec3 = if front_face { normal } else { -normal };

        // return the hit record
        Self {
            material,
            point,
            normal,
            t,
            front_face,
            u,
            v
        }
    }

    pub fn calc_face_normal(
        ray: &Ray,
        outward_normal: &DVec3,
    ) -> (bool, DVec3) {
        let front_face =
            ray.direction.dot(*outward_normal) < 0.;
        let normal = if front_face {
            *outward_normal
        } else {
            -*outward_normal
        };
        return (front_face, normal);
    }
}

#[non_exhaustive]
#[derive(Clone)]
pub enum Material {
    Lambertian {
        albedo: DVec3,
    },
    Metal {
        albedo: DVec3,
        fuzz: f64,
    },
    Dielectric {
        index_of_refraction: f64,
    }
}

pub struct Scattered {
    pub attenuation: DVec3,
    pub scattered: Ray,
}

impl Material {
    pub fn scatter(
        &self,
        r_in: &Ray,
        hit_record: HitRecord,
    ) -> Option<Scattered> {
        match self {
            Material::Lambertian { albedo } => {
                let mut scatter_direction = hit_record
                    .normal
                    + random_unit_vector();

                // Catch degenerate scatter direction
                if scatter_direction
                    .abs_diff_eq(DVec3::ZERO, 1e-8)
                {
                    scatter_direction = hit_record.normal;
                }

                Some(Scattered {
                    attenuation: *albedo,
                    scattered: Ray {
                        origin: hit_record.point,
                        direction: scatter_direction,
                    },
                })
            }

            Material::Metal { albedo, fuzz } => {
                let reflected: DVec3 = reflect(
                    r_in.direction.normalize(),
                    hit_record.normal,
                );
                let scattered = Ray {
                    origin: hit_record.point,
                    direction: reflected
                        + *fuzz * random_unit_vector(),
                };
                // absorb any scatter that is below the surface
                if scattered
                    .direction
                    .dot(hit_record.normal)
                    > 0.
                {
                    Some(Scattered {
                        attenuation: *albedo,
                        scattered,
                    })
                } else {
                    None
                }
            }

            Material::Dielectric {
                index_of_refraction,
            } => {
                let mut rng = rand::thread_rng();

                let attenuation = DVec3::splat(1.0);
                let refraction_ratio: f64 =
                    if hit_record.front_face {
                        index_of_refraction.recip()
                    } else {
                        *index_of_refraction
                    };

                let unit_direction =
                    r_in.direction.normalize();

                let cos_theta = unit_direction
                    .dot(hit_record.normal)
                    .neg()
                    .min(1.0);
                let sin_theta =
                    (1.0 - cos_theta * cos_theta).sqrt();

                let cannot_refract =
                    refraction_ratio * sin_theta > 1.0;

                    let direction = if cannot_refract || reflectance(cos_theta, refraction_ratio) > rng.gen::<f64>() {
                        // Reflection
                        reflect(unit_direction, hit_record.normal)
                    } else {
                        // Refraction
                        refract(unit_direction, hit_record.normal, refraction_ratio)
                    };

                    Some(Scattered {
                        attenuation,
                        scattered: Ray {
                            origin: hit_record.point,
                            direction: direction,
                        },
                    })
            }
            _ => None,
        }
    }
}



pub struct HittableList {
    pub objects: Vec<Box<dyn Hittable + Sync + Send>>,
}

impl HittableList {
    pub fn clear(&mut self) {
        self.objects = vec![];
    }

    pub fn add<T>(&mut self, object: T) where T: Hittable + 'static + Sync + Send, {
        self.objects.push(Box::new(object));
    }
}

impl Hittable for HittableList {
    fn hit(
        &self, ray: &Ray, interval: Range<f64>,
    ) -> Option<HitRecord> {
        // iterate through every object in the list and find the closest hit
        let (_closest, hit_record) = self
        .objects.iter()
        .fold(
            (interval.end, None),
            |acc, item|{
            if let Some(temp_rec) = item.hit(ray, interval.start..acc.0) {
                return (temp_rec.t, Some(temp_rec));
            } else {
                return acc;
            }
        });

        return hit_record;
    }
}

pub struct Sphere {
    pub center: DVec3,
    pub radius: f64,
    pub material: Material,
}

impl Sphere {
    pub fn new(center: DVec3, radius: f64, material: Material) -> Self {
        return Self {
            center,
            radius,
            material,
        };
    }
    pub fn get_sphere_uv(&self, p: DVec3) -> (f64, f64) {
        // p: a given point on the sphere of radius one, centered at the origin.
        // u: returned value [0,1] of angle around the Y axis from X=-1.
        // v: returned value [0,1] of angle from Y=-1 to Y=+1.
        //     <1 0 0> yields <0.50 0.50>       <-1  0  0> yields <0.00 0.50>
        //     <0 1 0> yields <0.50 1.00>       < 0 -1  0> yields <0.50 0.00>
        //     <0 0 1> yields <0.25 0.50>       < 0  0 -1> yields <0.75 0.50>
    
        let theta = (-p.y).acos();
        let phi = (-p.z).atan2(p.x) + PI;
    
        let u = phi / (2. * PI);
        let v = theta / PI;
        return (u, v);
    }
}



impl Hittable for Sphere {
    fn hit(
        &self,
        ray: &Ray,
        interval: Range<f64>,
        
    ) -> Option<HitRecord> {
        let oc = ray.origin - self.center;
        let a = ray.direction.length_squared();
        let half_b = oc.dot(ray.direction);
        let c = oc.length_squared() - self.radius * self.radius;
        let discriminant = half_b * half_b - a * c;

        if discriminant < 0. {
            return None;
        }
        let sqrtd = discriminant.sqrt();

        // find the nearest root that lies in the acceptable range
        let mut root = (-half_b - sqrtd) / a;
        if !interval.contains(&root) {
            root = (-half_b + sqrtd) / a;
            if !interval.contains(&root) {
                return None;
            }
        }

        let t = root;
        let point: DVec3 = ray.at(t);
        let outward_normal: DVec3 = (point - self.center) / self.radius;

        let (u, v) = self.get_sphere_uv(outward_normal);

        // return the hit record
        let rec = HitRecord::with_face_normal(
            self.material.clone(),
            point,
            outward_normal,
            t,
            true,
            u,
            v,
        );
        return Some(rec);
    }
}

fn reflect(v: DVec3, n: DVec3) -> DVec3 {
    return v - 2. * v.dot(n) * n;
}

fn refract(
    uv: DVec3,
    n: DVec3,
    etai_over_etat: f64,
) -> DVec3 {
    let cos_theta = uv.neg().dot(n).min(1.0);
    let r_out_perp = etai_over_etat * (uv + cos_theta * n);
    let r_out_parallel: DVec3 = (1.0
        - r_out_perp.length_squared())
    .abs()
    .sqrt()
    .neg()
        * n;
    return r_out_perp + r_out_parallel;
}

fn reflectance(cosine: f64, ref_idx: f64) -> f64 {
    // Use Schlick's approximation for reflectance.
    let mut r0 = (1. - ref_idx) / (1. + ref_idx);
    r0 = r0 * r0;
    return r0 + (1. - r0) * (1. - cosine).powf(5.);
}

fn random_in_unit_sphere() -> DVec3 {
    let mut rng = rand::thread_rng();
    loop {
        let vec: DVec3 = DVec3::new(
            rng.gen_range(-1.0..1.0),
            rng.gen_range(-1.0..1.0),
            rng.gen_range(-1.0..1.0),
        );
        if vec.length_squared() >= 1.0 {
            break vec;
        }
    }
}

fn random_unit_vector() -> DVec3 {
    return random_in_unit_sphere().normalize();
}
