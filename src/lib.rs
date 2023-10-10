
pub mod camera;
pub mod ray;
pub mod hittable;

use glam::DVec3;
use hittable::{HittableList, Sphere, Material};
use rand::Rng;
use crate::camera::Camera;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro128PlusPlus;

const ASPECT_RATIO: f64 = 16.0 / 9.0;
const IMAGE_WIDTH: i16 = 1200;


fn random_color() -> DVec3 {
    let mut rng = Xoshiro128PlusPlus::from_entropy();

    return DVec3::new(
        rng.gen::<f64>(),
        rng.gen::<f64>(),
        rng.gen::<f64>(),
    );
}

pub fn create_world() {
    let mut world: HittableList = HittableList { objects: vec![] };
    let mut rng: Xoshiro128PlusPlus = Xoshiro128PlusPlus::from_entropy();
    
    // ground
    world.add(
        Sphere {
            center: DVec3::new(0., -1000., 0.),
            radius: 1000.,
            material: Material::Lambertian {
                albedo: DVec3::new(0.5, 0.5, 0.5),
            },
        }
    );
     
    let radius = 0.35;

    for a in -5..5 {
        for b in -5..5 {
            let choose_mat = rng.gen::<f64>();
            let center = DVec3::new(    
                a as f64 + 0.9 * rng.gen::<f64>(),
                radius,
                b as f64 + 0.9 * rng.gen::<f64>(),
            );

            if (center - DVec3::new(4.0, 0.2, 0.0)).length() > 0.9 {
                if choose_mat < 0.8 {
                    // Lambertian
                    let albedo = random_color();
                    world.add(Sphere {
                        center,
                        radius,
                        material: Material::Lambertian { albedo },
                    });
                } else if choose_mat < 0.95 {
                    // metal
                    let albedo = random_color();
                    world.add(Sphere {
                        center,
                        radius,
                        material: Material::Metal {albedo: albedo, fuzz: 0.08},
                    });
                } else {
                    // glass
                    world.add(Sphere {
                        center,
                        radius,
                        material: Material::Dielectric {index_of_refraction: 0.},
                    });
                }
            }
        }
    }
    // initialize the camera
    let camera = Camera::new(IMAGE_WIDTH, ASPECT_RATIO);

    // render the image to disk
    let _ = camera.render_to_disk(world);
}


