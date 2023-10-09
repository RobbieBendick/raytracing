
pub mod camera;
pub mod ray;
pub mod hittable;

use std::thread;

use glam::DVec3;
use hittable::{HittableList, Sphere, Material};
use rand::Rng;
use crate::camera::Camera;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro128PlusPlus;

const ASPECT_RATIO: f64 = 16.0 / 9.0;
const IMAGE_WIDTH: i16 = 600;


fn random_color() -> DVec3 {
    let mut rng = Xoshiro128PlusPlus::from_entropy();

    return DVec3::new(
        rng.gen::<f64>(),
        rng.gen::<f64>(),
        rng.gen::<f64>(),
    );
}


fn create_spheres(sphere_count: i32 ) -> HittableList {
    let mut rng: Xoshiro128PlusPlus = Xoshiro128PlusPlus::from_entropy();
    let mut sphere_list: HittableList = HittableList { objects: vec![] };
    let radius = 0.35;
    for _ in 0..sphere_count {
        let choose_mat = rng.gen::<f64>();
        let center = DVec3::new(    
            rng.gen_range(-4.5..4.5) as f64 * rng.gen::<f64>(),
            radius,
            rng.gen_range(-4.5..4.5) as f64 * rng.gen::<f64>(),
        );

        if (center - DVec3::new(4.0, 0.2, 0.0)).length() > 0.9 {
            if choose_mat < 0.8 {
                // Lambertian
                let albedo = random_color();
                sphere_list.add(Sphere {
                    center,
                    radius,
                    material: Material::Lambertian { albedo },
                });
            } else if choose_mat < 0.95 {
                // metal
                let albedo = random_color();
                sphere_list.add(Sphere {
                    center,
                    radius,
                    material: Material::Metal {albedo: albedo, fuzz: 0.08},
                });
            } else {
                // glass
                sphere_list.add(Sphere {
                    center,
                    radius,
                    material: Material::Dielectric {index_of_refraction: 0.},
                });
            }
        }
    }
    return sphere_list;
}

pub fn create_world() {
    let mut world: HittableList = HittableList { objects: vec![] };

    // ground
    world.add(Sphere {
        center: DVec3::new(0., -1000., 0.),
        radius: 1000.,
        material: Material::Lambertian {
            albedo: DVec3::new(0.5, 0.5, 0.5),
        },
    });

    let num_cores = num_cpus::get();
    let num_spheres = 100;  // Adjust as needed
    let spheres_per_thread = num_spheres / num_cores;
    let mut leftovers = num_spheres % num_cores;

    let mut handles = vec![];

    for _ in 0..num_cores {
        let num_spheres_local = spheres_per_thread + if leftovers > 0 { leftovers -= 1; 1 } else { 0 };

        let handle = thread::spawn(move || create_spheres(num_spheres_local as i32));
        handles.push(handle);
    }

    for handle in handles {
        world.add(handle.join().unwrap());
    }

    // initialize the camera
    let camera = Camera::new(IMAGE_WIDTH, ASPECT_RATIO);

    // render the image to disk
    let _ = camera.render_to_disk(world);
}


