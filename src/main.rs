use glam::DVec3;
use hittable::{HittableList, Sphere, Material, Scattered};
use std::{io, thread};
use rand::Rng;
mod ray;
mod hittable;
mod camera;
use crate::camera::Camera;


const ASPECT_RATIO: f64 = 16.0 / 9.0;
const IMAGE_WIDTH: i16 = 1200;

fn random_color() -> DVec3 {
    return DVec3::new(
        rand::thread_rng().gen::<f64>(),
        rand::thread_rng().gen::<f64>(),
        rand::thread_rng().gen::<f64>(),
    );
}


fn create_spheres(from: i8, to: i8, radius: f64, a: i8) {
    thread::spawn(move || {
        let mut world_two: HittableList = HittableList { objects: vec![] };
        
        for e in from..to {
            let choose_mat = rand::thread_rng().gen::<f64>();
            let center = DVec3::new(    
                a as f64 + 0.9 * rand::thread_rng().gen::<f64>(),
                0.2,
                e as f64 + 0.9 * rand::thread_rng().gen::<f64>(),
            );

            if (center - DVec3::new(4.0, 0.2, 0.0)).length() > 0.9 {
                if choose_mat < 0.8 {
                    // Lambertian
                    let albedo = random_color();
                    world_two.add(Sphere {
                        center,
                        radius,
                        material: Material::Lambertian { albedo },
                    });
                } else if choose_mat < 0.95 {
                    // metal
                    let albedo = random_color();
                    world_two.add(Sphere {
                        center,
                        radius,
                        material: Material::Metal {albedo: albedo, fuzz: 0.08},
                    });
                } else {
                    // glass
                    world_two.add(Sphere {
                        center,
                        radius,
                        material: Material::Dielectric {index_of_refraction: 0.},
                    });
                }
            }
        }
    });
}

fn main() -> io::Result<()> {
    let mut world: HittableList = HittableList { objects: vec![] };
    

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

     
    let radius = rand::thread_rng().gen_range(0.0..0.5);

    for a in -11..11 {
        create_spheres(-11, -9, radius.clone(), a.clone());
        create_spheres(-9, -6, radius.clone(), a.clone());
        create_spheres(-6, -3, radius.clone(), a.clone());
        create_spheres(-3, 0, radius.clone(), a.clone());
        create_spheres(0, 3, radius.clone(), a.clone());
        create_spheres(3, 6, radius.clone(), a.clone());
        create_spheres(6, 9, radius.clone(), a.clone());

        for c in 9..11 {  
            let choose_mat = rand::thread_rng().gen::<f64>();
            let center = DVec3::new(    
                a as f64 + 0.9 * rand::thread_rng().gen::<f64>(),
                0.2,
                c as f64 + 0.9 * rand::thread_rng().gen::<f64>(),
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
    camera.render_to_disk(world);

    Ok(())
}