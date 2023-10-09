
use crate::ray::Ray;
use crate::hittable::Hittable;
use glam::DVec3;
use itertools::Itertools;
use rayon::prelude::IntoParallelIterator;
use std::{fs, io, f64::consts::PI};
use rand::Rng;
use rand_xoshiro::Xoshiro128PlusPlus;
use rand::SeedableRng;
use rayon::prelude::*;

pub struct Camera {
    image_width: i16,
    image_height: i16,
    max_value: i16,
    aspect_ratio: f64,
    center: DVec3,
    pixel_delta_u: DVec3,
    pixel_delta_v: DVec3,
    viewport_upper_left: DVec3,
    pixel_00_loc: DVec3,
    samples_per_pixel: i32,
    max_depth: i32,
    vfov: f64,
    look_from: DVec3,
    look_at: DVec3,
    v_up: DVec3,
    w: DVec3,
    u: DVec3,
    v: DVec3,
    defocus_angle: f64,
    focus_dist: f64,
    defocus_disk_u: DVec3,
    defocus_disk_v: DVec3,
}

impl Camera {

    pub fn new(image_width: i16, aspect_ratio: f64) -> Self {
        let vfov = 20.0;
        let look_from = DVec3::new(13., 2., 3.);
        let look_at = DVec3::ZERO;
        let v_up = DVec3::new(0., 1., 0.);

        let focus_dist = 10.0;

        let max_value = 255;
        let image_height = (image_width as f64 / aspect_ratio) as i16;


        let theta = degrees_to_radians(vfov);
        let h = (theta / 2.0).tan();
        let viewport_height = 2. * h as f64 * focus_dist as f64;

        let w = unit_vector(look_from - look_at);
        let u = unit_vector(cross(v_up, w));
        let v = cross(w, u);

        let viewport_width = viewport_height as f64 * (image_width as f64 / image_height as f64);

        let center = look_from;

        // calculate the vectors across the horizontal and down the vertical viewport edges.
        let viewport_u = viewport_width * u;
        let viewport_v = viewport_height * -v;

        // calculate the pixel deltas
        let pixel_delta_u = viewport_u / image_width as f64;
        let pixel_delta_v = viewport_v / image_height as f64;

        let viewport_upper_left = center - (focus_dist * w) - viewport_u / 2. - viewport_v / 2.;
        let pixel_00_loc = viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v);

        let defocus_angle = 0.6;
        let defocus_radius = focus_dist * (degrees_to_radians(defocus_angle / 2.)).tan();
        let defocus_disk_u = u * defocus_radius;
        let defocus_disk_v = v * defocus_radius;
        // return the camera's properties
        return Self {
            image_width,
            image_height,
            max_value,
            aspect_ratio,
            center,
            pixel_delta_u,
            pixel_delta_v,
            viewport_upper_left,
            pixel_00_loc: pixel_00_loc,
            samples_per_pixel: 50,
            max_depth: 50,
            vfov,
            look_from,
            look_at,
            v_up,
            w,
            u,
            v,
            defocus_angle,
            focus_dist,
            defocus_disk_u,
            defocus_disk_v,
        }

    }
    pub fn get_ray(&self, x: i16, y: i16) -> Ray {

        // get a randomly sampled camera ray for the pixel location at x,y
        let pixel_center = self.pixel_00_loc
            + (x as f64 * self.pixel_delta_u)
            + (y as f64 * self.pixel_delta_v);
        let pixel_sample =
            pixel_center + self.pixel_sample_square();

            let ray_origin = if self.defocus_angle <= 0. {
                self.center
            } else {
                defocus_disk_sample(self.center, self.defocus_disk_u, self.defocus_disk_v)
            };

            let ray_direction = pixel_sample - ray_origin;

        return Ray {
            origin: ray_origin,
            direction: ray_direction,
        }
    }

    fn generate_random_number() -> f64 {
        let mut rng = Xoshiro128PlusPlus::from_entropy();
        return rng.gen_range(0.0..1.0);
    }
    
    fn pixel_sample_square(&self) -> DVec3 {
        // returns a random point in the square surrounding a pixel at the origin.
        // this is used to sample the pixel for anti-aliasing.
        let px = -0.5 + Self::generate_random_number();
        let py = -0.5 + Self::generate_random_number();
        return (px * self.pixel_delta_u) + (py * self.pixel_delta_v);
    }

    pub fn render_to_disk<T>(&self, world: T) -> io::Result<()>
    where
        T: Hittable + std::marker::Sync,
    {
        let pixels: String = (0..self.image_height)
            .cartesian_product(0..self.image_width)
            .collect::<Vec<(i16, i16)>>()

            // run in parallel (multi-threaded)
            .into_par_iter()

            .map(|(y, x)| {
                let scale_factor = (self.samples_per_pixel as f64).recip();
    
                let multisampled_pixel_color: DVec3 = (0..self.samples_per_pixel)
                    .into_par_iter() // Use into_par_iter instead of into_iter
                    .map(|_| self.get_ray(x, y).color(self.max_depth, &world))
                    .sum::<DVec3>()
                    * scale_factor;
    
                let color = DVec3 {
                    x: linear_to_gamma(multisampled_pixel_color.x),
                    y: linear_to_gamma(multisampled_pixel_color.y),
                    z: linear_to_gamma(multisampled_pixel_color.z),
                }

                // lighten the collor
                .clamp(DVec3::splat(0.), DVec3::splat(0.999))
                * 256.;
    
                format!("{} {} {}", color.x, color.y, color.z)
            })
            .collect::<Vec<String>>()
            .join("\n");
        
        // write the image to file
        let _ = fs::write(
            "output.ppm",
            format!(
                "P3
{} {}
{}
{}",
                self.image_width, self.image_height, self.max_value, pixels
            ),
        );
        Ok(())
    }
}

fn degrees_to_radians(degrees: f64) -> f64 {
    return degrees * PI / 180.;
}

fn unit_vector(v: DVec3) -> DVec3 {
    return v / v.length();
}

fn cross(u: DVec3, v: DVec3) -> DVec3 {
    return DVec3 {
        x: u.y * v.z - u.z * v.y,
        y: u.z * v.x - u.x * v.z,
        z: u.x * v.y - u.y * v.x,
    };
}

fn linear_to_gamma(linear_component: f64) -> f64 {
    return linear_component.sqrt();
}

fn defocus_disk_sample(center: DVec3, defocus_disk_u: DVec3, defocus_disk_v: DVec3) -> DVec3 {
    let p = random_in_unit_disk();
    return center + (p.x * defocus_disk_u) + (p.y * defocus_disk_v);
}

fn random_in_unit_disk() -> DVec3 {
    let mut rng = Xoshiro128PlusPlus::from_entropy();

    loop {
        let p = DVec3::new(rng.gen_range(-1.0..1.0), rng.gen_range(-1.0..1.0), 0.0);

        if p.length_squared() < 1.0 {
            return p;
        }
    }
}