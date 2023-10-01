use glam::DVec3;
use itertools::Itertools;
use indicatif::ProgressIterator;
use std::{fs, io};

// image
const ASPECT_RATIO: f64 = 16.0 / 9.0;
const IMAGE_WIDTH: i32 = 700;
const IMAGE_HEIGHT: u32 = (IMAGE_WIDTH as f64 / ASPECT_RATIO) as u32;
const MAX_VALUE: i32 = 255;


// camera
const CAMERA_CENTER : DVec3 = DVec3::new(0., 0., 0.);
const FOCAL_LENGTH: f64 = 0.3;
const VIEWPORT_HEIGHT: f64 = 2.0;
const VIEWPORT_WIDTH: f64 = VIEWPORT_HEIGHT as f64 * IMAGE_WIDTH as f64 / IMAGE_HEIGHT as f64;


// calculate the vectors across the horizontal and down the vertical viewport edges.
const VIEWPORT_U: DVec3 = DVec3::new(VIEWPORT_WIDTH, 0., 0.);
const VIEWPORT_V: DVec3 = DVec3::new(0., -VIEWPORT_HEIGHT, 0.);

fn main() -> io::Result<()> {
    println!("Image size: {}x{}", IMAGE_WIDTH, IMAGE_HEIGHT);
    // calculate the horizontal and vertical delta vectors from pixel to pixel.
    let pixel_delta_u: DVec3 = VIEWPORT_U / IMAGE_WIDTH as f64;
    let pixel_delta_v: DVec3 = VIEWPORT_V / IMAGE_HEIGHT as f64;

    // calculate the location of the upper left pixel.
    let viewport_upper_left: DVec3 = CAMERA_CENTER - DVec3::new(0.,-2., FOCAL_LENGTH) - VIEWPORT_U / 2. + VIEWPORT_V / 2.;
    let pixel00_loc: DVec3 = viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v);


    let pixels: String = (0..IMAGE_HEIGHT)

        // get x, y coordinates of every pixel
        .cartesian_product(0..IMAGE_WIDTH)

        // progress bar
        .progress_count(
            IMAGE_HEIGHT as u64 * IMAGE_WIDTH as u64,
        )

        // map over every x and y coordinate on every pixel
        .map(|(y, x)| {
            let pixel_center: DVec3 = pixel00_loc + (x as f64 * pixel_delta_u) + (y as f64 * pixel_delta_v);

            let ray_direction: DVec3 = pixel_center - CAMERA_CENTER;

            let ray = Ray {
                origin: CAMERA_CENTER,
                direction: ray_direction,
            };

            let pixel_color: DVec3 = ray.color() * 255.0;

            format!(
                "{} {} {}",
                pixel_color.x,
                pixel_color.y,
                pixel_color.z
            )
        })
        .join("\n");

    // write the image to a file
    fs::write("output.ppm", format!(
        "P3
{IMAGE_WIDTH} {IMAGE_HEIGHT}
{MAX_VALUE}
{pixels}
"
        ),
    )?;
    Ok(())
}

struct Ray {
    origin: DVec3,
    direction: DVec3,
}

impl Ray {
    fn at(&self, t: f64) -> DVec3 {
        return self.origin + t * self.direction;
    }
    
  fn color(&self) -> DVec3 {
        // Check if the ray hits the sphere
        let t = Self::hit_sphere(&DVec3::new(0., 0., -1.2), 0.05, &self);
        if t > 0. {
            let N = (self.at(t) - DVec3::new(0., 0., -1.)).normalize();
            return 0.5 * (N + 1.0);
        }
        // Create a gradient background
        let unit_direction = self.direction.normalize();
        let t = 0.5 * (unit_direction.y + 1.0);

        // Background gradient color
        let background_color = (1.0 - t) * DVec3::new(1., 1., 1.) + t * DVec3::new(0.5, 0.7, 1.0);

        return background_color;
    }

    fn hit_sphere(center: &DVec3, radius: f64, ray: &Ray) -> f64 {
        let oc = ray.origin - *center;
        let a = ray.direction.length_squared();
        let half_b = 2.0 * oc.dot(ray.direction);
        let c = oc.length_squared() - radius * radius;

        // check if the ray hits the sphere
        let discriminant = half_b * half_b - a * c;
        if discriminant < 0. {
            return -1.0;
        }

        return (-half_b - discriminant.sqrt()) / a
    }
    
    
}