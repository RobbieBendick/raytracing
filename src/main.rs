use glam::DVec3;
use itertools::Itertools;
use indicatif::ProgressIterator;
use std::{fs, io};

const IMAGE_HEIGHT: i32 = 2;
const IMAGE_WIDTH: i32 = 3;
const MAX_VALUE: i32 = 255;


// Determine viewport dimensions.
const CAMERA_CENTER : DVec3 = DVec3::new(0., 0., 0.);
const FOCAL_LENGTH: f64 = 1.0;
const VIEWPORT_HEIGHT: f64 = 2.0;
const VIEWPORT_WIDTH: f64 = VIEWPORT_HEIGHT as f64 * IMAGE_WIDTH as f64 / IMAGE_HEIGHT as f64;
const VIEWPORT_U: DVec3 = DVec3::new(VIEWPORT_WIDTH, 0., 0.);
const VIEWPORT_V: DVec3 = DVec3::new(0., -VIEWPORT_HEIGHT, 0.);

fn main() -> io::Result<()> {

    // Calculate the horizontal and vertical vectors
    let pixel_delta_u: DVec3 = VIEWPORT_U / IMAGE_WIDTH as f64;
    let pixel_delta_v: DVec3 = VIEWPORT_V / IMAGE_HEIGHT as f64;


    // Calculate the location of the upper left pixel
    let viewport_upper_left: DVec3 = CAMERA_CENTER - DVec3::new(0.,0., FOCAL_LENGTH) - VIEWPORT_U / 2. + VIEWPORT_V / 2.;

    let pixel00_loc: DVec3 = viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v);


    let pixels = (0..IMAGE_HEIGHT)

        // get x, y coordinates of every pixel
        .cartesian_product(0..IMAGE_WIDTH)

        // progress bar
        .progress_count(
            IMAGE_HEIGHT as u64 * IMAGE_WIDTH as u64,
        )

        // map over every x and y coordinate on every pixel
        .map(|(y, x)| {
            let pixel_center = pixel00_loc + (x as f64 * pixel_delta_u) + (y as f64 * pixel_delta_v);

            let ray_direction = pixel_center - CAMERA_CENTER;

            let ray = Ray {
                origin: CAMERA_CENTER,
                direction: ray_direction,
            };

            let pixel_color = ray.color() * 255.0;

            format!(
                "{} {} {}",
                pixel_color.x,
                pixel_color.y,
                pixel_color.z
            )
        })
        .join("\n");
    
    println!("{}", pixels);
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
        self.origin + t * self.direction
    }
    fn color(&self) -> DVec3 {
        let t = Self::hit_sphere(&DVec3::new(0.,0.,-1.), 0.5, self);
        if t > 0. {
            let N = (self.at(t) - DVec3::new(0.,0.,-1.)).normalize();
            return 0.5 * (N + 1.0)
        }

        let unit_direction = self.direction.normalize();
        let a = 0.5 * (unit_direction.y + 1.0);
        return (1.0 - a) * DVec3::new(1.0,1.0,1.0) + a * DVec3::new(0.5, 0.7, 1.0);
    }
    

    fn hit_sphere(center: &DVec3, radius: f64, ray: &Ray) -> f64 {
        let oc = ray.origin - *center;
        let a = ray.direction.length_squared();
        let half_b = 2.0 * oc.dot(ray.direction);
        let c =  oc.length_squared() - radius * radius;
        let discriminant = half_b * half_b - a * c;

        if discriminant < 0. {
            return -1.0;
        } else {
            return ( -half_b - discriminant.sqrt() ) / a;
        }
    }
}