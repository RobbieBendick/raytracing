use glam::DVec3;
use itertools::Itertools;
use indicatif::ProgressIterator;
use std::{fs, io, ops::Range};
use rand::Rng;

const ASPECT_RATIO: f64 = 16.0 / 9.0;
const IMAGE_WIDTH: i16 = 400;

struct Camera {
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
}

impl Camera {

    fn new(image_width: i16, aspect_ratio: f64) -> Self {

        let image_height: i16 = (image_width as f64 / aspect_ratio) as i16;
        let max_value: i16 = 255;
        let center = DVec3::new(0., 0., 0.);

        let viewport_height: f64 = 2.0;
        let viewport_width: f64 = viewport_height as f64 * image_width as f64 / image_height as f64;

        // calculate the vectors across the horizontal and down the vertical viewport edges.
        let viewport_u: DVec3 = DVec3::new(viewport_width, 0., 0.);
        let viewport_v: DVec3 = DVec3::new(0., -viewport_height, 0.);

        let focal_length = 0.3;

        // calculate the pixel deltas
        let pixel_delta_u: DVec3 = viewport_u / image_width as f64;
        let pixel_delta_v: DVec3 = viewport_v / image_height as f64;

        let viewport_upper_left: DVec3 = center - DVec3::new(0.,-2., focal_length) - viewport_u / 2. + viewport_v / 2.;
        let pixel_00_loc: DVec3 = viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v);
        
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
            samples_per_pixel: 100,
            max_depth: 50,
        }

    }
    fn get_ray(&self, x: i16, y: i16) -> Ray {

        // get a randomly sampled camera ray for the pixel location at x,y
        let pixel_center = self.pixel_00_loc + (x as f64 * self.pixel_delta_u) + (y as f64 * self.pixel_delta_v);
        let pixel_sample = pixel_center + Self::pixel_sample_square(self);

        let ray_origin = self.center;
        let ray_direction = pixel_sample - ray_origin;

        return Ray {
            origin: ray_origin,
            direction: ray_direction,
        }
    }

    fn generate_random_number() -> f64 {
        let mut rng = rand::thread_rng();
        return rng.gen_range(0.0..1.0);
    }
    
    fn pixel_sample_square(&self) -> DVec3 {
        // returns a random point in the square surrounding a pixel at the origin.
        // this is used to sample the pixel for anti-aliasing.
        let px = -0.5 + Self::generate_random_number();
        let py = -0.5 + Self::generate_random_number();
        return (px * self.pixel_delta_u) + (py * self.pixel_delta_v);
    }

    fn render_to_disk<T>(&self, world: T) where T: Hittable{ 

        let pixels: String = (0..self.image_height)

            // get x, y coordinates of every pixel
            .cartesian_product(0..self.image_width)

            // progress bar
            .progress_count(
                self.image_height as u64 * self.image_width as u64,
            )

            // map over every x and y coordinate on every pixel
            .map(|(y, x)| {

                // get a fraction of a pixel. in this case, it's 1/10th of a pixel
                // we'll combine the colors of all the samples to get the final color of the pixel
                let scale_factor = (self.samples_per_pixel as f64).recip();

                // get the sum of all the colors of the samples for the pixel
                let multisampled_pixel_color = (0..self.samples_per_pixel)
                .into_iter()
                .map(|_| {
                    self.get_ray(x, y)
                        .color(self.max_depth, &world)
                })
                .sum::<DVec3>() * scale_factor;

                let color = DVec3 {
                    x: linear_to_gamma(multisampled_pixel_color.x),
                    y: linear_to_gamma(multisampled_pixel_color.y),
                    z: linear_to_gamma(multisampled_pixel_color.z),
                }

                // lighten the color

                .clamp(
                    DVec3::splat(0.),
                    DVec3::splat(0.999),
                ) * 256.;

                format!(
                    "{} {} {}",
                    color.x,
                    color.y,
                    color.z
                )
            })
            .join("\n");
        
        // write the image to a file
        fs::write("output.ppm", format!(
            "P3
{} {}
{}
{}",
            self.image_width, self.image_height, self.max_value, pixels
        ));
    }
}


fn linear_to_gamma(scalar:f64) -> f64 {
    return scalar.sqrt();
}

fn main() -> io::Result<()> {
    let mut world: HittableList = HittableList { objects: vec![] };

    world.add(
        Sphere {
            center: DVec3::new(0., 0., -1.),
            radius: 0.82,
        }
    );

    world.add(
        Sphere {
            center: DVec3::new(0., -100.5, -1.),
            radius: 100.,
        }
    );

    // initialize the camera
    let camera = Camera::new(IMAGE_WIDTH, ASPECT_RATIO);

    // render the image to disk
    camera.render_to_disk(world);

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
    
  fn color<T>(&self, depth: i32, world: &T) -> DVec3 
  where
    T: Hittable, {

        // if we've exceeded the ray bounce limit, no more light is gathered
        if depth <= 0 {
            return DVec3::ZERO;
        }

        // if the ray hits the sphere, return the sphere's color
        if let Some(rec) = world.hit(&self, (0.)..f64::INFINITY) {
            let direction = rec.normal + random_unit_vector();
            let ray = Ray {
                origin: rec.point,
                direction: direction,
            };
            return 0.4 * ray.color(depth - 1, world);   
        }

        // if the ray does not hit the sphere, return the background color
        let unit_direction = self.direction.normalize();
        let a = 0.5 * (unit_direction.y + 1.0);
        let background_color = (1.0 - a) * DVec3::new(1., 1., 1.) + a * DVec3::new(0.5, 0.7, 1.0);
        return background_color; 

    }
}


struct HitRecord {
    point: DVec3, normal: DVec3, t: f64, front_face: bool,
}

impl HitRecord {
    fn with_face_normal(
        point: DVec3,
        normal: DVec3,
        t: f64,
        front_face: bool,
    ) -> Self {
        let normal = if front_face { normal } else { -normal };

        // return the hit record
        Self {
            point,
            normal,
            t,
            front_face,
        }
    }
}


struct HittableList {
    objects: Vec<Box<dyn Hittable>>,
}

impl HittableList {
    fn clear(&mut self) {
        self.objects = vec![];
    }

    fn add<T>(&mut self, object: T) where T: Hittable + 'static, {
        self.objects.push(Box::new(object));
    }
}

trait Hittable {
    fn hit(
        &self,
        ray: &Ray,
        interval: Range<f64>,
        // ray_tmin: f64,
        // ray_tmax: f64,
        // record: HitRecord,
    ) -> Option<HitRecord>;
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


struct Sphere {
    center: DVec3,
    radius: f64,
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
        let point = ray.at(t);
        let outward_normal = (point - self.center) / self.radius;

        // return the hit record
        let rec = HitRecord::with_face_normal(
            point,
            outward_normal,
            t,
            true,
        );
        return Some(rec)
    }
}


fn random_in_unit_sphere() -> DVec3 {
    let mut rng = rand::thread_rng();
    loop {
        let vec = DVec3::new(
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

fn random_on_hemisphere(normal: &DVec3) -> DVec3 {
    let on_unit_sphere = random_unit_vector();

    // in the same hesiphere as the normal
    if on_unit_sphere.dot(*normal) > 0.0 {
        return on_unit_sphere;
    } else {
        return -on_unit_sphere;
    }
}