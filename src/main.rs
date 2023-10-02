use glam::DVec3;
use itertools::Itertools;
use indicatif::ProgressIterator;
use std::{fs, io, ops::Range};

const ASPECT_RATIO: f64 = 16.0 / 9.0;
const IMAGE_WIDTH: i32 = 400;

struct Camera {
    image_width: i32,
    image_height: u32,
    max_value: u8,
    aspect_ratio: f64,
    center: DVec3,
    pixel_delta_u: DVec3,
    pixel_delta_v: DVec3,
    viewport_upper_left: DVec3,
    pixel00_loc: DVec3,
}

impl Camera {

    fn new(image_width: i32, aspect_ratio: f64) -> Self {

        let image_height: u32 = (image_width as f64 / aspect_ratio) as u32;
        let max_value: u8 = 255;
        let center = DVec3::new(0., 0., 0.);

        let viewport_height: f64 = 2.0;
        let viewport_width: f64 = viewport_height as f64 * image_width as f64 / image_height as f64;


        // calculate the vectors across the horizontal and down the vertical viewport edges.
        let viewport_u: DVec3 = DVec3::new(viewport_width, 0., 0.);
        let viewport_v: DVec3 = DVec3::new(0., -viewport_height, 0.);

        let focal_length = 0.3;

        let pixel_delta_u: DVec3 = viewport_u / image_width as f64;
        let pixel_delta_v: DVec3 = viewport_v / image_height as f64;
        let viewport_upper_left: DVec3 = center - DVec3::new(0.,-2., focal_length) - viewport_u / 2. + viewport_v / 2.;
        let pixel00_loc: DVec3 = viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v);

        
        return Self {
            image_width,
            image_height,
            max_value,
            aspect_ratio,
            center,
            pixel_delta_u,
            pixel_delta_v,
            viewport_upper_left,
            pixel00_loc,
        }

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
                let pixel_center: DVec3 = self.pixel00_loc + (x as f64 * self.pixel_delta_u) + (y as f64 * self.pixel_delta_v);

                let ray_direction: DVec3 = pixel_center - self.center;

                let ray = Ray {
                    origin: self.center,
                    direction: ray_direction,
                };

                let pixel_color: DVec3 = ray.color(&world) * 255.0;

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
{} {}
{}
{}",
            self.image_width, self.image_height, self.max_value, pixels
        ));
    }
}




fn main() -> io::Result<()> {
    let mut world = HittableList { objects: vec![] };

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

    let camera = Camera::new(IMAGE_WIDTH, ASPECT_RATIO);
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
    
  fn color<T>(&self, world: &T) -> DVec3 
  where
    T: Hittable, {

        if let Some(rec) = world.hit(&self, (0.)..f64::INFINITY) {
            return 0.5 * (rec.normal + DVec3::new(1., 1., 1.))
        }

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

        // find the nearest root taht lies in the acceptable range
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

        let rec = HitRecord::with_face_normal(
            point,
            outward_normal,
            t,
            true,
        );
        return Some(rec)
    }
}