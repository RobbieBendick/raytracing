use glam::DVec3;
use crate::hittable::{Hittable, Scattered};

pub struct Ray {
    pub origin: DVec3,
    pub direction: DVec3,
}

impl Ray {
    pub fn at(&self, t: f64) -> DVec3 {
        return self.origin + t * self.direction;
    }

    pub fn color<T>(&self, depth: i32, world: &T) -> DVec3 
    where
        T: Hittable, {

        // if we've exceeded the ray bounce limit, no more light is gathered
        if depth <= 0 {
            return DVec3::ZERO;
        }
        if let Some(Scattered {
            attenuation,
            scattered,
        }) = world.hit(&self, (0.001)..f64::INFINITY).and_then(|rec| rec.material.scatter(self, rec.clone()))
        {
            // determine the color of the scattered ray
            let color_from_scatter = attenuation * scattered.color(depth - 1, world);
            return color_from_scatter;
        }
    
        // if the ray does not hit a sphere, return the background color
        let unit_direction: DVec3 = self.direction.normalize();
        let a = 0.5 * (unit_direction.y + 1.0);
        let background_color: DVec3 = (1.0 - a) * DVec3::new(1., 1., 1.) + a * DVec3::new(0.5, 0.7, 1.0);
        return background_color;
    }
}