use std::io;
use raytracing::create_world;

fn main() -> io::Result<()> {
    create_world();

    Ok(())
}