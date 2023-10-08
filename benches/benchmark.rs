
use criterion::{criterion_group, criterion_main, Criterion};
use raytracing::create_world;


// Benchmark function
fn bench_create_world(c: &mut Criterion) {
    // Set up the benchmark with a range of input values
    c.bench_function("create_world", |b| b.iter(|| create_world()));
}

// Entry point for the benchmarking application
criterion_group!(
    name = benches;
    config = Criterion::default().sample_size(10);
    targets = bench_create_world
);
criterion_main!(benches);