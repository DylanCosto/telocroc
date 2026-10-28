//! `telocroc`: allele-specific telomere length from long reads.

mod align;
mod alleles;
mod anchor;
mod cli;
mod input;
mod kmers;
mod linkage;
#[cfg(target_arch = "aarch64")]
mod neon;
mod numpy;
mod pipeline;
mod pyrandom;
mod resources;
mod screen;
mod telrep;
mod tvr;
mod util;
mod wavelet;

#[cfg(test)]
mod fixture_tests;

use clap::Parser;

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

fn main() {
    let argv: Vec<String> = std::env::args().collect();
    let args = cli::Args::parse_from(cli::normalize_args(argv.clone()));
    if let Err(e) = rayon::ThreadPoolBuilder::new().num_threads(args.threads.max(1)).build_global() {
        eprintln!("Error: could not start {} threads: {e}", args.threads);
        std::process::exit(1);
    }
    #[cfg(feature = "profile")]
    let profiler = profiling::start();
    let result = pipeline::run(&args, &argv);
    #[cfg(feature = "profile")]
    profiling::finish(profiler);
    if let Err(e) = result {
        eprintln!("Error: {e:#}");
        std::process::exit(1);
    }
}

#[cfg(feature = "profile")]
mod profiling {
    use std::collections::{HashMap, HashSet};
    use std::io::Write;

    pub fn start() -> Option<(String, pprof::ProfilerGuard<'static>)> {
        let prefix = std::env::var("TELOCROC_PROFILE").ok()?;
        Some((prefix, pprof::ProfilerGuardBuilder::default().frequency(997).build().expect("start profiler")))
    }

    pub fn finish(profiler: Option<(String, pprof::ProfilerGuard<'static>)>) {
        let Some((prefix, guard)) = profiler else { return };
        let report = guard.report().build().expect("build profile");
        let svg = std::fs::File::create(format!("{prefix}.svg")).expect("create flame graph");
        report.flamegraph(svg).expect("write flame graph");
        // inclusive and self sample counts per function
        let (mut inclusive, mut own) = (HashMap::<String, isize>::new(), HashMap::<String, isize>::new());
        let mut total = 0;
        for (frames, &count) in &report.data {
            total += count;
            let names: Vec<String> =
                frames.frames.iter().map(|f| f.first().map(|s| s.name()).unwrap_or_default()).collect();
            if let Some(leaf) = names.first() {
                *own.entry(leaf.clone()).or_default() += count;
            }
            let mut seen = HashSet::new();
            for n in names {
                if seen.insert(n.clone()) {
                    *inclusive.entry(n).or_default() += count;
                }
            }
        }
        let mut out = std::fs::File::create(format!("{prefix}.txt")).expect("create profile report");
        for (title, map) in [("inclusive", &inclusive), ("self", &own)] {
            let mut v: Vec<_> = map.iter().collect();
            v.sort_by(|a, b| b.1.cmp(a.1));
            writeln!(out, "== {title} ({total} samples)").expect("write profile report");
            for (name, count) in v.into_iter().take(60) {
                writeln!(out, "{:6.2}%  {name}", 100.0 * *count as f64 / total as f64).expect("write profile report");
            }
        }
    }
}
