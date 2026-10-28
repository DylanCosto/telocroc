//! Locating Telogator2's bundled data (kmer table, subtelomere reference) and other
//! environment lookups.

use std::path::{Path, PathBuf};

use anyhow::{Result, bail};

use crate::cli::ReadType;
use crate::kmers::KmerTable;

/// The default `kmers.tsv`, compiled in.
const DEFAULT_KMERS: &str = include_str!("../resources/kmers.tsv");

/// Directories searched for bundled resources, in order: `$TELOCROC_RESOURCES`,
/// `resources/` next to the executable, and this repository's `resources/`.
fn resource_dirs() -> Vec<PathBuf> {
    let mut dirs = Vec::new();
    if let Some(d) = std::env::var_os("TELOCROC_RESOURCES") {
        dirs.push(PathBuf::from(d));
    }
    if let Some(exe_dir) = std::env::current_exe().ok().and_then(|e| e.parent().map(Path::to_path_buf)) {
        dirs.push(exe_dir.join("resources"));
        dirs.push(exe_dir.join("../share/telocroc/resources"));
    }
    dirs.push(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("resources"));
    dirs
}

/// Resolves a path as given, or relative to a resource directory (the way Telogator2
/// looks names like `resources/non-human/kmers_maize.tsv` up in its package).
fn find(name: &Path) -> Option<PathBuf> {
    if name.exists() {
        return Some(name.to_path_buf());
    }
    let stripped = name.strip_prefix("resources").unwrap_or(name);
    resource_dirs().into_iter().map(|d| d.join(stripped)).find(|p| p.exists())
}

/// The subtelomere reference (`-t`), plus the default winnowmap k15 file when the
/// default reference is used.
///
/// # Errors
///
/// Fails if the reference cannot be found.
pub fn telogator_ref(given: Option<&Path>) -> Result<(PathBuf, Option<PathBuf>)> {
    if let Some(name) = given {
        let Some(found) = find(name) else {
            bail!("provided subtel reference (-t) could not be found: {}", name.display());
        };
        println!("using user-specified subtel reference: {}", name.file_name().unwrap_or_default().to_string_lossy());
        println!(" - found at {}", found.display());
        return Ok((found, None));
    }
    println!("using default subtel reference");
    let Some(reference) = find(Path::new("telogator-ref.fa.gz")) else {
        bail!(
            "default subtel reference telogator-ref.fa.gz not found; run scripts/fetch-resources.sh, \
             set TELOCROC_RESOURCES, or pass -t"
        );
    };
    Ok((reference, find(Path::new("telogator-ref-k15.txt"))))
}

/// The kmer table (`-k`, or the bundled default).
///
/// # Errors
///
/// Fails if the file cannot be found or parsed.
pub fn kmer_table(given: Option<&Path>, read_type: ReadType) -> Result<KmerTable> {
    match given {
        Some(name) => {
            let Some(found) = find(name) else {
                bail!("provided telomere kmers (-k) could not be found: {}", name.display());
            };
            println!("using user-specified telomere kmers: {}", name.file_name().unwrap_or_default().to_string_lossy());
            println!(" - found at {}", found.display());
            KmerTable::from_path(&found, read_type)
        }
        None => {
            println!("using default telomere kmers");
            KmerTable::parse(DEFAULT_KMERS, read_type)
        }
    }
}

/// Whether `exe` is a file or a command on `PATH` (`check_aligner_exe`).
pub fn executable_exists(exe: &Path) -> bool {
    if exe.is_file() {
        return true;
    }
    if exe.components().count() > 1 {
        return false;
    }
    std::env::var_os("PATH").is_some_and(|paths| std::env::split_paths(&paths).any(|d| d.join(exe).is_file()))
}

/// A seed in `1..=99_999_999` for runs without `--rng`.
pub fn random_seed() -> u64 {
    use std::hash::{BuildHasher, Hasher};
    let mut h = std::collections::hash_map::RandomState::new().build_hasher();
    h.write_u128(std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_nanos()));
    1 + h.finish() % 99_999_999
}
