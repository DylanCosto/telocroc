//! Regressions for empty clusters and successful reruns with no retained reads.

use std::process::Command;

#[test]
fn fast_alignment_and_empty_reruns() {
    let dir = std::env::temp_dir().join(format!("telocroc-regression-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let input = dir.join("reads.fa");
    let reference = dir.join("ref.fa");
    let output = dir.join("out");
    let mut seed = 372u64;
    let subtel: String = (0..2500)
        .map(|_| {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            b"ACGT"[(seed >> 32) as usize & 3] as char
        })
        .collect();
    let seq = format!("{subtel}{}{}", "TCAGGG".repeat(50), "TTAGGG".repeat(300));
    let reads: String = (0..3).map(|i| format!(">r{i}\n{seq}\n")).collect();
    std::fs::write(&reference, format!(">CHM13_chr1q\n{subtel}\n")).unwrap();
    let run = |extra: &[&str]| {
        let out = Command::new(env!("CARGO_BIN_EXE_telocroc"))
            .arg("-i")
            .arg(&input)
            .arg("-o")
            .arg(&output)
            .arg("-t")
            .arg(&reference)
            // No anchoring is requested; the executable is only checked for existence.
            .arg("--minimap2")
            .arg(std::env::current_exe().unwrap())
            .args(["--debug-noanchor", "-p2", "--rng", "10"])
            .args(extra)
            .output()
            .unwrap();
        assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    };
    let table = output.join("tlens_by_allele.tsv");
    for (replacement, extra) in [
        (format!(">no_tel\n{}\n", "A".repeat(4500)), vec![]),
        (reads.clone(), vec!["-l", "10000"]),
        (reads.clone(), vec!["--filt-tel", "10000"]),
    ] {
        std::fs::write(&input, &reads).unwrap();
        run(&["--fast-aln"]);
        assert_eq!(std::fs::read_to_string(&table).unwrap().lines().count(), 2);
        std::fs::write(&input, replacement).unwrap();
        run(&extra);
        let empty = std::fs::read_to_string(&table).unwrap();
        assert_eq!(empty.lines().count(), 1, "previous alleles survived an empty rerun");
        assert!(empty.starts_with("#chr\t"));
    }
    std::fs::remove_dir_all(&dir).unwrap();
}
