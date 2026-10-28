//! The telomere repeat table (`kmers.tsv`): which repeat kmers to look for, and the letter
//! each one paints onto a read's repeat "color vector".

use std::fs;
use std::path::Path;

use anyhow::{Context, Result, bail};

use crate::cli::ReadType;
use crate::util::revcomp;

/// One repeat kmer from `kmers.tsv`.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Kmer {
    pub seq: Vec<u8>,
    pub color: String,
    pub letter: u8,
    pub flags: Vec<String>,
}

impl Kmer {
    pub fn has_flag(&self, flag: &str) -> bool {
        self.flags.iter().any(|f| f == flag)
    }
}

/// The kmer table after read-type filtering, sorted longest first as `read_kmer_tsv` does.
#[derive(Clone, Debug)]
pub struct KmerTable {
    pub kmers: Vec<Kmer>,
    /// Reverse complements of `kmers`, in the same order.
    pub kmers_rev: Vec<Vec<u8>>,
    /// For each kmer, the indices of the other kmers that contain it as a substring.
    pub is_substring: Vec<Vec<usize>>,
    /// Canonical repeats in file order (not sorted).
    pub canonical: Vec<Vec<u8>>,
    pub canonical_rev: Vec<Vec<u8>>,
    /// Letter of the first canonical kmer in the file.
    pub canonical_letter: u8,
}

impl KmerTable {
    /// Parses `kmers.tsv`, keeping the rows that apply to `read_type`.
    ///
    /// # Errors
    ///
    /// Fails if the file cannot be read, a row is malformed, or no kmer is canonical.
    pub fn from_path(path: &Path, read_type: ReadType) -> Result<Self> {
        let text = fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        Self::parse(&text, read_type)
    }

    pub(crate) fn parse(text: &str, read_type: ReadType) -> Result<Self> {
        let mut rows = Vec::new();
        let mut canonical = Vec::new();
        let mut canonical_letter = None;
        for line in text.lines() {
            if line.starts_with('#') || line.trim().is_empty() {
                continue;
            }
            let fields: Vec<&str> = line.trim().split('\t').collect();
            if fields.len() < 4 || fields[2].len() != 1 {
                bail!("malformed kmers.tsv row: {line:?}");
            }
            let flags: Vec<String> = fields[3].split(',').map(str::to_string).collect();
            let has = |f: &str| flags.iter().any(|x| x == f);
            if has("pacbio_only") && read_type != ReadType::Hifi {
                continue;
            }
            if has("nanopore_only") && read_type != ReadType::Ont {
                continue;
            }
            let kmer = Kmer {
                seq: fields[0].as_bytes().to_vec(),
                color: fields[1].to_string(),
                letter: fields[2].as_bytes()[0],
                flags,
            };
            if kmer.has_flag("canonical") {
                canonical.push(kmer.seq.clone());
                canonical_letter.get_or_insert(kmer.letter);
            }
            rows.push(kmer);
        }
        let Some(canonical_letter) = canonical_letter else {
            bail!("kmer list does not contain any kmer marked as canonical");
        };
        // sorted(set((len, kmer, color, letter, flags)), reverse=True)
        rows.sort_by(|a, b| (a.seq.len(), a).cmp(&(b.seq.len(), b)));
        rows.dedup();
        rows.reverse();
        let is_substring = (0..rows.len())
            .map(|i| {
                (0..rows.len())
                    .filter(|&j| j != i && memchr::memmem::find(&rows[j].seq, &rows[i].seq).is_some())
                    .collect()
            })
            .collect();
        Ok(Self {
            kmers_rev: rows.iter().map(|k| revcomp(&k.seq)).collect(),
            canonical_rev: canonical.iter().map(|k| revcomp(k)).collect(),
            kmers: rows,
            is_substring,
            canonical,
            canonical_letter,
        })
    }

    pub fn letters(&self) -> impl Iterator<Item = u8> + '_ {
        self.kmers.iter().map(|k| k.letter)
    }

    /// Distinct letters of kmers carrying `flag`.
    pub fn letters_with_flag(&self, flag: &str) -> Vec<u8> {
        let mut v: Vec<u8> = self.kmers.iter().filter(|k| k.has_flag(flag)).map(|k| k.letter).collect();
        v.sort_unstable();
        v.dedup();
        v
    }
}
