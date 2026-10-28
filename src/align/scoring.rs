//! Scoring schemes, matching `get_scoring_matrix` and `get_aligner_object`.
//!
//! Repeat color vectors are aligned as if each kmer letter were an amino acid; `A` marks
//! unknown sequence and `-` a gap character inside an existing alignment row.

/// The 20-letter alphabet repeat letters are drawn from.
pub const AMINO: &[u8; 20] = b"ACDEFGHIKLMNPQRSTVWY";
/// Letter for sequence not covered by any repeat kmer.
pub const UNKNOWN: u8 = b'A';
/// Gap character inside alignment rows.
pub const GAP: u8 = b'-';
/// Matrix alphabet size: `AMINO` plus `-`.
pub const MATRIX_LETTERS: usize = 21;

/// Index of a letter in the matrix alphabet.
pub fn matrix_index(c: u8) -> usize {
    match c {
        b'A' => 0,
        b'C' => 1,
        b'D' => 2,
        b'E' => 3,
        b'F' => 4,
        b'G' => 5,
        b'H' => 6,
        b'I' => 7,
        b'K' => 8,
        b'L' => 9,
        b'M' => 10,
        b'N' => 11,
        b'P' => 12,
        b'Q' => 13,
        b'R' => 14,
        b'S' => 15,
        b'T' => 16,
        b'V' => 17,
        b'W' => 18,
        b'Y' => 19,
        b'-' => 20,
        _ => panic!("letter {:?} is not in the alignment alphabet", c as char),
    }
}

/// Which substitution matrix `get_scoring_matrix` builds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MatrixKind {
    Tvr,
    Msa,
    MsaRefinement,
}

/// Substitution scores.
#[derive(Clone, Debug)]
pub enum Substitution {
    /// Scores over the 21-letter matrix alphabet.
    Matrix(Box<[[i32; MATRIX_LETTERS]; MATRIX_LETTERS]>),
    /// Biopython's default when no matrix is set: equal letters score `matches`.
    MatchMismatch,
}

/// Biopython's match/mismatch scores when no substitution matrix is set.
pub const MATCH_SCORE: i32 = 5;
pub const MISMATCH_SCORE: i32 = -4;

impl Substitution {
    /// Builds a matrix exactly as `get_scoring_matrix(canon_char, which_type)` does,
    /// including the order of its overlapping assignments.
    pub fn matrix(canonical: u8, kind: MatrixKind) -> Self {
        let mut m = [[0i32; MATRIX_LETTERS]; MATRIX_LETTERS];
        let ix = matrix_index;
        let (canon, unk, gap) = (ix(canonical), ix(UNKNOWN), ix(GAP));
        for &c1 in AMINO {
            m[ix(c1)][ix(c1)] = 5;
            for &c2 in AMINO {
                if c1 != c2 {
                    m[ix(c1)][ix(c2)] = -4;
                }
            }
        }
        m[unk][unk] = 2;
        m[gap][gap] = 0;
        for &c1 in AMINO {
            let c = ix(c1);
            m[c][canon] = -4;
            m[canon][c] = -4;
            m[c][unk] = -2;
            m[unk][c] = -2;
            m[c][gap] = -4;
            m[gap][c] = -4;
        }
        let mut set_sym = |a: usize, b: usize, v: i32| {
            m[a][b] = v;
            m[b][a] = v;
        };
        match kind {
            MatrixKind::Tvr => {
                set_sym(canon, canon, 0);
                set_sym(canon, unk, -2);
                set_sym(unk, gap, -2);
            }
            MatrixKind::Msa => {
                set_sym(canon, canon, 0);
                set_sym(canon, unk, 1);
                set_sym(unk, gap, -2);
            }
            MatrixKind::MsaRefinement => {
                for &c1 in AMINO {
                    set_sym(ix(c1), gap, 0);
                }
                set_sym(canon, canon, 2);
                set_sym(canon, unk, 1);
                set_sym(unk, gap, 1);
            }
        }
        Self::Matrix(Box::new(m))
    }

    /// Score of aligning letter `a` against letter `b`.
    pub fn score(&self, a: u8, b: u8) -> i32 {
        match self {
            Self::Matrix(m) => m[matrix_index(a)][matrix_index(b)],
            Self::MatchMismatch if a == b => MATCH_SCORE,
            Self::MatchMismatch => MISMATCH_SCORE,
        }
    }
}

/// Gap scores for one Biopython aligner preset (`which_type`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GapKind {
    /// `tvr`: open -4, extend -4.
    Tvr,
    /// `msa`: open -16, extend -4.
    Msa,
}

/// A configured global aligner: substitution scores, affine gaps (Biopython convention:
/// a gap of length k scores `open + (k-1) * extend`) and whether trailing gaps are free.
#[derive(Clone, Debug)]
pub struct Scoring {
    pub sub: Substitution,
    pub open: i32,
    pub extend: i32,
    /// `gap_bool = (True, False)`: right (trailing) end gaps score zero.
    pub free_right_gaps: bool,
}

impl Scoring {
    pub fn new(sub: Substitution, gaps: GapKind, free_right_gaps: bool) -> Self {
        let (open, extend) = match gaps {
            GapKind::Tvr => (-4, -4),
            GapKind::Msa => (-16, -4),
        };
        Self { sub, open, extend, free_right_gaps }
    }

    /// Score of a leading (left) gap of length `len >= 1`.
    pub fn left_gap(&self, len: usize) -> i32 {
        self.open + self.extend * (len as i32 - 1)
    }

    /// Sum of each letter's self-score (`iden_score` half in `tvr_distance`).
    pub fn self_score(&self, s: &[u8]) -> f64 {
        match &self.sub {
            Substitution::Matrix(m) => {
                let mut counts = [0usize; MATRIX_LETTERS];
                for &c in s {
                    counts[matrix_index(c)] += 1;
                }
                counts.iter().enumerate().map(|(i, &n)| n as f64 * f64::from(m[i][i])).sum()
            }
            Substitution::MatchMismatch => f64::from(MATCH_SCORE) * s.len() as f64,
        }
    }
}
