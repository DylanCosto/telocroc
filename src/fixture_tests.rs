//! Checks against values produced by the real Python libraries
//! (`tools/make_fixtures.py` writes `tests/fixtures/python_reference.json`).

use serde_json::Value;

use crate::cli::ReadType;
use crate::kmers::KmerTable;
use crate::linkage::{self, Method};
use crate::{numpy, telrep, wavelet};

fn fixtures() -> Value {
    let text = include_str!("../tests/fixtures/python_reference.json");
    serde_json::from_str(text).expect("fixture JSON parses")
}

fn floats(v: &Value) -> Vec<f64> {
    v.as_array().expect("array").iter().map(|x| x.as_f64().expect("number")).collect()
}

fn usizes(v: &Value) -> Vec<usize> {
    v.as_array().expect("array").iter().map(|x| x.as_u64().expect("integer") as usize).collect()
}

#[test]
fn wavelet_smoothing_matches_pywavelets_bit_for_bit() {
    for case in fixtures()["wavelet"].as_array().expect("cases") {
        let (x, want) = (floats(&case["x"]), floats(&case["y"]));
        let got = wavelet::smooth(&x);
        assert_eq!(got.len(), want.len(), "length for n = {}", x.len());
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            assert_eq!(g.to_bits(), w.to_bits(), "n = {}, index {i}: {g} vs {w}", x.len());
        }
    }
}

#[test]
fn percentile_matches_numpy() {
    for case in fixtures()["percentile"].as_array().expect("cases") {
        let v = floats(&case["v"]);
        let q = case["q"].as_f64().expect("q");
        let want = case["p"].as_f64().expect("p");
        assert_eq!(numpy::percentile(&v, q), want, "np.percentile({v:?}, {q})");
    }
}

#[test]
fn linkage_cuts_match_scipy() {
    for case in fixtures()["linkage"].as_array().expect("cases") {
        let method = match case["method"].as_str().expect("method") {
            "ward" => Method::Ward,
            _ => Method::Single,
        };
        let n = case["n"].as_u64().expect("n") as usize;
        let cut = case["cut"].as_f64().expect("cut");
        let mut got = linkage::cluster(&floats(&case["d"]), n, method, cut);
        got.sort();
        let want: Vec<Vec<usize>> = case["clusters"].as_array().expect("clusters").iter().map(usizes).collect();
        assert_eq!(got, want, "{method:?} n = {n} cut = {cut}");
    }
}

#[test]
fn kmer_hits_match_telogator2() {
    let table = KmerTable::parse(include_str!("../resources/kmers.tsv"), ReadType::Hifi).expect("kmer table");
    let rev: Vec<&[u8]> = table.kmers_rev.iter().map(Vec::as_slice).collect();
    for case in fixtures()["kmer_hits"].as_array().expect("cases") {
        let seq = case["seq"].as_str().expect("seq").as_bytes();
        let got = telrep::nonoverlapping_kmer_hits(seq, &rev, &table.is_substring);
        let want: Vec<Vec<(usize, usize)>> = case["hits"]
            .as_array()
            .expect("hits")
            .iter()
            .map(|k| k.as_array().expect("spans").iter().map(|s| (usizes(s)[0], usizes(s)[1])).collect())
            .collect();
        assert_eq!(got, want);
    }
}
