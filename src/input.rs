//! Streaming readers for FASTA/FASTQ (optionally gzipped), BAM and CRAM input.

use std::fs::File;
use std::io::{self, BufRead, BufReader, Read};
use std::num::NonZero;
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdout, Command, Stdio};
use std::thread;

use anyhow::{Context, Result, bail};
use crossbeam_channel::{Receiver, Sender, bounded};
use flate2::read::MultiGzDecoder;
use noodles::sam::alignment::Record as _;
use noodles::sam::alignment::record::Data;
use noodles::sam::alignment::record::data::field::{Tag, Value, value::Array};
use noodles::{bam, bgzf, cram, fasta, sam};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InputKind {
    Fasta,
    Fastq,
    Bam,
    Cram,
}

impl InputKind {
    /// Classifies a file by suffix, as Telogator2's `get_file_type` does.
    ///
    /// # Errors
    ///
    /// Fails for unknown suffixes.
    pub fn of(path: &Path) -> Result<Self> {
        let name = path.to_string_lossy().to_lowercase();
        let stem = name.strip_suffix(".gz").unwrap_or(&name);
        Ok(if stem.ends_with(".fa") || stem.ends_with(".fasta") {
            Self::Fasta
        } else if stem.ends_with(".fq") || stem.ends_with(".fastq") {
            Self::Fastq
        } else if name.ends_with(".bam") {
            Self::Bam
        } else if name.ends_with(".cram") {
            Self::Cram
        } else {
            bail!("unknown file suffix, cannot determine input type: {}", path.display());
        })
    }
}

/// How `.gz` FASTA/FASTQ input is decompressed.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum Gunzip {
    /// zlib-rs on a thread of its own.
    #[default]
    Builtin,
    /// An external `igzip -dc` (ISA-L), about 1.3x as fast.
    Igzip(PathBuf),
}

impl Gunzip {
    /// Uses `igzip` if given, else `igzip` from `PATH` when there is one, else the
    /// built-in decoder.
    ///
    /// # Errors
    ///
    /// Fails if `igzip` is given but is not an executable.
    pub fn new(igzip: Option<&Path>) -> Result<Self> {
        Ok(match igzip {
            Some(exe) if crate::resources::executable_exists(exe) => Self::Igzip(exe.to_path_buf()),
            Some(exe) => bail!("igzip executable not found: {}", exe.display()),
            None if crate::resources::executable_exists(Path::new("igzip")) => Self::Igzip("igzip".into()),
            None => Self::Builtin,
        })
    }
}

/// How input files are decoded, the same for every input.
#[derive(Clone, Debug, Default)]
pub struct Decoding {
    /// Reference for CRAM input.
    pub cram_ref: Option<PathBuf>,
    /// Decompressor for gzipped FASTA/FASTQ.
    pub gunzip: Gunzip,
}

/// A read from a FASTA/FASTQ file.
#[derive(Clone, Debug, Default)]
pub struct SeqRead {
    pub name: String,
    pub seq: Vec<u8>,
}

/// Line-oriented FASTA/FASTQ parser with Telogator2's naming rules (tabs in headers
/// become spaces).
pub struct FastxReader {
    lines: Box<dyn BufRead + Send>,
    kind: InputKind,
    pending_header: Option<String>,
    line: Vec<u8>,
}

impl std::fmt::Debug for FastxReader {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FastxReader").field("kind", &self.kind).finish_non_exhaustive()
    }
}

impl FastxReader {
    /// Opens a FASTA or FASTQ file; `.gz` files are decompressed (multi-member aware)
    /// with `gunzip`.
    ///
    /// # Errors
    ///
    /// Fails if the file cannot be opened or the decompressor cannot be started.
    pub fn open(path: &Path, kind: InputKind, gunzip: &Gunzip) -> Result<Self> {
        let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
        let gzipped = path.to_string_lossy().to_lowercase().ends_with(".gz");
        let lines: Box<dyn BufRead + Send> = match gunzip {
            _ if !gzipped => Box::new(BufReader::with_capacity(1 << 20, file)),
            Gunzip::Builtin => Box::new(Inflater::spawn(file)?),
            Gunzip::Igzip(exe) => Box::new(Igzip::spawn(exe, file)?),
        };
        Ok(Self { lines, kind, pending_header: None, line: Vec::new() })
    }

    /// Reads one line without its trailing whitespace. Returns `false` at end of file.
    fn next_line(&mut self) -> Result<bool> {
        self.line.clear();
        if self.lines.read_until(b'\n', &mut self.line)? == 0 {
            return Ok(false);
        }
        while self.line.last().is_some_and(u8::is_ascii_whitespace) {
            self.line.pop();
        }
        Ok(true)
    }

    fn header_name(line: &[u8]) -> String {
        let trimmed = line.trim_ascii_start();
        String::from_utf8_lossy(trimmed.get(1..).unwrap_or_default()).replace('\t', " ")
    }

    /// Fills `read` with the next record. Returns `false` at end of file.
    ///
    /// # Errors
    ///
    /// Fails on I/O or decompression errors.
    pub fn next_into(&mut self, read: &mut SeqRead) -> Result<bool> {
        read.seq.clear();
        match self.kind {
            InputKind::Fastq => {
                loop {
                    if !self.next_line()? {
                        return Ok(false);
                    }
                    if !self.line.trim_ascii().is_empty() {
                        break;
                    }
                }
                read.name = Self::header_name(&self.line);
                if self.next_line()? {
                    read.seq.extend_from_slice(self.line.trim_ascii());
                }
                // The `+` and quality lines are not needed.
                self.lines.skip_until(b'\n')?;
                self.lines.skip_until(b'\n')?;
                Ok(true)
            }
            _ => {
                let header = match self.pending_header.take() {
                    Some(h) => h,
                    None => loop {
                        if !self.next_line()? {
                            return Ok(false);
                        }
                        if self.line.contains(&b'>') {
                            break Self::header_name(&self.line);
                        }
                    },
                };
                read.name = header;
                while self.next_line()? {
                    if self.line.contains(&b'>') {
                        self.pending_header = Some(Self::header_name(&self.line));
                        break;
                    }
                    read.seq.extend_from_slice(self.line.trim_ascii());
                }
                Ok(true)
            }
        }
    }
}

/// Decompressed bytes per chunk handed from the inflating thread to the parser.
const INFLATE_CHUNK: usize = 4 << 20;

/// A gzip file decompressed on its own thread, so inflating overlaps with parsing (plain
/// gzip cannot be decompressed in parallel).
struct Inflater {
    chunks: Receiver<io::Result<(Vec<u8>, usize)>>,
    recycle: Sender<Vec<u8>>,
    chunk: Vec<u8>,
    filled: usize,
    pos: usize,
}

impl Inflater {
    fn spawn(file: File) -> Result<Self> {
        let (tx, chunks) = bounded(4);
        let (recycle, spare) = bounded::<Vec<u8>>(4);
        thread::Builder::new().name("inflate".into()).spawn(move || {
            let mut decoder = MultiGzDecoder::new(BufReader::with_capacity(1 << 20, file));
            loop {
                let mut chunk = spare.try_recv().unwrap_or_default();
                chunk.resize(INFLATE_CHUNK, 0);
                let mut filled = 0;
                while filled < chunk.len() {
                    match decoder.read(&mut chunk[filled..]) {
                        Ok(0) => break,
                        Ok(n) => filled += n,
                        Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
                        Err(e) => {
                            let _ = tx.send(Err(e));
                            return;
                        }
                    }
                }
                // Stop at the end of the file, or once the reader is gone.
                if filled == 0 || tx.send(Ok((chunk, filled))).is_err() {
                    return;
                }
            }
        })?;
        Ok(Self { chunks, recycle, chunk: Vec::new(), filled: 0, pos: 0 })
    }
}

impl Read for Inflater {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let n = {
            let available = self.fill_buf()?;
            let n = available.len().min(buf.len());
            buf[..n].copy_from_slice(&available[..n]);
            n
        };
        self.consume(n);
        Ok(n)
    }
}

impl BufRead for Inflater {
    fn fill_buf(&mut self) -> io::Result<&[u8]> {
        if self.pos == self.filled {
            match self.chunks.recv() {
                Ok(Ok((chunk, filled))) => {
                    let used = std::mem::replace(&mut self.chunk, chunk);
                    let _ = self.recycle.try_send(used);
                    (self.filled, self.pos) = (filled, 0);
                }
                Ok(Err(e)) => return Err(e),
                // The inflating thread has finished the file.
                Err(_) => return Ok(&[]),
            }
        }
        Ok(&self.chunk[self.pos..self.filled])
    }

    fn consume(&mut self, amount: usize) {
        self.pos += amount;
    }
}

/// The output of an `igzip -dc` child process reading the file on its stdin. Reports
/// the child's failure as an error at the end of its output, and kills it if dropped
/// before then.
struct Igzip {
    child: Child,
    out: BufReader<ChildStdout>,
    finished: bool,
}

impl Igzip {
    fn spawn(exe: &Path, file: File) -> Result<Self> {
        let mut child = Command::new(exe)
            .arg("-dc")
            .stdin(file)
            .stdout(Stdio::piped())
            .spawn()
            .with_context(|| format!("starting {}", exe.display()))?;
        let stdout = child.stdout.take().expect("stdout is piped");
        Ok(Self { child, out: BufReader::with_capacity(1 << 20, stdout), finished: false })
    }
}

impl Read for Igzip {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let n = {
            let available = self.fill_buf()?;
            let n = available.len().min(buf.len());
            buf[..n].copy_from_slice(&available[..n]);
            n
        };
        self.consume(n);
        Ok(n)
    }
}

impl BufRead for Igzip {
    fn fill_buf(&mut self) -> io::Result<&[u8]> {
        if !self.finished && self.out.fill_buf()?.is_empty() {
            self.finished = true;
            let status = self.child.wait()?;
            if !status.success() {
                return Err(io::Error::other(format!("igzip -dc failed ({status})")));
            }
        }
        self.out.fill_buf()
    }

    fn consume(&mut self, amount: usize) {
        self.out.consume(amount);
    }
}

impl Drop for Igzip {
    fn drop(&mut self) {
        if !self.finished {
            let _ = self.child.kill();
            let _ = self.child.wait();
        }
    }
}

/// A BAM reader inflating BGZF blocks on a thread pool of its own.
pub type BamReader = bam::io::Reader<bgzf::io::MultithreadedReader<File>>;

/// Opens a BAM file, past its header, for streaming with `threads` decompression threads.
///
/// # Errors
///
/// Fails if the file cannot be opened or its header cannot be read.
pub fn open_bam(path: &Path, threads: usize) -> Result<BamReader> {
    let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let workers = NonZero::new(threads).unwrap_or(NonZero::<usize>::MIN);
    let mut reader = bam::io::Reader::from(bgzf::io::MultithreadedReader::with_worker_count(workers, file));
    reader.read_header().with_context(|| format!("reading header of {}", path.display()))?;
    Ok(reader)
}

/// A CRAM reader of raw containers.
pub type CramReader = cram::io::Reader<BufReader<File>>;

/// What decoding a CRAM container needs besides the container itself.
#[derive(Debug)]
pub struct CramContext {
    pub header: sam::Header,
    pub reference: fasta::Repository,
}

/// Opens a CRAM file, past its header. Containers are read raw, to be decoded elsewhere
/// with the returned context.
///
/// # Errors
///
/// Fails if the file, its header or the reference cannot be read.
pub fn open_cram(path: &Path, cram_ref: Option<&Path>) -> Result<(CramReader, CramContext)> {
    let reference = match cram_ref {
        Some(r) => fasta::Repository::new(fasta::repository::adapters::IndexedReader::new(open_indexed_fasta(r)?)),
        None => fasta::Repository::default(),
    };
    let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mut reader = cram::io::Reader::new(BufReader::with_capacity(1 << 20, file));
    let header = reader.read_header().with_context(|| format!("reading header of {}", path.display()))?;
    Ok((reader, CramContext { header, reference }))
}

/// Opens an indexed FASTA, writing `<path>.fai` first if it is missing (as htslib does).
fn open_indexed_fasta(path: &Path) -> Result<fasta::io::IndexedReader<fasta::io::BufReader<File>>> {
    if !path.with_added_extension("fai").exists() {
        index_fasta(path)?;
    }
    fasta::io::indexed_reader::Builder::default()
        .build_from_path(path)
        .with_context(|| format!("opening reference {}", path.display()))
}

/// Writes `<path>.fai` for an uncompressed FASTA file.
///
/// # Errors
///
/// Fails if the file cannot be read or the index cannot be written.
pub fn index_fasta(path: &Path) -> Result<()> {
    let fai = path.with_added_extension("fai");
    let index = fasta::fs::index(path).with_context(|| format!("indexing {}", path.display()))?;
    fasta::fai::fs::write(&fai, &index).with_context(|| format!("writing {}", fai.display()))
}

/// BAM's 4-bit base codes, by code.
const NT16: &[u8; 16] = b"=ACMGRSVTWYHKDBN";

/// Decodes a BAM record's sequence into `out` (uppercase IUPAC letters).
pub fn decode_seq(rec: &bam::Record, out: &mut Vec<u8>) {
    let seq = rec.sequence();
    let len = seq.len();
    let packed = seq.as_bytes();
    out.clear();
    out.reserve(len);
    for &b in &packed[..len / 2] {
        out.push(NT16[(b >> 4) as usize]);
        out.push(NT16[(b & 15) as usize]);
    }
    if len % 2 == 1 {
        out.push(NT16[(packed[len / 2] >> 4) as usize]);
    }
}

/// Decodes a CRAM record's sequence into `out`, normalized as BAM's 4-bit packing would
/// (uppercase IUPAC letters, anything else `N`).
pub fn decode_cram_seq(rec: &cram::Record<'_>, out: &mut Vec<u8>) {
    out.clear();
    out.extend(rec.sequence().iter().map(|b| NT16_NORMALIZED[usize::from(b)]));
}

/// Each byte's letter after a round trip through BAM's 4-bit code (htslib's
/// `seq_nt16_table`, then [`NT16`]).
const NT16_NORMALIZED: [u8; 256] = {
    let mut table = [b'N'; 256];
    let mut code = 0;
    while code < NT16.len() {
        let letter = NT16[code];
        table[letter as usize] = letter;
        table[letter.to_ascii_lowercase() as usize] = letter;
        code += 1;
    }
    table[b'U' as usize] = b'T';
    table[b'u' as usize] = b'T';
    table
};

/// A record's name as htslib gives it (`*` when there is none).
pub fn record_name(rec: &impl sam::alignment::Record) -> String {
    rec.name().map_or_else(|| "*".to_string(), |n| String::from_utf8_lossy(n).into_owned())
}

/// Whether a record's MM/ML tags carry at least one 5mC or 5hmC call that lands on a
/// C of the read (Telogator2's `extract_methylation` returning data).
pub fn has_methylation<'r>(data: &impl Data<'r>, seq: &[u8]) -> bool {
    let (Some(Ok(Value::String(mm))), Some(Ok(Value::Array(Array::UInt8(_))))) =
        (data.get(&Tag::BASE_MODIFICATIONS), data.get(&Tag::BASE_MODIFICATION_PROBABILITIES))
    else {
        return false;
    };
    let Ok(mm) = std::str::from_utf8(mm) else {
        return false;
    };
    if mm.is_empty() {
        return false;
    }
    let num_c = seq.iter().filter(|&&b| b == b'C').count();
    for spec in mm.trim_end_matches(';').split(';') {
        let mut parts = spec.split(',');
        let descriptor = parts.next().unwrap_or_default();
        if !descriptor.contains('C') || !(descriptor.contains('m') || descriptor.contains('h')) {
            continue;
        }
        if let Some(first) = parts.find(|p| !p.is_empty()) {
            // The first call lands on C number `skip`; any later calls land further on.
            if first.parse::<usize>().is_ok_and(|skip| skip < num_c) {
                return true;
            }
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use flate2::Compression;
    use flate2::write::GzEncoder;

    use super::*;

    fn try_records(path: &Path, kind: InputKind, gunzip: &Gunzip) -> Result<Vec<(String, Vec<u8>)>> {
        let mut reader = FastxReader::open(path, kind, gunzip)?;
        let mut read = SeqRead::default();
        let mut out = Vec::new();
        while reader.next_into(&mut read)? {
            out.push((read.name.clone(), read.seq.clone()));
        }
        Ok(out)
    }

    fn records(path: &Path, kind: InputKind, gunzip: &Gunzip) -> Vec<(String, Vec<u8>)> {
        try_records(path, kind, gunzip).expect("test file parses")
    }

    /// The built-in decoder, plus igzip when it is on `PATH`.
    fn decoders() -> Vec<Gunzip> {
        let mut out = vec![Gunzip::Builtin];
        match Gunzip::new(None).expect("PATH lookup cannot fail") {
            Gunzip::Builtin => eprintln!("igzip not on PATH; testing the built-in decoder only"),
            igzip => out.push(igzip),
        }
        out
    }

    /// Gzipped input (here two members, with records straddling every 4 MB chunk) parses
    /// exactly as the same text uncompressed.
    #[test]
    fn gzip_input_parses_like_plain_text() {
        let dir = std::env::temp_dir().join(format!("telocroc-input-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("temp dir");
        let mut fastq = Vec::new();
        let mut fasta = Vec::new();
        for i in 0..3000 {
            let seq: Vec<u8> = (0..1000 + i * 7 % 3001).map(|k| b"ACGT"[(k * 7 + i) % 4]).collect();
            let blank = if i % 50 == 0 { "\n\n" } else { "" };
            writeln!(fastq, "{blank}@read{i}\tx").and_then(|()| fastq.write_all(&seq)).expect("vec write");
            write!(fastq, "\r\n+\n{}\n", "I".repeat(seq.len())).expect("vec write");
            writeln!(fasta, ">read{i}\tx").and_then(|()| fasta.write_all(&seq[..500])).expect("vec write");
            fasta.write_all(b"\n").and_then(|()| fasta.write_all(&seq[500..])).expect("vec write");
            fasta.write_all(b"\n").expect("vec write");
        }
        for (name, kind, text) in [("r.fq", InputKind::Fastq, &fastq), ("r.fa", InputKind::Fasta, &fasta)] {
            let plain = dir.join(name);
            std::fs::write(&plain, text).expect("write plain");
            let mut gz = Vec::new();
            let half = text.len() / 2;
            for part in [&text[..half], &text[half..]] {
                let mut enc = GzEncoder::new(Vec::new(), Compression::fast());
                enc.write_all(part).expect("vec write");
                gz.extend(enc.finish().expect("vec write"));
            }
            let zipped = dir.join(format!("{name}.gz"));
            std::fs::write(&zipped, gz).expect("write gz");
            let want = records(&plain, kind, &Gunzip::Builtin);
            assert_eq!(want.len(), 3000);
            assert_eq!(want[7].0, "read7 x");
            for gunzip in decoders() {
                let got = records(&zipped, kind, &gunzip);
                assert_eq!(got.len(), want.len(), "{name}, {gunzip:?}: record count");
                assert!(got == want, "{name}, {gunzip:?}: records differ");
            }
            let truncated = dir.join(format!("cut.{name}.gz"));
            let gz = std::fs::read(&zipped).expect("read gz");
            std::fs::write(&truncated, &gz[..gz.len() / 3]).expect("write gz");
            for gunzip in decoders() {
                assert!(try_records(&truncated, kind, &gunzip).is_err(), "{name}, {gunzip:?}: truncated file read");
            }
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// CRAM bases normalize as htslib's 4-bit packing does: case folded, U read as T,
    /// anything else N.
    #[test]
    fn cram_bases_normalize_like_bam_packing() {
        let cases = [(b'a', b'A'), (b'C', b'C'), (b'=', b'='), (b'u', b'T'), (b'y', b'Y'), (b'x', b'N'), (b'*', b'N')];
        for (base, want) in cases {
            assert_eq!(NT16_NORMALIZED[usize::from(base)], want, "base {:?}", char::from(base));
        }
    }

    /// A 5mC or 5hmC call counts only if it lands on one of the read's Cs.
    #[test]
    fn methylation_needs_a_call_on_a_read_c() {
        use noodles::sam::alignment::record_buf::data::field::Value as BufValue;
        use noodles::sam::alignment::record_buf::data::field::value::Array as BufArray;

        let data = |mm: &str| -> sam::alignment::record_buf::Data {
            let ml = BufValue::Array(BufArray::UInt8(vec![200]));
            [(Tag::BASE_MODIFICATIONS, BufValue::from(mm)), (Tag::BASE_MODIFICATION_PROBABILITIES, ml)]
                .into_iter()
                .collect()
        };
        let seq = b"ACGTCC";
        for (mm, want) in [("C+m?,2;", true), ("C+h,0;", true), ("C+m?,3;", false), ("A+a,0;", false), ("", false)] {
            assert_eq!(has_methylation(&&data(mm), seq), want, "MM:Z:{mm}");
        }
        let no_ml: sam::alignment::record_buf::Data =
            [(Tag::BASE_MODIFICATIONS, BufValue::from("C+m,0;"))].into_iter().collect();
        assert!(!has_methylation(&&no_ml, seq), "MM without ML");
    }
}
