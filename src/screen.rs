//! Stage 0a: stream every input read and keep the ones with enough tandem canonical
//! repeats (`extract_telomere_reads`).
//!
//! The reader thread only parses records; decoding, kmer counting and compression of the
//! kept reads run on worker threads. Kept reads come back in input order.

use std::collections::HashSet;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::thread;

use anyhow::{Context, Result};
use crossbeam_channel::{Receiver, Sender, bounded};
use flate2::Compression;
use flate2::write::GzEncoder;
use memchr::memmem::Finder;
use noodles::bam;
use noodles::cram::io::reader::Container;
use noodles::sam::alignment::Record as _;

use crate::input::{self, BamReader, CramContext, CramReader, Decoding, FastxReader, InputKind, SeqRead};

mod packed;

use packed::PackedPrefilter;

const BATCH_READS: usize = 256;
const BATCH_BASES: usize = 4_000_000;
/// Stack per screen worker: noodles' CRAM decoders overflow the default 2 MiB.
const WORKER_STACK: usize = 32 << 20;

/// A telomere read kept by the screen.
#[derive(Clone, Debug)]
pub struct TelRead {
    pub name: String,
    pub seq: Vec<u8>,
}

/// Counts and reads from the screen.
#[derive(Debug, Default)]
pub struct ScreenResult {
    pub reads: Vec<TelRead>,
    pub all_reads: u64,
    pub supplementary: u64,
    pub total_bp_all: u64,
    pub total_bp_tel: u64,
    /// Names of kept reads with usable MM/ML methylation calls.
    pub methylated: HashSet<String>,
}

/// Work for a screen worker. `records` and `reads` are reused buffers, filled up to `len`.
enum BatchData {
    Records {
        records: Vec<bam::Record>,
        len: usize,
    },
    Reads {
        reads: Vec<SeqRead>,
        len: usize,
    },
    /// One CRAM container, decoded by the worker.
    Container {
        container: Container,
        context: Arc<CramContext>,
    },
}

struct Batch {
    id: usize,
    data: BatchData,
}

#[derive(Default)]
struct BatchOut {
    id: usize,
    reads: Vec<TelRead>,
    methylated: Vec<String>,
    gz_member: Vec<u8>,
    all_reads: u64,
    supplementary: u64,
    bp_all: u64,
    bp_tel: u64,
}

struct Matcher<'k> {
    fwd: Finder<'k>,
    rev: Finder<'k>,
    min_hits: usize,
    packed: Option<PackedPrefilter>,
}

impl Matcher<'_> {
    /// `false` only for BAM records that cannot pass [`Self::is_telomeric`].
    fn may_match_packed(&self, rec: &bam::Record) -> bool {
        let seq = rec.sequence();
        self.packed.as_ref().is_none_or(|p| p.may_match(seq.as_bytes(), seq.len(), self.min_hits))
    }

    /// `seq.count(kmer) >= min_hits or seq.count(rc) >= min_hits` (non-overlapping counts).
    fn is_telomeric(&self, seq: &[u8]) -> bool {
        self.fwd.find_iter(seq).take(self.min_hits).count() >= self.min_hits
            || self.rev.find_iter(seq).take(self.min_hits).count() >= self.min_hits
    }
}

/// Screens all inputs and writes the kept reads to `out_fa_gz`.
///
/// # Errors
///
/// Fails on unreadable input or output files.
pub fn extract_telomere_reads(
    inputs: &[PathBuf],
    out_fa_gz: &Path,
    kmer: &[u8],
    kmer_rc: &[u8],
    min_hits: usize,
    threads: usize,
    decoding: &Decoding,
) -> Result<ScreenResult> {
    let matcher = Matcher {
        fwd: Finder::new(kmer),
        rev: Finder::new(kmer_rc),
        min_hits,
        packed: PackedPrefilter::new(kmer, kmer_rc),
    };
    let workers = threads.max(1);
    let (work_tx, work_rx) = bounded::<Batch>(2 * workers);
    let (done_tx, done_rx) = bounded::<Result<BatchOut>>(2 * workers);
    let (recycle_tx, recycle_rx) = bounded::<BatchData>(4 * workers);

    let mut outs: Vec<BatchOut> = thread::scope(|scope| -> Result<Vec<BatchOut>> {
        for _ in 0..workers {
            let (work_rx, done_tx, recycle_tx) = (work_rx.clone(), done_tx.clone(), recycle_tx.clone());
            let matcher = &matcher;
            thread::Builder::new()
                .name("screen".into())
                .stack_size(WORKER_STACK)
                .spawn_scoped(scope, move || worker(matcher, &work_rx, &done_tx, &recycle_tx))?;
        }
        drop((work_rx, done_tx, recycle_tx));
        let collector = scope.spawn(move || done_rx.iter().collect::<Vec<_>>());
        let read_result = read_all(inputs, threads, decoding, &work_tx, &recycle_rx);
        drop(work_tx);
        let outs = collector.join().expect("screen collector thread panicked");
        read_result?;
        outs.into_iter().collect::<Result<Vec<_>>>()
    })?;
    outs.sort_unstable_by_key(|o| o.id);

    let mut file =
        BufWriter::new(File::create(out_fa_gz).with_context(|| format!("creating {}", out_fa_gz.display()))?);
    let mut result = ScreenResult::default();
    for out in outs {
        file.write_all(&out.gz_member)?;
        result.all_reads += out.all_reads;
        result.supplementary += out.supplementary;
        result.total_bp_all += out.bp_all;
        result.total_bp_tel += out.bp_tel;
        result.methylated.extend(out.methylated);
        result.reads.extend(out.reads);
    }
    if result.reads.is_empty() {
        file.write_all(&GzEncoder::new(Vec::new(), Compression::default()).finish()?)?;
    }
    file.flush()?;
    Ok(result)
}

fn read_all(
    inputs: &[PathBuf],
    threads: usize,
    decoding: &Decoding,
    work_tx: &Sender<Batch>,
    recycle_rx: &Receiver<BatchData>,
) -> Result<()> {
    let mut next_id = 0;
    for path in inputs {
        match InputKind::of(path)? {
            InputKind::Bam => {
                let mut reader = input::open_bam(path, threads)?;
                read_bam(&mut reader, path, work_tx, recycle_rx, &mut next_id)?;
            }
            InputKind::Cram => {
                let (mut reader, context) = input::open_cram(path, decoding.cram_ref.as_deref())?;
                read_cram(&mut reader, &Arc::new(context), path, work_tx, recycle_rx, &mut next_id);
            }
            kind => {
                let mut reader = FastxReader::open(path, kind, &decoding.gunzip)?;
                read_fastx(&mut reader, work_tx, recycle_rx, &mut next_id)?;
            }
        }
    }
    Ok(())
}

fn read_bam(
    reader: &mut BamReader,
    path: &Path,
    work_tx: &Sender<Batch>,
    recycle_rx: &Receiver<BatchData>,
    next_id: &mut usize,
) -> Result<()> {
    let fresh = || match recycle_rx.try_recv() {
        Ok(BatchData::Records { records, .. }) => records,
        _ => Vec::new(),
    };
    let mut records = fresh();
    let (mut len, mut bases) = (0, 0);
    loop {
        if records.len() <= len {
            records.push(bam::Record::default());
        }
        match reader.read_record(&mut records[len]) {
            Ok(0) => break,
            Ok(_) => {}
            Err(e) => {
                // Telogator2 treats a truncated or corrupt file as ending early.
                eprintln!("Warning: stopped reading {} early: {e}", path.display());
                break;
            }
        }
        bases += records[len].sequence().len();
        len += 1;
        if len >= BATCH_READS || bases >= BATCH_BASES {
            let records = std::mem::replace(&mut records, fresh());
            let batch = Batch { id: *next_id, data: BatchData::Records { records, len } };
            *next_id += 1;
            work_tx.send(batch).expect("screen workers stopped early");
            (len, bases) = (0, 0);
        }
    }
    if len > 0 {
        work_tx
            .send(Batch { id: *next_id, data: BatchData::Records { records, len } })
            .expect("screen workers stopped early");
        *next_id += 1;
    }
    Ok(())
}

/// Sends each container as a batch of its own; workers decode them.
fn read_cram(
    reader: &mut CramReader,
    context: &Arc<CramContext>,
    path: &Path,
    work_tx: &Sender<Batch>,
    recycle_rx: &Receiver<BatchData>,
    next_id: &mut usize,
) {
    loop {
        let mut container = match recycle_rx.try_recv() {
            Ok(BatchData::Container { container, .. }) => container,
            _ => Container::default(),
        };
        match reader.read_container(&mut container) {
            Ok(0) => break,
            Ok(_) => {}
            Err(e) => {
                // Telogator2 treats a truncated or corrupt file as ending early.
                eprintln!("Warning: stopped reading {} early: {e}", path.display());
                break;
            }
        }
        let batch = Batch { id: *next_id, data: BatchData::Container { container, context: Arc::clone(context) } };
        *next_id += 1;
        work_tx.send(batch).expect("screen workers stopped early");
    }
}

fn read_fastx(
    reader: &mut FastxReader,
    work_tx: &Sender<Batch>,
    recycle_rx: &Receiver<BatchData>,
    next_id: &mut usize,
) -> Result<()> {
    let fresh = || match recycle_rx.try_recv() {
        Ok(BatchData::Reads { reads, .. }) => reads,
        _ => Vec::new(),
    };
    let mut reads = fresh();
    let (mut len, mut bases) = (0, 0);
    loop {
        if reads.len() <= len {
            reads.push(SeqRead::default());
        }
        if !reader.next_into(&mut reads[len])? {
            break;
        }
        bases += reads[len].seq.len();
        len += 1;
        if len >= BATCH_READS || bases >= BATCH_BASES {
            let reads = std::mem::replace(&mut reads, fresh());
            let batch = Batch { id: *next_id, data: BatchData::Reads { reads, len } };
            *next_id += 1;
            work_tx.send(batch).expect("screen workers stopped early");
            (len, bases) = (0, 0);
        }
    }
    if len > 0 {
        work_tx
            .send(Batch { id: *next_id, data: BatchData::Reads { reads, len } })
            .expect("screen workers stopped early");
        *next_id += 1;
    }
    Ok(())
}

fn worker(
    matcher: &Matcher<'_>,
    work_rx: &Receiver<Batch>,
    done_tx: &Sender<Result<BatchOut>>,
    recycle_tx: &Sender<BatchData>,
) {
    let mut seq_buf = Vec::new();
    for batch in work_rx.iter() {
        let mut out = BatchOut { id: batch.id, ..Default::default() };
        let screened = match &batch.data {
            BatchData::Records { records, len } => {
                for rec in &records[..*len] {
                    let seq_len = rec.sequence().len();
                    if seq_len == 0 {
                        continue;
                    }
                    if rec.flags().is_supplementary() {
                        out.supplementary += 1;
                        continue;
                    }
                    out.all_reads += 1;
                    out.bp_all += seq_len as u64;
                    if !matcher.may_match_packed(rec) {
                        continue;
                    }
                    input::decode_seq(rec, &mut seq_buf);
                    if matcher.is_telomeric(&seq_buf) {
                        let methylated = input::has_methylation(&rec.data(), &seq_buf);
                        keep(input::record_name(rec), &seq_buf, methylated, &mut out);
                    }
                }
                Ok(())
            }
            BatchData::Reads { reads, len } => {
                for read in &reads[..*len] {
                    if read.seq.is_empty() {
                        continue;
                    }
                    out.all_reads += 1;
                    out.bp_all += read.seq.len() as u64;
                    if matcher.is_telomeric(&read.seq) {
                        keep(read.name.clone(), &read.seq, false, &mut out);
                    }
                }
                Ok(())
            }
            BatchData::Container { container, context } => {
                screen_container(matcher, container, context, &mut seq_buf, &mut out)
            }
        };
        let out = screened.map(|()| {
            out.gz_member = compress_reads(&out.reads);
            out
        });
        // Ignore a full recycle queue; the reader will just allocate a new batch.
        let _ = recycle_tx.try_send(batch.data);
        if done_tx.send(out).is_err() {
            return;
        }
    }
}

/// Decodes and screens the records of one CRAM container.
fn screen_container(
    matcher: &Matcher<'_>,
    container: &Container,
    context: &CramContext,
    seq_buf: &mut Vec<u8>,
    out: &mut BatchOut,
) -> Result<()> {
    let compression_header = container.compression_header().context("reading CRAM compression header")?;
    for slice in container.slices() {
        let slice = slice.context("reading CRAM slice")?;
        let (core, external) = slice.decode_blocks().context("decoding CRAM slice")?;
        let records = slice
            .records(context.reference.clone(), &context.header, &compression_header, &core, &external)
            .context("decoding CRAM records")?;
        for rec in &records {
            let seq_len = rec.sequence().len();
            if seq_len == 0 {
                continue;
            }
            if rec.flags()?.is_supplementary() {
                out.supplementary += 1;
                continue;
            }
            out.all_reads += 1;
            out.bp_all += seq_len as u64;
            input::decode_cram_seq(rec, seq_buf);
            if matcher.is_telomeric(seq_buf) {
                let methylated = input::has_methylation(&rec.data(), seq_buf);
                keep(input::record_name(rec), seq_buf, methylated, out);
            }
        }
    }
    Ok(())
}

fn keep(name: String, seq: &[u8], methylated: bool, out: &mut BatchOut) {
    out.bp_tel += seq.len() as u64;
    if methylated {
        out.methylated.push(name.clone());
    }
    out.reads.push(TelRead { name, seq: seq.to_vec() });
}

/// One gzip member holding `>name\nseq\n` for each kept read (members concatenate).
fn compress_reads(reads: &[TelRead]) -> Vec<u8> {
    if reads.is_empty() {
        return Vec::new();
    }
    let mut enc = GzEncoder::new(Vec::new(), Compression::new(6));
    for r in reads {
        enc.write_all(b">").and_then(|()| enc.write_all(r.name.as_bytes())).expect("writing to a Vec cannot fail");
        enc.write_all(b"\n").and_then(|()| enc.write_all(&r.seq)).expect("writing to a Vec cannot fail");
        enc.write_all(b"\n").expect("writing to a Vec cannot fail");
    }
    enc.finish().expect("writing to a Vec cannot fail")
}

/// `quick_grab_all_reads_nodup`: drop reads shorter than `min_len`, then keep one read per
/// name (the last one seen, at the position of the first).
pub fn dedup_by_name(reads: Vec<TelRead>, min_len: usize) -> (Vec<TelRead>, usize) {
    let mut index: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    let mut out: Vec<TelRead> = Vec::new();
    let mut filtered = 0;
    for r in reads {
        if r.seq.len() < min_len {
            filtered += 1;
            continue;
        }
        match index.get(&r.name) {
            Some(&i) => out[i] = r,
            None => {
                index.insert(r.name.clone(), out.len());
                out.push(r);
            }
        }
    }
    (out, filtered)
}
