//! Replay an unprofiled package to time its resident input/output protocol.
use anyhow::{Context, Result, ensure};
use clap::Parser;
use ipu_package::Application;
use ipu_runtime::{Runtime, init_tracing};
use std::{
    fs,
    path::PathBuf,
    time::{Duration, Instant},
};

#[derive(Parser)]
struct Args {
    package: PathBuf,
    configuration: PathBuf,
    #[arg(long)]
    data: PathBuf,
    #[arg(long)]
    sdk: PathBuf,
    #[arg(long, default_value = "/dev/ipu0")]
    device: String,
    #[arg(long, default_value_t = 100)]
    poll_us: u64,
    #[arg(long)]
    device_lock: PathBuf,
}

fn main() -> Result<()> {
    init_tracing();
    let args = Args::parse();
    let application = Application::read(fs::File::open(&args.package)?)?;
    ensure!(
        application
            .outputs
            .iter()
            .all(|b| !b.name.starts_with("profile.")),
        "use --no-profile so readback contains only model outputs"
    );
    let run = application
        .host_exchange
        .calls
        .iter()
        .find(|c| c.name == "run")
        .context("package has no run call")?;
    let input = fs::read(args.data.join("input.bin"))?;
    ensure!(
        input.len() == input_bytes(run),
        "input size does not match package"
    );
    let weights = fs::read(args.data.join("weights.bin"))?;
    let expected = fs::read(args.data.join("output.bin"))?;
    let lock = fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(&args.device_lock)?;
    lock.lock()?;
    let runtime = Runtime::open(&args.device, &fs::read(&args.configuration)?)?;
    runtime.load(
        &application,
        &fs::read(args.sdk.join("bin/ipu/tile_bootloader_cc_ipu21.elf"))?,
        application.host_exchange.startup_mark,
    )?;
    let mut session = runtime.host_session(&application)?;
    session.start()?;
    if let Some(init) = application
        .host_exchange
        .calls
        .iter()
        .find(|c| c.name == "initialize")
    {
        ensure!(
            weights.len() == input_bytes(init),
            "weight size does not match package"
        );
        let initialized = session.invoke_streaming_deferred("initialize", &weights)?;
        session.collect(&initialized)?;
    }
    // Parameter upload and attachment are deliberately outside the measurement.
    session.set_poll_interval(Duration::from_micros(args.poll_us));
    ensure!(run.invocations > 0, "package has no inference invocations");
    let mut latencies = Vec::with_capacity(run.invocations as usize);
    let loop_started = Instant::now();
    for inference in 0..run.invocations {
        let started = Instant::now();
        let output = session.invoke("run", &input)?;
        latencies.push(started.elapsed());
        ensure!(
            output == expected,
            "inference {inference} output differs from saved reference run"
        );
    }
    let elapsed = loop_started.elapsed();
    for (inference, latency) in latencies.iter().enumerate() {
        println!(
            "inference={inference} latencyUs={:.3}",
            latency.as_secs_f64() * 1e6
        );
    }
    println!(
        "inferences={} inputBytes={} outputBytes={} pollUs={} totalMs={:.3} meanMs={:.6} inferencesPerSecond={:.3}",
        run.invocations,
        input.len(),
        expected.len(),
        args.poll_us,
        elapsed.as_secs_f64() * 1e3,
        elapsed.as_secs_f64() * 1e3 / f64::from(run.invocations),
        f64::from(run.invocations) / elapsed.as_secs_f64(),
    );
    println!("hostExchangeReplay=PASS");
    Ok(())
}

fn input_bytes(call: &ipu_package::HostCall) -> usize {
    call.inputs
        .iter()
        .map(|s| (s.file_offset + s.size) as usize)
        .max()
        .unwrap_or(0)
}
