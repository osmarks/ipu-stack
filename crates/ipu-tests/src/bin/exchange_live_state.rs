use anyhow::{Context, Result};
use clap::Parser;
use ipu_driver::Device;
use ipu_target::exchange::Topology;

#[derive(Parser)]
#[command(about = "Read live IPU21 exchange state without resetting the device")]
struct Arguments {
    #[arg(long, default_value = "/dev/ipu0")]
    device: String,
    /// Logical tiles to inspect. Defaults to every tile with nonzero exchange
    /// state or an exception, plus tiles 0 through 3.
    #[arg(long)]
    tile: Vec<u16>,
    /// Also stop context zero briefly to recover its program counter.
    #[arg(long)]
    program_counter: bool,
}

fn main() -> Result<()> {
    let arguments = Arguments::parse();
    let device = Device::open(&arguments.device)?;
    let topology = Topology::c600();
    let requested = (!arguments.tile.is_empty()).then_some(arguments.tile.as_slice());
    for logical in 0..u16::try_from(topology.tile_count())? {
        if requested.is_some_and(|tiles| !tiles.contains(&logical)) {
            continue;
        }
        let physical = topology.physical(logical)?;
        let context = device
            .tile_context_state(physical, 0)
            .with_context(|| format!("read tile {logical} context state"))?;
        let error = device
            .tile_exchange_receive_error(physical)
            .with_context(|| format!("read tile {logical} ERERR"))?;
        let (incoming_dcount, exchange_control) = device
            .tile_exchange_state(physical)
            .with_context(|| format!("read tile {logical} exchange state"))?;
        let selected = requested.is_some()
            || logical < 4
            || context != 1
            || error as u8 != 0
            || incoming_dcount != 0
            || exchange_control != 0;
        if !selected {
            continue;
        }
        let pc = arguments
            .program_counter
            .then(|| device.read_tile_program_counter(physical, 0))
            .transpose()
            .with_context(|| format!("read tile {logical} PC"))?;
        let incoming_mux_pair = device
            .read_tile_special_register(physical, 0, 0xa1)
            .with_context(|| format!("read tile {logical} INCOMING_MUXPAIR"))?;
        let incoming_format = device
            .read_tile_special_register(physical, 0, 0xa3)
            .with_context(|| format!("read tile {logical} INCOMING_FORMAT"))?;
        println!(
            "logical={logical} physical={physical} context={context} ererr={error:?} dcount={incoming_dcount} exchangeCtl=0x{exchange_control:08x} incomingMuxPair={incoming_mux_pair} incomingFormat={incoming_format}{}",
            pc.map(|pc| format!(" pc=0x{pc:08x}")).unwrap_or_default()
        );
    }
    Ok(())
}
