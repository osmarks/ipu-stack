//! Destructive hardware check of the C600 reset/clear sequence.
//! Run with: c600-init.ipucfg an-idle.ipuexe tile_bootloader_cc_ipu21.elf
//! The package is loaded only to establish a runnable diagnostic environment.
use ipu_driver::{Device, TILE_MEMORY_SIZE};
use ipu_package::TILE_MEMORY_BASE;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    ipu_driver::block_device_interrupt_signals()?;
    let config = std::env::args()
        .nth(1)
        .ok_or("configuration file required")?;
    let device = Device::open("/dev/ipu0")?;
    device.initialize()?;
    device.replay_configuration(&std::fs::read(&config)?)?;
    let package = std::env::args().nth(2).ok_or("idle package required")?;
    let bootloader = std::env::args().nth(3).ok_or("bootloader ELF required")?;
    let application = ipu_package::Application::read(std::fs::File::open(package)?)?;
    ipu_driver::Loader::new(&device, &std::fs::read(bootloader)?)?.load(&application, 23)?;
    for write in &application.device_config_writes {
        device.write_config(write.offset, write.value)?;
    }
    let addresses = [
        TILE_MEMORY_BASE,
        0x60000,
        0x80000,
        0xa0000,
        TILE_MEMORY_BASE + TILE_MEMORY_SIZE as u32 - 4,
    ];
    for tile in [0, 1, 63, 735, 1471] {
        // Match gc-info's TDI quiescence protocol, including inactive contexts.
        device.write_config(0x30004 + u32::from(tile) * 0x40, 0x4000_007f)?;
        for address in addresses {
            device.write_tile_word_from_inactive_context(tile, 1, address, 0x7e007e00)?;
            assert_eq!(
                device.read_tile_words_from_inactive_context(tile, 1, address, 1)?,
                [0x7e007e00]
            );
        }
    }
    device.initialize()?;
    device.replay_configuration(&std::fs::read(&config)?)?;
    let start = std::time::Instant::now();
    device.reset_tile_memory(1472)?;
    println!("memory reset: {:?}", start.elapsed());
    let mut failures = 0;
    for tile in [0, 1, 63, 735, 1471] {
        device.write_config(0x30004 + u32::from(tile) * 0x40, 0x4000_007f)?;
        for address in addresses {
            let value = device.read_tile_words_from_inactive_context(tile, 1, address, 1)?[0];
            failures += usize::from(value != 0);
        }
    }
    assert_eq!(failures, 0);
    println!("poison/reset/readback PASS");
    Ok(())
}
