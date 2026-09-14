//! Shared infrastructure for standalone hardware diagnostics.
use anyhow::Result;
use ipu_package::{Application, Binding, RegionSlice};
use ipu_runtime::Runtime;
use std::{fs, path::Path};

/// Keep exclusive device access until the loaded runtime has been dropped.
pub struct KernelDevice {
    runtime: Runtime,
    _lock: fs::File,
}

impl KernelDevice {
    pub fn runtime(&self) -> &Runtime {
        &self.runtime
    }

    pub fn load(
        sdk: &Path,
        configuration: &Path,
        lock: &Path,
        application: &Application,
    ) -> Result<Self> {
        let lock = fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(lock)?;
        lock.lock()?;
        let runtime = Runtime::open("/dev/ipu0", &fs::read(configuration)?)?;
        runtime.load(
            application,
            &fs::read(sdk.join("bin/ipu/tile_bootloader_cc_ipu21.elf"))?,
            application.host_exchange.startup_mark,
        )?;
        Ok(Self {
            runtime,
            _lock: lock,
        })
    }
}

/// Before/after timestamps for one case on each consecutive C600 logical tile.
pub fn cycle_binding(cases: u16, address: u32) -> Binding {
    Binding {
        name: "cycles".into(),
        dtype: "u32".into(),
        shape: vec![u32::from(cases), 2],
        slices: (0..cases)
            .map(|tile| RegionSlice {
                tile: u32::from(ipu_exchange::c600_logical_to_physical(tile)),
                tile_address: address,
                file_offset: u64::from(tile) * 8,
                size: 8,
            })
            .collect(),
    }
}
