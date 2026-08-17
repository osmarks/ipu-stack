use ipu_driver::{Device, DriverError, HostSession, Loader, block_device_interrupt_signals};
use ipu_package::Application;
use tracing_subscriber::EnvFilter;

pub type Result<T> = std::result::Result<T, DriverError>;

/// Thin ownership wrapper around an initialized device.
///
/// Packaging and code generation deliberately live in separate crates. The
/// runtime only initializes hardware, loads an application, and opens host
/// exchange sessions.
pub struct Runtime {
    device: Device,
}

impl Runtime {
    pub fn open(device_path: &str, configuration: &[u8]) -> Result<Self> {
        block_device_interrupt_signals()?;
        let device = Device::open(device_path)?;
        device.initialize()?;
        device.replay_configuration(configuration)?;
        Ok(Self { device })
    }

    pub fn load(
        &self,
        application: &Application,
        bootloader: &[u8],
        final_mark: u32,
    ) -> Result<()> {
        Loader::new(&self.device, bootloader)?.load(application, final_mark)?;
        for write in &application.device_config_writes {
            self.device.write_config(write.offset, write.value)?;
        }
        Ok(())
    }

    pub fn host_session(&self, application: &Application) -> Result<HostSession<'_>> {
        Ok(HostSession::new(
            &self.device,
            application.host_exchange.clone(),
        )?)
    }

    pub fn device(&self) -> &Device {
        &self.device
    }
}

pub fn init_tracing() {
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    let builder = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(false);
    if std::env::var("IPU_LOG_FORMAT").as_deref() == Ok("json") {
        builder.json().try_init().ok();
    } else {
        builder.try_init().ok();
    }
}
