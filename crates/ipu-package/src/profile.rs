//! Profile records shared by application plans and measured execution reports.

use super::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProfileStepKind {
    Exchange,
    Compute,
    Synchronization,
    Idle,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProfileExchangeActivityKind {
    Send,
    Receive,
    PartnerBusy,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProfileExchangeActivity {
    /// Number of destination tiles; zero means unknown in older profiles.
    pub fanout: u16,
    pub paired: bool,
    pub kind: ProfileExchangeActivityKind,
    /// Estimated event-cycle offset within the exchange phase.
    pub start_cycle: u32,
    /// Estimated event-cycle offset within the exchange phase.
    pub end_cycle: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProfileStep {
    pub local_index: u32,
    pub phase: u32,
    pub epoch: u32,
    pub operation: String,
    pub kind: ProfileStepKind,
    pub kernel: String,
    pub metadata: Vec<ProfileMetadata>,
    pub exchange_activities: Vec<ProfileExchangeActivity>,
    pub exchange_event_cycles: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProfileMetadata {
    pub name: String,
    pub value: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CycleSample {
    pub step: ProfileStep,
    pub start_cycle: u32,
    pub end_cycle: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TileProfile {
    pub physical_tile: u32,
    pub samples: Vec<CycleSample>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProfileReport {
    pub clock_hz: u64,
    pub tiles: Vec<TileProfile>,
}

impl ProfileReport {
    pub fn write(&self, mut output: impl Write) -> Result<(), PackageError> {
        let mut message = message::Builder::new_default();
        let mut root = message.init_root::<profile_capnp::profile::Builder>();
        root.set_schema_version(5);
        root.set_clock_hz(self.clock_hz);
        let mut tiles = root.reborrow().init_tiles(self.tiles.len() as u32);
        for (tile_index, tile) in self.tiles.iter().enumerate() {
            let mut output_tile = tiles.reborrow().get(tile_index as u32);
            output_tile.set_physical_tile(tile.physical_tile);
            let mut samples = output_tile
                .reborrow()
                .init_samples(tile.samples.len() as u32);
            for (sample_index, sample) in tile.samples.iter().enumerate() {
                let mut output_sample = samples.reborrow().get(sample_index as u32);
                output_sample.set_start_cycle(sample.start_cycle);
                output_sample.set_end_cycle(sample.end_cycle);
                sample.step.write(output_sample.reborrow().init_step());
            }
        }
        serialize::write_message(&mut output, &message)?;
        Ok(())
    }

    pub fn read(mut input: impl Read) -> Result<Self, PackageError> {
        let message = serialize::read_message(&mut input, capnp_reader_options())?;
        let root = message.get_root::<profile_capnp::profile::Reader>()?;
        if !matches!(root.get_schema_version(), 1..=5) {
            return Err(PackageError::Invalid(format!(
                "unsupported profile schema version {}",
                root.get_schema_version()
            )));
        }
        let tiles = root
            .get_tiles()?
            .iter()
            .map(|tile| {
                let samples = tile
                    .get_samples()?
                    .iter()
                    .map(|sample| {
                        let step = sample.get_step()?;
                        Ok(CycleSample {
                            step: ProfileStep::read(step)?,
                            start_cycle: sample.get_start_cycle(),
                            end_cycle: sample.get_end_cycle(),
                        })
                    })
                    .collect::<Result<_, PackageError>>()?;
                Ok(TileProfile {
                    physical_tile: tile.get_physical_tile(),
                    samples,
                })
            })
            .collect::<Result<_, PackageError>>()?;
        Ok(Self {
            clock_hz: root.get_clock_hz(),
            tiles,
        })
    }
}

impl ProfileStep {
    pub(super) fn write(&self, mut output: profile_capnp::profile_step::Builder<'_>) {
        output.set_local_index(self.local_index);
        output.set_phase(self.phase);
        output.set_epoch(self.epoch);
        output.set_operation(&self.operation);
        output.set_kind(match self.kind {
            ProfileStepKind::Exchange => profile_capnp::StepKind::Exchange,
            ProfileStepKind::Compute => profile_capnp::StepKind::Compute,
            ProfileStepKind::Synchronization => profile_capnp::StepKind::Synchronization,
            ProfileStepKind::Idle => profile_capnp::StepKind::Idle,
        });
        output.set_kernel(&self.kernel);
        output.set_exchange_event_cycles(self.exchange_event_cycles);
        let mut metadata = output.reborrow().init_metadata(self.metadata.len() as u32);
        for (index, entry) in self.metadata.iter().enumerate() {
            let mut output_entry = metadata.reborrow().get(index as u32);
            output_entry.set_name(&entry.name);
            output_entry.set_value(&entry.value);
        }
        let mut activities = output
            .reborrow()
            .init_exchange_activities(self.exchange_activities.len() as u32);
        for (index, activity) in self.exchange_activities.iter().enumerate() {
            let mut output_activity = activities.reborrow().get(index as u32);
            output_activity.set_kind(match activity.kind {
                ProfileExchangeActivityKind::Send => profile_capnp::ExchangeActivityKind::Send,
                ProfileExchangeActivityKind::Receive => {
                    profile_capnp::ExchangeActivityKind::Receive
                }
                ProfileExchangeActivityKind::PartnerBusy => {
                    profile_capnp::ExchangeActivityKind::PartnerBusy
                }
            });
            output_activity.set_start_cycle(activity.start_cycle);
            output_activity.set_end_cycle(activity.end_cycle);
            output_activity.set_fanout(activity.fanout);
            output_activity.set_paired(activity.paired);
        }
    }

    pub(super) fn read(
        step: profile_capnp::profile_step::Reader<'_>,
    ) -> Result<Self, PackageError> {
        Ok(ProfileStep {
            local_index: step.get_local_index(),
            phase: step.get_phase(),
            epoch: step.get_epoch(),
            operation: step.get_operation()?.to_str()?.into(),
            kind: match step.get_kind()? {
                profile_capnp::StepKind::Exchange => ProfileStepKind::Exchange,
                profile_capnp::StepKind::Compute => ProfileStepKind::Compute,
                profile_capnp::StepKind::Synchronization => ProfileStepKind::Synchronization,
                profile_capnp::StepKind::Idle => ProfileStepKind::Idle,
            },
            kernel: step.get_kernel()?.to_str()?.into(),
            metadata: step
                .get_metadata()?
                .iter()
                .map(|entry| {
                    Ok(ProfileMetadata {
                        name: entry.get_name()?.to_str()?.into(),
                        value: entry.get_value()?.to_str()?.into(),
                    })
                })
                .collect::<Result<_, PackageError>>()?,
            exchange_activities: step
                .get_exchange_activities()?
                .iter()
                .map(|activity| {
                    Ok(ProfileExchangeActivity {
                        fanout: activity.get_fanout(),
                        paired: activity.get_paired(),
                        kind: match activity.get_kind()? {
                            profile_capnp::ExchangeActivityKind::Send => {
                                ProfileExchangeActivityKind::Send
                            }
                            profile_capnp::ExchangeActivityKind::Receive => {
                                ProfileExchangeActivityKind::Receive
                            }
                            profile_capnp::ExchangeActivityKind::PartnerBusy => {
                                ProfileExchangeActivityKind::PartnerBusy
                            }
                        },
                        start_cycle: activity.get_start_cycle(),
                        end_cycle: activity.get_end_cycle(),
                    })
                })
                .collect::<Result<_, PackageError>>()?,
            exchange_event_cycles: step.get_exchange_event_cycles(),
        })
    }
}
