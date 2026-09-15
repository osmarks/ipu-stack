//! Embed logical layout owners onto device tiles, independently of distribution.
use super::LayoutError;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::sync::Arc;

/// Relabel physical assignments through one device permutation. Intern the
/// resulting bases so repeated domains keep sharing their embedding arrays.
pub(crate) fn remap_owners<'a>(
    owners: impl Iterator<Item = &'a mut OwnerMap>,
    mapping: &[u16],
    tile_count: u16,
) -> Result<(), LayoutError> {
    let mut sorted = mapping.to_vec();
    sorted.sort_unstable();
    if !sorted.into_iter().eq(0..tile_count) {
        return Err(LayoutError::InvalidTilePermutation { tiles: tile_count });
    }
    if mapping.iter().copied().eq(0..tile_count) {
        return Ok(());
    }
    let mapping: Arc<[u16]> = mapping.into();
    let mut bases = std::collections::HashMap::<_, _, foldhash::fast::FixedState>::default();
    for owners in owners {
        owners.validate(1, tile_count)?;
        let base = owners.with_rotation(0);
        let remapped = bases.entry(base).or_insert_with_key(|base| {
            base.remapped(&mapping)
                .expect("validated owner and device maps")
        });
        *owners = remapped.with_rotation(owners.rotation());
    }
    Ok(())
}

/// Rotate logical owner ordinals, then embed them onto device tiles. An omitted
/// embedding uses the whole device; an explicit embedding can name any subset.
/// Cloning values shares the embedding's storage. Keeping the rotation before
/// the embedding lets many values share one mapping while retaining their
/// existing relative distributions.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct OwnerMap {
    rotation: u16,
    embedding: Option<Arc<[u16]>>,
}

impl OwnerMap {
    pub const fn rotated(rotation: u16) -> Self {
        Self {
            rotation,
            embedding: None,
        }
    }

    pub fn embedded(tiles: impl Into<Arc<[u16]>>) -> Self {
        Self {
            rotation: 0,
            embedding: Some(tiles.into()),
        }
    }

    pub fn rotation(&self) -> u16 {
        self.rotation
    }

    /// A full identity embedding is the implicit device domain. Keep smaller
    /// identity subsets explicit: their rotations wrap at a different length.
    pub(crate) fn canonicalized(&self, tiles: u16) -> Self {
        if self
            .embedding
            .as_ref()
            .is_some_and(|map| map.iter().copied().eq(0..tiles))
        {
            Self::rotated(self.rotation)
        } else {
            self.clone()
        }
    }

    /// Rotate within the selected owner domain; the embedding itself is shared.
    pub(crate) fn with_rotation(&self, rotation: u16) -> Self {
        let rotation = self
            .embedding
            .as_ref()
            .filter(|map| !map.is_empty())
            .map_or(rotation, |map| (usize::from(rotation) % map.len()) as u16);
        Self {
            rotation,
            embedding: self.embedding.clone(),
        }
    }

    pub(crate) fn shifted(&self, delta: i32, tiles: u16) -> Option<Self> {
        let domain = self.domain(tiles)?;
        (domain != 0).then(|| {
            self.with_rotation(
                (i32::from(self.rotation) + delta).rem_euclid(i32::from(domain)) as u16,
            )
        })
    }

    fn domain(&self, tiles: u16) -> Option<u16> {
        self.embedding
            .as_ref()
            .map_or(Some(tiles), |map| u16::try_from(map.len()).ok())
    }

    pub(crate) fn has_embedding(&self) -> bool {
        self.embedding.is_some()
    }

    /// Interpret this local assignment within a chosen working domain. An
    /// implicit assignment rotates within that domain; an explicit one selects
    /// its entries. The selected domain may be smaller than the whole device.
    pub(crate) fn in_domain(&self, domain: &Self, tiles: u16) -> Option<Self> {
        match &self.embedding {
            None => domain.shifted(i32::from(self.rotation), tiles),
            Some(embedding) => Some(Self {
                rotation: self.rotation,
                embedding: Some(
                    embedding
                        .iter()
                        .map(|&tile| domain.tile(tile, tiles))
                        .collect::<Option<Arc<[_]>>>()?,
                ),
            }),
        }
    }

    /// Relabel physical tiles without changing distribution, domain or rotation.
    /// The caller validates the shared device permutation once.
    pub(crate) fn remapped(&self, mapping: &Arc<[u16]>) -> Option<Self> {
        Some(Self {
            rotation: self.rotation,
            embedding: Some(match &self.embedding {
                None => Arc::clone(mapping),
                Some(embedding) => embedding
                    .iter()
                    .map(|&tile| mapping.get(usize::from(tile)).copied())
                    .collect::<Option<Arc<[_]>>>()?,
            }),
        })
    }

    pub(crate) fn tile(&self, owner: u16, tiles: u16) -> Option<u16> {
        let domain = self.domain(tiles)?;
        if owner >= domain || self.rotation >= domain {
            return None;
        }
        let ordinal = ((u32::from(owner) + u32::from(self.rotation)) % u32::from(domain)) as u16;
        let tile = self
            .embedding
            .as_ref()
            .map_or(ordinal, |map| map[usize::from(ordinal)]);
        (tile < tiles).then_some(tile)
    }

    pub(crate) fn validate(&self, owners: u16, tiles: u16) -> Result<(), LayoutError> {
        let invalid = || LayoutError::InvalidOwnerMap { owners, tiles };
        let domain = self.domain(tiles).ok_or_else(invalid)?;
        if owners == 0 || tiles == 0 || owners > tiles || owners > domain || self.rotation >= domain
        {
            return Err(invalid());
        }
        if let Some(embedding) = &self.embedding {
            let mut used = BTreeSet::new();
            if embedding.len() < usize::from(owners)
                || embedding
                    .iter()
                    .any(|&tile| tile >= tiles || !used.insert(tile))
            {
                return Err(invalid());
            }
        }
        Ok(())
    }
}
