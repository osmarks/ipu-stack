//! Embed logical layout owners onto device tiles, independently of distribution.
use super::LayoutError;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::sync::Arc;

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
