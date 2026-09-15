//! Provenance for replayable choices on family-created work. Roles are named by
//! the constructor; coordinates identify algorithmic blocks, never arena IDs or
//! the number of previously emitted operations.
use crate::graph::OperationId;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct LocalSite {
    pub role: String,
    pub coordinates: Vec<u32>,
}

impl From<&str> for LocalSite {
    fn from(role: &str) -> Self {
        Self {
            role: role.into(),
            coordinates: Vec::new(),
        }
    }
}

impl LocalSite {
    pub(crate) fn at(mut self, coordinate: u32) -> Self {
        self.coordinates.push(coordinate);
        self
    }

    pub(crate) fn child(&self, role: &str) -> Self {
        Self {
            role: format!("{}/{role}", self.role),
            coordinates: self.coordinates.clone(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct WorkSite {
    pub source: OperationId,
    pub local: LocalSite,
}

/// A result slot is part of its compute/copy family's contract, rather than an
/// arena index or the number of operations previously emitted.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ResultSite {
    pub work: WorkSite,
    pub result: u32,
}

/// JSON object keys cannot represent structured work identities. Keep their
/// fields intact as key/value pairs, rejecting ambiguous duplicate requests.
pub(super) mod map {
    use serde::{Deserialize, Deserializer, Serialize, Serializer, de::Error};
    use std::collections::BTreeMap;

    pub fn serialize<K: Serialize, T: Serialize, S: Serializer>(
        values: &BTreeMap<K, T>,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        values.iter().collect::<Vec<_>>().serialize(serializer)
    }

    pub fn deserialize<
        'de,
        K: Deserialize<'de> + Ord,
        T: Deserialize<'de>,
        D: Deserializer<'de>,
    >(
        deserializer: D,
    ) -> Result<BTreeMap<K, T>, D::Error> {
        let mut result = BTreeMap::new();
        for (site, value) in Vec::<(K, T)>::deserialize(deserializer)? {
            if result.insert(site, value).is_some() {
                return Err(D::Error::custom("duplicate work-site choice"));
            }
        }
        Ok(result)
    }
}
