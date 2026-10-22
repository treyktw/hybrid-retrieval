//! A small synthetic corpus for the demo and the integration tests.
//!
//! Fifteen short notes on brewing coffee, attributed to two made-up handbooks
//! so provenance has something to carry. Ids are fixed so the demo prints the
//! same thing on every run.

use engine_retrieval::{Provenance, RetrievalDocument};
use serde_json::json;
use std::collections::BTreeMap;
use uuid::Uuid;

pub const HANDBOOK_SOURCE_ID: Uuid = Uuid::from_u128(0x0000_0000_0000_4000_8000_0000_0000_0a01);
pub const FIELD_GUIDE_SOURCE_ID: Uuid = Uuid::from_u128(0x0000_0000_0000_4000_8000_0000_0000_0b02);

/// `(page, topic, text)` for each note, in handbook order.
const NOTES: [(u32, &str, &str); 15] = [
    (
        3,
        "extraction",
        "Extraction is the process of dissolving flavour compounds from ground coffee into water. Acids dissolve first, then sugars, and bitter compounds dissolve last.",
    ),
    (
        4,
        "extraction",
        "Under-extracted coffee tastes sour and thin because the brew stopped before sugars dissolved. Over-extracted coffee tastes bitter and dry because too many late compounds dissolved.",
    ),
    (
        7,
        "grind",
        "Grind size controls extraction speed. A finer grind exposes more surface area, so water extracts faster and an espresso shot can turn bitter if nothing else changes.",
    ),
    (
        8,
        "grind",
        "A coarser grind slows extraction. French press uses a coarse grind and a long four minute steep so the cup is full bodied without becoming muddy.",
    ),
    (
        11,
        "water",
        "Water temperature between 90 and 96 degrees Celsius suits most brewing. Hotter water extracts faster; cooler water is gentler and can leave light roasts sour.",
    ),
    (
        12,
        "water",
        "Water hardness matters. Some minerals help extraction, but very hard water scales the machine and very soft water can taste flat.",
    ),
    (
        15,
        "ratio",
        "Brew ratio is the weight of coffee to the weight of water. Filter coffee often starts near 1 to 16, while espresso uses a much tighter ratio near 1 to 2.",
    ),
    (
        18,
        "espresso",
        "Espresso forces hot water through a compact puck at around nine bar of pressure. A typical shot runs 25 to 30 seconds.",
    ),
    (
        19,
        "espresso",
        "Channeling happens when water finds a weak path through the espresso puck. Part of the bed over-extracts while the rest under-extracts, giving a shot that is sour and bitter at once.",
    ),
    (
        22,
        "roast",
        "Light roasts keep more origin acidity and need finer grinds or hotter water to extract fully. Dark roasts extract easily and turn bitter quickly.",
    ),
    (
        25,
        "storage",
        "Store whole beans in an airtight container away from light and heat. Ground coffee goes stale within hours because of the larger exposed surface.",
    ),
    (
        5,
        "pour-over",
        "Pour-over brewing starts with a bloom: a small pour that lets trapped carbon dioxide escape so later water can reach the grounds evenly.",
    ),
    (
        9,
        "cold-brew",
        "Cold brew steeps coarse grounds in cold water for twelve to eighteen hours. Low temperature extracts fewer bitter compounds, so the result is smooth and sweet.",
    ),
    (
        14,
        "tasting",
        "When tasting, separate bitterness from astringency. Bitterness is a taste on the tongue; astringency is a drying feeling, often from over-extraction.",
    ),
    (
        17,
        "maintenance",
        "Backflush an espresso machine weekly and descale it according to water hardness. Old coffee oils in the group head add a stale bitter note to every shot.",
    ),
];

pub fn note_id(index: usize) -> Uuid {
    Uuid::from_u128(0x0000_0000_0000_4000_9000_0000_0000_0000 + index as u128 + 1)
}

/// The corpus as retrieval documents with no vectors yet; indexing embeds
/// them. The first eleven notes come from the handbook, the rest from the
/// field guide.
pub fn sample_documents() -> Vec<RetrievalDocument> {
    NOTES
        .iter()
        .enumerate()
        .map(|(index, (page, topic, text))| {
            let (source_id, source_title) = if index < 11 {
                (HANDBOOK_SOURCE_ID, "Home Brewing Handbook")
            } else {
                (FIELD_GUIDE_SOURCE_ID, "Cafe Field Guide")
            };
            RetrievalDocument {
                id: note_id(index),
                content: (*text).to_owned(),
                provenance: Provenance {
                    source_id,
                    source_title: Some(source_title.into()),
                    source_kind: Some("pdf".into()),
                    uri: None,
                    location: Some(json!({ "page": page })),
                },
                metadata: BTreeMap::from([("topic".into(), json!(topic))]),
                vector: Vec::new(),
                token_count: None,
            }
        })
        .collect()
}
