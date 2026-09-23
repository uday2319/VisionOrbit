"""Shared land-cover text vocabulary and the pure text helpers over it (brief §2A, §2B).

Both region grounding (:mod:`app.services.grounding`) and single-image question answering
(:mod:`app.services.vqa`) turn a natural-language query into a land-cover request against the
*same* four classes this repo can actually measure — water, vegetation, built-up land and bare
soil — and refuse the *same* things it cannot: object classes it has no detector for, and spatial
relations it cannot evaluate. Keeping that vocabulary in one module makes the two capabilities
answer consistently and, more importantly, refuse consistently: a query rejected as "no detector
for ships" by grounding is rejected for the same reason, in the same words, by VQA.

No language model is involved anywhere here. A query either contains a phrase this repo has an
analyser for, or it is refused. That is what makes the refusals trustworthy — they follow from the
absence of a detector, not from a sampled token — and it is why the vocabulary is deterministic,
ordered, and matched longest-phrase-first.
"""
from __future__ import annotations

import re

from ..core.types import LandCoverClass

# --------------------------------------------------------------------------------------------
# Land-cover targets
#
# Ordered so that longest-match-first and a deliberate class ordering both hold: bare soil is
# listed before vegetation so "fallow field" is not read as a crop. Every phrase maps to one of
# the four classes the classifier produces; nothing maps to a class this repo cannot measure.
# --------------------------------------------------------------------------------------------

LAND_COVER_TARGETS: tuple[tuple[str, LandCoverClass], ...] = (
    # --- water -------------------------------------------------------------------------------
    ("water body", LandCoverClass.WATER),
    ("water bodies", LandCoverClass.WATER),
    ("waterbody", LandCoverClass.WATER),
    ("waterbodies", LandCoverClass.WATER),
    ("water", LandCoverClass.WATER),
    ("lake", LandCoverClass.WATER),
    ("lakes", LandCoverClass.WATER),
    ("pond", LandCoverClass.WATER),
    ("ponds", LandCoverClass.WATER),
    ("reservoir", LandCoverClass.WATER),
    ("reservoirs", LandCoverClass.WATER),
    ("river", LandCoverClass.WATER),
    ("rivers", LandCoverClass.WATER),
    ("stream", LandCoverClass.WATER),
    ("streams", LandCoverClass.WATER),
    ("canal", LandCoverClass.WATER),
    ("canals", LandCoverClass.WATER),
    ("lagoon", LandCoverClass.WATER),
    ("lagoons", LandCoverClass.WATER),
    ("sea", LandCoverClass.WATER),
    ("ocean", LandCoverClass.WATER),
    ("flooded area", LandCoverClass.WATER),
    ("flooded areas", LandCoverClass.WATER),
    ("flooding", LandCoverClass.WATER),
    ("flood", LandCoverClass.WATER),
    # --- bare soil, listed before vegetation so "fallow field" is not read as a crop ---------
    ("bare soil", LandCoverClass.BARE_SOIL),
    ("bare ground", LandCoverClass.BARE_SOIL),
    ("bare earth", LandCoverClass.BARE_SOIL),
    ("bare land", LandCoverClass.BARE_SOIL),
    ("bare area", LandCoverClass.BARE_SOIL),
    ("bare areas", LandCoverClass.BARE_SOIL),
    ("open ground", LandCoverClass.BARE_SOIL),
    ("barren land", LandCoverClass.BARE_SOIL),
    ("barren", LandCoverClass.BARE_SOIL),
    ("fallow field", LandCoverClass.BARE_SOIL),
    ("fallow fields", LandCoverClass.BARE_SOIL),
    ("fallow", LandCoverClass.BARE_SOIL),
    ("ploughed field", LandCoverClass.BARE_SOIL),
    ("ploughed fields", LandCoverClass.BARE_SOIL),
    ("plowed field", LandCoverClass.BARE_SOIL),
    ("plowed fields", LandCoverClass.BARE_SOIL),
    ("unvegetated", LandCoverClass.BARE_SOIL),
    ("soil", LandCoverClass.BARE_SOIL),
    ("sand", LandCoverClass.BARE_SOIL),
    ("sandy", LandCoverClass.BARE_SOIL),
    ("desert", LandCoverClass.BARE_SOIL),
    ("dirt", LandCoverClass.BARE_SOIL),
    ("dry land", LandCoverClass.BARE_SOIL),
    # --- vegetation --------------------------------------------------------------------------
    ("vegetation", LandCoverClass.VEGETATION),
    ("vegetated", LandCoverClass.VEGETATION),
    ("vegetated area", LandCoverClass.VEGETATION),
    ("vegetated areas", LandCoverClass.VEGETATION),
    ("forest", LandCoverClass.VEGETATION),
    ("forests", LandCoverClass.VEGETATION),
    ("forested", LandCoverClass.VEGETATION),
    ("woodland", LandCoverClass.VEGETATION),
    ("tree", LandCoverClass.VEGETATION),
    ("trees", LandCoverClass.VEGETATION),
    ("crop", LandCoverClass.VEGETATION),
    ("crops", LandCoverClass.VEGETATION),
    ("cropland", LandCoverClass.VEGETATION),
    ("farmland", LandCoverClass.VEGETATION),
    ("farm", LandCoverClass.VEGETATION),
    ("farms", LandCoverClass.VEGETATION),
    ("field", LandCoverClass.VEGETATION),
    ("fields", LandCoverClass.VEGETATION),
    ("plantation", LandCoverClass.VEGETATION),
    ("plantations", LandCoverClass.VEGETATION),
    ("orchard", LandCoverClass.VEGETATION),
    ("orchards", LandCoverClass.VEGETATION),
    ("grass", LandCoverClass.VEGETATION),
    ("grassland", LandCoverClass.VEGETATION),
    ("foliage", LandCoverClass.VEGETATION),
    ("greenery", LandCoverClass.VEGETATION),
    ("green area", LandCoverClass.VEGETATION),
    ("green areas", LandCoverClass.VEGETATION),
    # --- built-up ----------------------------------------------------------------------------
    ("built up area", LandCoverClass.BUILT_UP),
    ("built up areas", LandCoverClass.BUILT_UP),
    ("built up", LandCoverClass.BUILT_UP),
    ("builtup", LandCoverClass.BUILT_UP),
    ("building", LandCoverClass.BUILT_UP),
    ("buildings", LandCoverClass.BUILT_UP),
    ("urban area", LandCoverClass.BUILT_UP),
    ("urban areas", LandCoverClass.BUILT_UP),
    ("urban", LandCoverClass.BUILT_UP),
    ("settlement", LandCoverClass.BUILT_UP),
    ("settlements", LandCoverClass.BUILT_UP),
    ("city", LandCoverClass.BUILT_UP),
    ("cities", LandCoverClass.BUILT_UP),
    ("town", LandCoverClass.BUILT_UP),
    ("towns", LandCoverClass.BUILT_UP),
    ("village", LandCoverClass.BUILT_UP),
    ("villages", LandCoverClass.BUILT_UP),
    ("house", LandCoverClass.BUILT_UP),
    ("houses", LandCoverClass.BUILT_UP),
    ("housing", LandCoverClass.BUILT_UP),
    ("rooftop", LandCoverClass.BUILT_UP),
    ("rooftops", LandCoverClass.BUILT_UP),
    ("roof", LandCoverClass.BUILT_UP),
    ("roofs", LandCoverClass.BUILT_UP),
    ("construction", LandCoverClass.BUILT_UP),
    ("developed area", LandCoverClass.BUILT_UP),
    ("developed areas", LandCoverClass.BUILT_UP),
    ("developed land", LandCoverClass.BUILT_UP),
    ("industrial area", LandCoverClass.BUILT_UP),
    ("industrial areas", LandCoverClass.BUILT_UP),
    ("residential area", LandCoverClass.BUILT_UP),
    ("residential areas", LandCoverClass.BUILT_UP),
    ("infrastructure", LandCoverClass.BUILT_UP),
)

# Object classes users plausibly ask for that this repo has no detector for. Matching any of
# them to the nearest land-cover class would be a fabrication: a car park is not "built-up
# land" in any sense the user meant, and a count of ships cannot come out of a backscatter
# threshold. Each is refused by name so the message can say what is missing rather than
# "unsupported query".
UNSUPPORTED_OBJECTS: tuple[str, ...] = (
    "road", "roads", "highway", "highways", "motorway", "street", "streets", "lane",
    "railway", "railways", "railroad", "rail line", "track", "tracks",
    "bridge", "bridges", "flyover", "overpass", "tunnel", "tunnels",
    "ship", "ships", "vessel", "vessels", "boat", "boats", "barge", "ferry",
    "aircraft", "airplane", "aeroplane", "plane", "planes", "jet", "helicopter",
    "airport", "airports", "airstrip", "runway", "runways", "helipad",
    "car", "cars", "vehicle", "vehicles", "truck", "trucks", "bus", "buses", "train",
    "parking", "car park", "parking lot", "car parks",
    "park", "parks", "playground", "golf course", "stadium", "stadiums",
    "school", "schools", "hospital", "hospitals", "church", "temple", "mosque",
    "factory", "factories", "warehouse", "warehouses", "silo", "silos",
    "port", "ports", "harbour", "harbor", "jetty", "pier", "dock", "docks",
    "dam", "dams", "bund", "embankment", "levee", "canal lock",
    "pipeline", "pipelines", "power line", "power lines", "transmission line",
    "tower", "towers", "antenna", "mast", "chimney", "windmill",
    "solar panel", "solar panels", "solar farm", "wind turbine", "wind turbines",
    "tent", "tents", "camp", "camps", "refugee camp",
    "swimming pool", "swimming pools", "well", "borewell", "quarry", "mine", "mines",
    "landslide", "landslides", "wildfire", "fire", "smoke", "cloud", "clouds",
    "person", "people", "crowd", "animal", "animals", "cattle", "livestock",
)

# Spatial relations that need one object's position relative to another. Nothing in this repo
# computes them, and quietly ignoring the relation would answer a different question from the
# one asked — "buildings near the river" would come back as every building in the scene.
SPATIAL_RELATIONS: tuple[str, ...] = (
    "near", "nearby", "next to", "beside", "adjacent to", "adjacent", "between",
    "along", "alongside", "surrounded by", "surrounding", "close to", "closest to",
    "nearest to", "nearest", "touching", "bordering", "on the bank of", "on the banks of",
    "upstream", "downstream", "north of the", "south of the", "east of the", "west of the",
)


def normalise_query(raw: str) -> str:
    """Lower-case, strip punctuation and collapse whitespace, keeping digits and decimals.

    Superscripts are folded first so that "km²" survives as "km2" instead of losing its unit
    when the non-alphanumeric sweep runs, and hyphens become spaces so "built-up" and "built up"
    are one entry in the vocabulary rather than two that could drift apart.
    """
    text = raw.lower().replace("²", "2").replace("^2", "2").replace("sq.", "sq ")
    text = re.sub(r"[^a-z0-9.]+", " ", text)
    text = re.sub(r"(?<!\d)\.(?!\d)", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def find_phrase(text: str, phrases: tuple[str, ...]) -> tuple[str, int] | None:
    """The first whole-word phrase from ``phrases`` present in ``text``, with its offset.

    Earliest position wins, and the longest phrase wins at equal position. Position first
    because the noun a question is *about* comes early in it: "how many ships are in the
    harbour" should be refused for the ships, which is what was asked for, not for the harbour.
    Length second so that a multi-word phrase is never split by the single word it starts with
    ("parking lot" rather than "parking", "north east" rather than "north").
    """
    best: tuple[int, int, str] | None = None
    for phrase in phrases:
        match = re.search(rf"\b{re.escape(phrase)}\b", text)
        if match is None:
            continue
        candidate = (match.start(), -len(phrase), phrase)
        if best is None or candidate[:2] < best[:2]:
            best = candidate
    return (best[2], best[0]) if best is not None else None


def targets_in(text: str) -> list[tuple[str, LandCoverClass, int]]:
    """Every target phrase present, longest first, with no phrase nested inside a longer one.

    Nesting has to be resolved before two classes can be compared, or "fallow field" would
    register as both bare soil and vegetation and every such query would be refused as
    ambiguous when it is in fact perfectly clear.
    """
    found: list[tuple[str, LandCoverClass, int]] = []
    for phrase, label in LAND_COVER_TARGETS:
        for match in re.finditer(rf"\b{re.escape(phrase)}\b", text):
            found.append((phrase, label, match.start()))
    found.sort(key=lambda item: (-len(item[0]), item[2]))

    kept: list[tuple[str, LandCoverClass, int]] = []
    taken: list[tuple[int, int]] = []
    for phrase, label, start in found:
        end = start + len(phrase)
        if any(start >= s and end <= e for s, e in taken):
            continue
        kept.append((phrase, label, start))
        taken.append((start, end))
    return kept
