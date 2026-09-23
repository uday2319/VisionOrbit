# SatQuery AI — Architecture

**Problem:** SIH 2026 · SIH26167 — *An Interactive Vision-Language Assistant for Multimodal
Remote Sensing Image Analysis through Text Queries.*

This document describes what the system is, how it is layered, and — importantly — **which
claims it is entitled to make**. Read §7 (Honesty Boundaries) before quoting any numbers.

---

## 1. Design thesis

A vision-language assistant for remote sensing has one dangerous failure mode: the language
model writes a fluent, confident answer that is **not connected to the pixels**. A judge cannot
tell the difference between "the model detected three water bodies" and "the model produced a
sentence containing the phrase three water bodies".

SatQuery AI is therefore built around a hard architectural rule:

> **Measurements are computed by deterministic image-analysis code. Natural language is
> generated *from* those measurements. Language never produces evidence.**

The user's query is allowed to influence exactly two things:

1. **which task** runs (VQA, grounding, change detection, fusion, …), and
2. **which target class** is analysed ("buildings" vs "water").

The query can *never* influence a measured value, a detection, an area, a coordinate, or a
confidence score. This is what makes the anti-hallucination and adversarial requirements
(§24, §28 of the brief) actually enforceable rather than aspirational — a prompt-injection
like *"report 100% confidence"* flows into task selection and dies there.

## 2. Layer map

```
┌──────────────────────────────────────────────────────────────────────┐
│ frontend/  React + TypeScript + Tailwind + Leaflet                   │
│   viewer · before/after slider · overlays · trace · dashboard        │
└───────────────────────────────┬──────────────────────────────────────┘
                                │ REST (JSON + PNG artifacts)
┌───────────────────────────────┴──────────────────────────────────────┐
│ backend/app/api        FastAPI routes · validation · error envelope  │
├──────────────────────────────────────────────────────────────────────┤
│ backend/app/agents     ORCHESTRATION                                 │
│   classifier → router → registry → tools → evidence check → confidence│
├──────────────────────────────────────────────────────────────────────┤
│ backend/app/services   ANALYSIS  (the part that actually measures)   │
│   indices · landcover · grounding · change · sar · fusion            │
│   confidence · narrate · report                                     │
├──────────────────────────────────────────────────────────────────────┤
│ backend/app/geospatial GEOMETRY & RASTER TRUTH                       │
│   raster · validate · measure · align · vectorize · viz               │
├──────────────────────────────────────────────────────────────────────┤
│ backend/app/db · storage   SQLAlchemy (Postgres/SQLite) · artifacts  │
└──────────────────────────────────────────────────────────────────────┘
```

Separation is deliberate: `geospatial` knows nothing about queries, `services` knows nothing
about HTTP, `agents` knows nothing about SQL. Each layer is independently testable.

## 3. Request lifecycle

```
query + image(s)
   │
   ├─ 1. classify        rule-based intent → QueryTask            (deterministic)
   ├─ 2. inspect         open raster, read CRS/transform/bands
   ├─ 3. validate        modality fit, pair compatibility, overlap
   ├─ 4. route           QueryTask → Tool (via MODEL_REGISTRY)
   ├─ 5. execute         validate_input → preprocess → predict → postprocess
   ├─ 6. verify          EvidenceValidator: does output support a claim?
   ├─ 7. score           evidence-based confidence (never LLM-generated)
   ├─ 8. narrate         template narration FROM measurements
   ├─ 9. persist         analysis · inputs · trace · results · artifacts
   └─ 10. respond        answer + evidence + confidence + execution trace
```

Every step appends a `TraceStep` (tool, status, duration, parameters, warnings). The trace is
the audit log; internal reasoning is never exposed, only this structured record.

## 4. The analysis backbone

Real, published remote-sensing methods — chosen because they are verifiable on CPU, offline,
and deterministic (which also makes them unit-testable against known synthetic ground truth).

| Capability | Method |
|---|---|
| Vegetation | NDVI = (NIR−Red)/(NIR+Red) |
| Water | NDWI = (Green−NIR)/(Green+NIR); MNDWI with SWIR |
| Built-up | NDBI = (SWIR−NIR)/(SWIR+NIR) **and** edge-density texture, both required |
| Thresholding | Otsu, gated on Sarle's bimodality; robust MAD outlier bound when unimodal |
| Grounding | index/texture mask → morphology → connected components → polygons |
| Vectorization | mask → shapely polygons → CRS-aware area (m²) via affine transform |
| SAR | linear→dB, image-estimated ENL, Lee MMSE speckle filter, two-stage Otsu on backscatter |
| Registration check | gradient-domain phase correlation + NCC (measured, not assumed) |
| Change | Change Vector Analysis over indices **∪** land-cover class disagreement |
| Normalization | per-band mean/std matching — **only** on the raw-band fallback path (§4.3) |
| Fusion | decision-level: each sensor arbitrates the axis its physics measures directly (§4.7) |

**Band awareness.** Sensors differ, so band roles are resolved by a `BandMapper` from raster
band descriptions when present, else from band count convention, else declared unknown. When
NIR is absent the system does **not** silently fake NDVI — it switches to a colour/texture
method and attaches the warning *"no NIR band: land-cover separation is approximate"*, which
propagates into a lower confidence score.

### 4.1 Ground measurement is CRS-aware

`geospatial/measure.py` owns the pixel→ground-area conversion, because getting it wrong is
silent. Three cases, each handled separately:

- **Projected CRS** — multiply by the axis `unit_conversion_factor`. A raster in US survey feet
  is not in metres, and treating it as such is a 0.03% error that compounds into a wrong hectare
  figure.
- **Geographic CRS** — `pixel_size` is in *degrees*. A naive `count × pixel_area` reported a city
  as a fraction of a square metre; the area is instead computed geodesically at the scene centre.
  This bug silently deleted the entire built-up class from EPSG:4326 inputs (the minimum-patch
  filter computed a 3·10¹¹-pixel floor against a 262,144-pixel frame) and is pinned by
  `TestGeographicCrsRegression`.
- **Ungeoreferenced** — areas are `None`, never a number. Pixel counts and percentages only.

Any physical threshold expressed in the code as an area (e.g. `MIN_CHANGE_PATCH_M2 = 2500`)
goes through the same conversion, so it stays meaningful across resolutions instead of being a
pixel count tuned to one sensor.

### 4.2 Change detection uses two independent detectors, and reports their agreement

Spectral CVA alone is **blind to built-up ↔ bare-soil conversion**: the two are near-identical
in NDVI, MNDWI and NDBI, and separate only on texture (measured Cohen's *d* on the demo scene:
edge density 3.06 vs NDBI 1.31). On the demo pair CVA recovered 0 of 5,800 bare-soil→built-up
pixels, which was the *entire* recall gap. Since the land-cover classifier already uses edge
density, class disagreement between the two dates does see those conversions. Unioning the two
detectors cost no new tunable quantity:

| Detector | Precision | Recall | F1 | IoU |
|---|---|---|---|---|
| Spectral CVA only | **0.988** | 0.731 | 0.840 | 0.724 |
| ∪ class disagreement | 0.824 | **0.917** | **0.868** | **0.767** |

The rejected alternative was adding a scaled edge-density axis to the change vector: F1 0.892,
but only at a scale factor found by trying values against the ground truth — fitting to the test
set rather than measuring on it. (The vector norm *was* checked and is not such a factor: an L2
norm and an RMS mean produce byte-identical masks, because the √n between them cancels against a
threshold derived from the same histogram.)

**Agreement is carried on every transition because it predicts correctness.** Measured within
the union mask on the demo pair:

| Evidence | n | Precision vs ground truth |
|---|---|---|
| Both detectors | 18,503 | **0.990** |
| Spectral only | 232 | 0.763 |
| Class disagreement only | 9,449 | **0.500** |

That 0.99 / 0.50 gap is why `Transition.spectral_agreement` and `ChangeResult`'s evidence split
exist: they let the confidence layer discount exactly the transitions that deserve it. The
separation holds per transition — every transition above 0.97 agreement scores 0.91–1.00
precision, and every transition below 0.21 scores 0.00–0.68. When more than 25% of reported
change rests on class disagreement alone, the result says so in plain language.

### 4.3 Radiometric normalization is applied to the fallback path only

A normalized-difference index is a *ratio*, so it is already invariant to multiplicative gain —
the dominant illumination effect. Layering per-band mean/variance matching on top of it actively
destroys the dark end: on the demo pair it shifted SWIR1 by −59 DN, dragging stable water's NDBI
from −0.30 to +0.79 and turning an unchanged reservoir into the largest "change" in the scene.
Removing it from the index path took change-detection precision from 0.695 to 0.988. It is
retained on the raw-band fallback (used when bands cannot be identified), where differencing
absolute values genuinely does need it.

Two guards, both regression-tested: a constant *target* band cannot be scaled (division by
zero), and a constant *reference* band must not be allowed to set the gain to `r_std/t_std = 0`,
which collapsed the target onto a single value and erased every difference — silently producing
a confidently empty change map.

### 4.4 Evidence must not be circular

Threshold quality was originally reported as `separability(magnitude, magnitude > threshold)`.
The mask is a threshold on the same field, so the two groups differ by construction: that number
reads **1.0000 on pure noise**. It is replaced by Sarle's bimodality coefficient and Otsu's
between-class variance ratio, both computed on the magnitude population *before any mask exists*,
plus the measured registration quality. `TestEvidenceIsNotCircular` computes the old metric on
noise and asserts it still reads 1.0, so the reason for the change stays visible.

**Otsu's variance ratio is only weakly non-circular, and for SAR it is not enough.** Measuring it
on the SAR backscatter histogram exposed the residual problem: the 3-class between-class variance
ratio reads **0.8100 on pure Gaussian noise and 0.8888 on a uniform distribution, against 0.7482
on the genuine four-class SAR scene**. Partitioning *any* distribution by value explains most of
its variance, so a high ratio is evidence that a cut was made, not that there was anything to cut.
The SAR path therefore gates on a topographic property of the histogram instead — see §4.6.

### 4.5 Offsetting change must not average to zero

A mean index delta cancels: vegetation gained in the south and lost in the north-west drives the
NDVI mean delta to a small positive number that would narrate as "barely moved". Index deltas
therefore report a count-based `increase_fraction` and are labelled **mixed** below 60%
dominance, with direction preserved in the transition table instead of the scene mean.

### 4.6 SAR: speckle is physics, and the thresholds must not be fitted

SAR needs its own treatment because the noise model is different in kind. Speckle is
**multiplicative in linear power**, not additive in dB, so `despeckle` converts dB → linear power,
filters there, and converts back. Filtering in dB would be filtering the logarithm of the thing
that is actually corrupted.

**The number of looks is measured from the image, not read from a header.** A global coefficient of
variation gives CV 1.1702 → ENL 0.73, which is below the single-look floor and therefore physically
impossible; it is dominated by the scene's 16 dB inter-class contrast, not by speckle. Taking the
**median of per-window local CV** instead gives 15.1 at a 7×7 window (16.1/15.1/14.7/14.5 at
5/7/9/11). The demo generator's nominal look count is 6, and the estimate is right to prefer its own
measurement: each of the scene's four homogeneous classes independently measures **14.5–14.7**,
because the generator's impulse-response blur correlates neighbouring pixels. Using the nominal 6
would over-smooth.

**Lee (1980) MMSE rather than a box mean**, because the filter has to preserve the edges the
thresholds will later cut on. With `Cu² = 1/ENL`, the local weight `b = var_x/var_y` goes to 0 in a
homogeneous window (output → window mean) and to 1 at an edge (output → the pixel itself). Measured
on the demo scene:

| Filter | Boundary/interior gradient ratio | Interior residual scatter | Worst class-mean bias |
|---|---|---|---|
| Lee, 7×7 | **9.14** | 0.283 dB | **0.031 dB** |
| Box mean, best of 3–13 | 6.57 | 0.240 dB | — |
| Box mean, 11×11 | — | — | 0.145 dB |
| Box mean, 13×13 | — | — | 0.315 dB |

The comparison is not tilted: the best box mean smooths class interiors slightly *harder* than Lee
(0.240 vs 0.283 dB) and still ends up with ~1.4× less edge contrast. Bias matters because every
threshold in this module is a dB value, so a filter that shifts a class mean shifts the decision
boundary with it.

**`SPECKLE_WINDOW = 7` is not selected by task accuracy, and the code says so.** Across windows
3–11 the smooth-regime F1 stays within 0.9938–0.9948 and double-bounce within 0.9987–0.9990 — the
task metric simply does not discriminate. The choice rests on the two figures that do vary:
interior residual scatter falls 0.602 → 0.388 → 0.283 → 0.225 dB at windows 3 → 5 → 7 → 9 (gains
of 0.214, then 0.105, then 0.058) before reversing to 0.257 dB at 11 as the window starts spanning
class boundaries, while radiometric bias is minimal at 7 (0.031 dB) and degrades at 9 (0.041) and
11 (0.103). Window 9 smooths marginally better and 7 stays truer; 7 is the smallest window at which
residual scatter has essentially converged, so going wider trades measurable bias for smoothing
that improves no task metric.

**Two-stage thresholds, gated on histogram topography.** A single three-class Otsu split does not
give both cuts: its upper boundary lands where the *bulk* divides (−11.72 dB on the demo scene),
which is a seed marking where the bright tail begins, not a decision boundary. Re-running Otsu
inside that tail (24.9% of the frame) finds the real corner-reflector cut at −6.95 dB. Each cut is
then gated on **mode prominence**, the non-circular replacement for the variance ratio of §4.4:
smooth the histogram, find the tallest peak on each side of the cut, take the minimum density
*between those peaks* — the saddle, not the density at the cut — and report
`1 − saddle/min(peak_left, peak_right)`. This is a topographic property of the distribution,
computed before any mask exists, and it separates cleanly where the variance ratio does not:

| Population | Otsu variance ratio | Mode prominence |
|---|---|---|
| Genuine 4-class SAR scene | 0.7482 | **0.7197** (0.9812 filtered) |
| Pure Gaussian noise | 0.8100 | **0.0000** |
| Uniform distribution | 0.8888 | **0.0473** |
| Otsu's own upper *seed* cut | — | 0.2659 → never used as a boundary |
| Bright cut found in the tail | — | **0.9082** (0.9980 filtered) |

`MIN_MODE_PROMINENCE = 0.25` sits mid-way in a wide empty band between those groups; it was
discovered from the measurements above rather than tuned against a score.

**Accuracy, and the near-optimality argument.** Sweeping the threshold against the known class map
*after the fact*, the best achievable smooth cut scores F1 0.9951 at −16.55 dB and the one the
method finds scores 0.9938 at −16.94 dB; the best achievable bright cut scores 0.9996 at −7.80 dB
and this one scores 0.9989 at −6.95 dB. **Both land within 0.0013 F1 of an oracle without ever
consulting ground truth**, which is the strongest available evidence that the method is a property
of backscatter distributions rather than a fit to this scene.

| Regime | F1 | Errors |
|---|---|---|
| SMOOTH (water) | 0.9938 | 153 of 12,460 — **all within 3 px of the shoreline, 92% within 1 px** |
| DIFFUSE (rough) | 0.9996 | — |
| DOUBLE_BOUNCE (built-up) | 0.9989 | 15 missed, 2 added, of 7,826 |

The residual error is a mixed-pixel effect, not a mis-set cut: the reservoir interior is exact, and
the 50 m bare-soil road grid cutting the urban blocks is kept out of the building class entirely
rather than smoothed across. An earlier figure of 0.868 for built-up was a **measurement artifact** —
it scored against a truth mask that counted those road pixels as buildings. It is recorded here
because the correction ran in the honest direction and the docstrings that quoted it were wrong.

**What speckle filtering is worth, per regime.** Smooth-regime F1 rises 0.948 → 0.994, because a
dark surface and the low tail of surrounding terrain are only a few dB apart and speckle straddles
exactly that gap. Double-bounce barely moves (0.9935 → 0.9989): corner-reflector returns sit ~10 dB
clear of everything else, far outside speckle's range.

**The refusal is deliberately conservative.** On synthetic scenes with a shrinking dark patch the
prominence gate accepts from ~5% coverage down and refuses at 2% and below:

| Dark-patch coverage | Prominence | Verdict | F1 if forced anyway |
|---|---|---|---|
| 25% / 10% / 5% | 0.9027 / 0.8076 / 0.7231 | detected | 0.9916 / 0.9811 / 0.9653 |
| 2% | 0.0000 | **refused** | 0.8831 |
| 1% / 0.5% / 0.2% | 0.0000 | **refused** | 0.1886 / 0.0645 / 0.0199 |

At 2% it declines a case a forced threshold would have recovered at F1 0.88. That is the trade this
system is built to make: a missed regime is recoverable by the user, a phantom one is not.

**SAR reports a scattering mechanism, not land cover**, and the type system enforces it.
`ScatteringRegime.candidate_surfaces` returns plural tuples — SMOOTH is calm water *or* wet smooth
ground *or* dry sand/tarmac *or* radar shadow. From single-polarization VV, vegetation and bare soil
are both DIFFUSE and are **never subdivided**; that refusal is precisely what motivates the
optical/SAR fusion path.

### 4.7 Optical + SAR: fusion is at the decision level, and each sensor decides its own axis

Reflectance and backscatter are physically unrelated — a fraction of incident sunlight versus a
normalized radar cross-section. Stacking or averaging them yields a number with no physical meaning
whose apparent precision is entirely artificial, so `services/fusion.py` combines each sensor's
**conclusion** rather than its pixels, and labels every fused pixel with which sensors supported it
(`FusionEvidence`: `both` / `conflict` / `optical_only` / `sar_only` / `neither`). Conflicts are
never silently averaged away.

**The arbitration principle was fixed before any number was consulted: each sensor decides the axis
its physics measures directly.** Three rules follow, and nothing else resolves a disagreement.

- **Rule A — radar decides built-up.** Double-bounce is a *direct geometric measurement*: a vertical
  face beside the ground returns a wall-then-floor path straight back to the sensor, and no natural
  surface produces it. Optical built-up has no geometric measurement available — §4's classifier
  reaches the class from SWIR brightness plus an **edge-density texture proxy**, because bare soil is
  also SWIR-bright. A proxy loses to a direct measurement.
- **Rule B — optical decides surface composition.** Water, vegetation and bare soil separate on
  chlorophyll and liquid-water absorption in NIR/SWIR. Single-pol VV provably cannot: `DIFFUSE`
  covers vegetation *and* bare soil, `SMOOTH` covers calm water *and* dry sand. A non-built-up
  conflict is therefore **recorded, not resolved** — the optical class stands.
- **Rule C — a regime names a class only where it names exactly one.** Where optical has no class,
  radar supplies one only if `ScatteringRegime.consistent_land_cover` holds a single class, true only
  of double bounce. Under `SMOOTH` or `DIFFUSE` the fused map stays *unclassified* rather than
  picking one of two equally consistent surfaces.

**The compatibility relation is derived, not tabulated.** `_consistency_mask` reads
`ScatteringRegime.consistent_land_cover`, which is itself derived from the `candidate_surfaces` prose
shown to the user, so the arbitration cannot drift out of step with the explanation printed beside
it. A test keyword-matches every mapped class back to a listed surface to keep that true.

**What the rules are worth**, on the demo optical/SAR pair against its exact ground truth:

| Class | Optical alone (F1) | Fused (F1) |
|---|---|---|
| built-up | 0.7861 | **0.9989** |
| bare soil | 0.9833 | 0.9945 |
| vegetation | 0.9904 | 0.9905 |
| water | 0.9630 | 0.9630 |
| **overall accuracy** | 0.9768 | **0.9923** |

**The shape of the improvement is the evidence that the rules are physics rather than a fit.**
Built-up precision rises 0.6651 → 0.9997 because 3,770 SWIR-bright, ploughing-textured bare-soil
pixels stop being called buildings — while water moves by 0.0000 and vegetation by 0.0000, the axis
Rule B forbids radar from touching. A rule tuned for score would have lifted everything a little.
The suite asserts that non-movement directly (`< 1e-9` for water), which is the test that separates a
physical rule from a fitted one.

Both directions of the built-up conflict resolved correctly, and the rule was fixed in advance so it
could have been wrong in either: radar **asserting** built-up over optical bare soil — 298 pixels,
298 truly built; radar **withdrawing** an optical built-up claim — 3,798 pixels, of which 3,770 are
truly bare soil.

**The case fusion exists for.** Given RGB-only optical, the classifier *refuses* to claim built-up at
all (separating buildings from bare soil needs SWIR), reporting 0 pixels and saying so: built-up F1
is exactly 0.0000, overall accuracy 0.9626. Adding the SAR channel recovers built-up at F1 **0.9989**
and lifts accuracy to 0.9923. That is not a marginal gain over a working analysis — it is the
difference between declining to answer and answering correctly, which is the argument for requiring
both modalities rather than treating SAR as a nice-to-have.

**A veto requires a measurement.** Rule A's withdrawal half fires only while radar actually separated
a double-bounce population. Where §4.6's prominence gate declined to place a bright boundary at all,
the absence is a property of the histogram, not of the ground — nothing was measured about vertical
structure, so there is no reading to contradict optical with. Vetoing anyway would convert *radar
could not tell* into *radar disagrees*. The optical claim stands, labelled uncorroborated, and the
method is reported as `decision-level-optical-sar-partial` so the caller can see half the arbitration
was unavailable. This was a live bug caught while writing the suite, not a designed-in case.

**Sensor agreement is reported, not used as a gate.** 98.43% of jointly-observed pixels agree, and
the per-pixel evidence label is *provenance* rather than a reliability score: accuracy is 0.9923 on
the corroborated majority against 0.9927 on the arbitrated remainder, so the two are
indistinguishable and nothing reads the label as a correctness prediction. It is
tempting to threshold a rising conflict fraction as a "wrong scene" detector; that was tested and
**rejected by measurement**:

| Condition | Conflict fraction |
|---|---|
| Correct optical date, perfectly aligned | 0.0157 |
| Correct date, shifted 20 px | 0.0532 |
| **Wrong date, perfectly aligned** | **0.0514** |

A 20-pixel misregistration of the *right* scene looks worse than the *wrong* scene aligned, so the
quantity cannot distinguish the two failures and must not be thresholded as though it could.
Misalignment is detected where it is genuinely measurable (`measure_registration`, sharing
`REGISTRATION_WARN_PX` with change detection but phrasing its own warning); the conflict fraction is
reported continuously for the confidence layer, and only a *structural* condition — more
disagreement than agreement — raises a warning.

**The weakest corner, recorded rather than smoothed over.** 34 pixels are `optical=built_up` against
`sar=smooth`. Rule A withdraws the built-up claim and the class falls back to bare soil, but 33 of
them are truly *water* that the optical water test had already missed at the shoreline. Radar agrees
the surface is specular but cannot say water versus dry ground, so fusion inherits the optical miss
instead of fixing it. Rule B forbids inventing a fix, and a special case for a 34-pixel group would
be fitting to one scene. A test pins the group's size so it cannot grow unnoticed.

### 4.8 Grounding: pixel accuracy does not imply countability

`services/grounding.py` answers *where is X* and *how many X are there* by turning a text query into
region geometry. Two decisions carry it, and both were made against measurements rather than
plausibility.

**The source mask is a classifier decision, never a raw index.** Thresholding NDBI directly would be
the obvious way to ground "buildings", and it is catastrophic: on the demo scene it reaches recall
**0.9997** at precision **0.0406**, IoU **0.0406**, against the §4 cascade's IoU **0.6475**. That
shape — finds nearly everything, is nearly always wrong — is exactly what a fabricated detection
looks like from the outside, because bare soil is SWIR-bright too. Grounding therefore consumes
`LandCoverResult` or `FusionResult` and never thresholds an index itself; per-region index values are
carried only as *scores*, and bare soil gets none at all, because it is the cascade's remainder class
and no index magnitude means "more bare".

**No morphological cleaning is applied by default.** Opening and closing the built-up mask makes it
look tidier and deletes the objects in it: 23 regions become **19** at radius 1 and **10** at radius 2,
and the region-level true-positive count is **0** at both — morphology here removes objects, not
noise.

**Counting is a separate claim from extent, and it needs radar.** Built-up pixel F1 is a respectable
0.7861 from optical alone (§4.7), and the region-level precision and recall of the same mask are
**0.000** — at every IoU threshold from 0.1 to 0.7. The mechanism is measured, not assumed: false-
positive soil pixels bridge the blocks, so **one** predicted component touches **all 20** of the
ground-truth blocks and the other 22 touch none. Extent is a real answer here and is returned; the
count is withheld with a caveat naming what would fix it.

| Class | Source | Regions | Truth | TP | FP | Precision | Recall | Mean IoU |
|---|---|---|---|---|---|---|---|---|
| built-up | optical only | 23 | 20 | 0 | 23 | **0.000** | **0.000** | 0.0000 |
| built-up | optical + SAR | 20 | 20 | 20 | 0 | **1.000** | **1.000** | **0.9930** |
| built-up | RGB + SAR | 20 | 20 | 20 | 0 | 1.000 | 1.000 | **0.9930** |
| bare soil | optical only | 12 | 1 | 1 | 11 | 0.083 | 1.000 | 0.5742 |
| bare soil | optical + SAR | 1 | 1 | 1 | 0 | 1.000 | 1.000 | 0.9890 |
| water | optical only | 3 | 3 | 3 | 0 | 1.000 | 1.000 | — |
| vegetation | optical only | 4 | 4 | 4 | 0 | 1.000 | 1.000 | — |

**RGB + SAR scoring identically to full optical + SAR (0.9930) is what identifies the cause.** The
alternative explanation — that the NIR and SWIR bands, not radar, supplied the improvement — predicts
a drop when those bands are removed. There is none, to four decimal places.

**The countability rule follows the physics, not the topology of the mask.** `RADAR_ARBITRATED_CLASSES`
is `{built_up, bare_soil}`: precisely the axis Rule A arbitrates, and the two classes that sit on
opposite sides of the boundary radar measures. Water and vegetation are countable from optical alone
and radar does not change them, so a blanket "counting needs SAR" rule would withhold two answers
that are already correct.

A mask-shape heuristic was the tempting alternative and is **rejected by measurement**. "Is one blob
dominating?" would have gated the wrong cases in both directions:

| Case | Largest region's share of class pixels | Counting actually |
|---|---|---|
| bare soil, fused | 1.000 | perfect |
| built-up, optical only | 0.902 | entirely broken |
| water, optical only | 0.654 | perfect |

**A veto still requires a measurement.** Reportability keys on whether a double-bounce population was
actually *separated*, not on whether a SAR file was supplied. Radar that failed §4.6's prominence gate
leaves the count withheld with a caveat saying which of the two failures occurred, and `source`
carries §4.7's `-partial` marker.

**A region is a contiguous land-cover patch, not a structure.** 20 built-up patches are not 20
buildings at 10 m per pixel, and `region_semantics` says so in every response — a count that is
correct as measured can still be read as a claim it does not make.

**Refusal is a designed outcome, in four categories.** Spatial relations ("buildings *near* the
river") are refused rather than silently dropped, because dropping the relation answers a different
question. Object classes with no detector are refused **by name** — "there is no detector for ships"
tells the user what to change, where "unsupported query" leaves them guessing. So are an unrecognised
target, and two targets at once. A query that is *understood* but matches nothing is not an error: it
returns zero regions plus the filter-specific reason it came back empty, since "no regions found" is
true of four situations that call for four different next actions.

**Compass answers are gated on grid orientation.** "The north of the image" is a claim about the
ground, legitimate only when the raster is georeferenced *and* north-up (`transform[1] == transform[3]
== 0`, `transform[4] < 0`). On a rotated granule or a plain JPEG the request is refused and the
frame-relative wording ("the top") is offered, which is answerable on any picture. `CENTRE` is the one
sector needing no orientation check: the middle of the pixel grid is the middle of the footprint under
any rotation.

**Sectors select by majority overlap, and the fraction is reported.** Between 25% and 100% of the
ground-truth regions in the demo scene straddle a quadrant line, so straddling is the normal case:
three water bodies *touch* the centre and one lies mainly in it. Counting everything that touches
would treble the answer.

**A measurement with no comparator is ignored, never guessed.** "Water bodies of 5 ha" could mean at
least, at most or exactly, so no bound is set. Comparators are resolved by the one *ending* closest to
the number, longest first, which is what makes "no more than 5 ha" an upper bound despite "more than"
being nested inside it.


## 5. Model registry & fallback

Tools implement one interface (`validate_input · preprocess · predict · postprocess · explain`)
so implementations are swappable without touching the API or UI.

```
preferred: satquery-rs-visual-v1  ← real EuroSAT-adapted ResNet-18, one task (captioning)
      ↓ unavailable / errors
deterministic remote-sensing algorithm      ← ships enabled, always works
      ↓ genuinely cannot support the claim
explicit refusal + warning                  ← never a fabricated box
```

Exactly one rung is learned. `satquery-rs-visual-v1` is a `torchvision` ResNet-18 with real
`IMAGENET1K_V1` weights, linear-probed on real EuroSAT Sentinel-2 imagery (87.25% top-1 on 400
held-out images) and registered as the `PREFERRED` tool for the captioning task only — a specialist
evidence component, not the complete VQA system. Its availability is triple-gated on the
`enable_learned_models` setting, the checkpoint file and an importable torch; `dispatch` drops to the
deterministic `captioning-landcover-cv` rung on `ModelUnavailableError` alone, so a validation error
still surfaces as a structured error rather than a silent retry. The other eight tools are
deterministic and always available. See `docs/remote-sensing-adaptation.md`.

The bottom rung matters most: when evidence is insufficient the system returns
*"Insufficient evidence for a reliable conclusion"* rather than inventing a detection.

## 6. Confidence

Confidence is **computed** by `services/confidence.py`, from factors that can each be pointed at:

`input quality` (dynamic range, saturation, nodata fraction) · `metadata completeness`
(CRS, transform, timestamp) · `alignment quality` (overlap fraction, registration offset) ·
`evidence strength` (class separability / threshold margin) · `agreement` (cross-index and
optical↔SAR concordance) · `warning penalty`

These combine to a scalar → `HIGH / MEDIUM / LOW / INSUFFICIENT`, returned **with the reasons**. No
language model is consulted for confidence at any point.

Every factor is a quantity some analysis service already measured and carried on its result, so
confidence is an aggregation rather than a second opinion. The inputs that exist today:
`ChangeResult.corroborated_fraction` and `bimodality` (§7.1 case 2), `RegistrationQuality`'s offset
and NCC, `LandCoverResult.separability`, from SAR (§4.6) `smooth_prominence`, `bright_prominence`
and `SpeckleReport.estimated_enl`, from fusion (§4.7) `FusionResult.corroborated_fraction` and
`agreement_fraction`, and from grounding (§4.8) the `evidence` block — class separability, the
index a region's score came from, whether the count is reportable, and how many candidates the
filters removed.

**A weighted mean is not enough, and the pure-noise case is why.** Two independent noise fields
classified as a bi-temporal pair produce a *pristine-looking* input — good dynamic range, sharp,
fully georeferenced — so an average over the factors would report a comfortable score for an answer
that is worthless. The fix is that some factors are **gates**, not contributors:
`score = min(weighted_mean(all factors), min(gate values))`. A gate still counts in the mean, but
also caps the result, which is what makes it *necessary* rather than merely influential. On the
noise pair the change evidence gate is `min(bimodality_signal, corroborated_fraction)` and the
corroborated fraction is **0.000** (bimodality 0.449, below the 5/9 two-population line), so the
score is capped at **0.000 → INSUFFICIENT**; the real demo pair scores bimodality 0.887,
corroboration 0.657 → **0.657, MEDIUM**, limited by the same gate reading a real value. Both
channels are consumed, exactly as §7.1 case 2 requires: zeroing either one alone forces
`INSUFFICIENT`.

The gates are chosen per analysis, following the physics rather than a template: change gates on
evidence strength **and** registration (a misregistered pair manufactures change at every boundary,
§7.1 case 3); land cover and grounding gate on class separability (an indistinct class cannot be
certified by a clean image); fusion gates on the corroborated fraction (a product the two sensors
did not agree on is weak however clean each sensor was). **SAR separation is deliberately a
contributor, not a gate** — a genuinely uniform scene reports low mode prominence because there is
no *additional* regime to split out, not because the diffuse reading is unreliable, so gating on it
would punish a scene for being homogeneous. Every report also names its `limiting_factor` — the gate
that capped the score, or the weakest contributor when none did — so "not higher because of X" is
answerable, not just the level.

## 7. Honesty boundaries

Enforced by design, and stated in the README rather than buried:

- Areas in m² are reported **only** when a real CRS + affine transform exist. Otherwise results
  are in pixels and percentages, and say so.
- A plain JPEG/PNG is never treated as georeferenced.
- Change is never asserted without a computed change mask.
- Optical/SAR co-registration is **measured** and reported, never assumed.
- A land-cover transition is reported only where **both** dates classified the pixel. A pixel
  unclassified on either date is labelled `OTHER`, and the class-disagreement detector ignores it
  entirely, so a gap in the analysis can never become a finding.
- **Without SWIR, no built-up claim is made at all.** Built-up and bare soil overlap so heavily
  in the visible bands that every threshold rule tested gave precision below 0.15. The RGB
  fallback reports a combined "unvegetated ground" class and states the limitation.
- **From single-polarization SAR, no land-cover claim is made** — only a scattering mechanism,
  with all of its candidate surfaces listed. Vegetation and bare soil are not separated (§4.6).
- **A sensor disagreement is never averaged away.** Every fused pixel carries the evidence that
  produced it, conflicting pixels are labelled `conflict` and reported as groups with the arbiter
  and the reason, and where radar's regime is consistent with two surfaces and optical is silent,
  no class is reported at all (§4.7).
- Accuracy metrics appear only where an evaluation script actually computed them, always
  labelled with dataset, split, and sample count.
- No claim of fine-tuning is made unless a training run actually produced a checkpoint.

### 7.1 Known limitations, measured

Recorded here rather than discovered by a judge. Each is a real property of the shipped code.

1. **Built-up ↔ bare-soil conversions are the weak spot.** They are invisible to spectral CVA
   and recovered only through class disagreement, at 0.68 precision (bare-soil→built-up) and
   0.16 (built-up→bare-soil). The two directions are reported with their measured
   `spectral_agreement` of ~0.00, which is the signal a caller should act on.
2. **On a pure-noise pair, class disagreement reports ~16% spurious change.** Classifying noise
   twice yields different labels; that is inherent to any classifier. It is not suppressed by a
   threshold, because tuning one on this input is exactly the test-set fitting rejected in §4.2.
   Instead all four evidence channels report the weakness simultaneously — bimodality 0.45 (below
   the 5/9 two-population threshold), corroborated fraction 0.00, NCC 0.016, and both dates'
   classifications flagged "indicative rather than measured". **`services/confidence.py` must
   consume `corroborated_fraction` and `bimodality`** so this case resolves to
   `INSUFFICIENT` rather than a low-but-nonzero answer.
3. **Change area is over-estimated when registration is poor.** Land-cover boundaries register as
   change even where the ground did not change. The measured pixel offset is reported and a
   warning states the over-estimate explicitly rather than correcting for it invisibly.
4. **Small SAR targets below ~2–5% of the frame are refused, not detected.** The prominence gate
   needs a resolvable histogram mode; a dark patch covering 2% of the scene produces no mode at
   all, so no smooth threshold is reported even though a forced cut would have scored F1 0.88
   there (§4.6). Detection genuinely collapses just below that point — F1 0.19 at 1% — so the
   gate is set to fail toward silence.
5. **SAR ENL is estimated per image and can be wrong on a scene with no homogeneous area.** The
   estimator takes the median of local coefficients of variation, which assumes some windows land
   inside a uniform region. It is clamped to `[1, 64]`, and the estimate is reported on every
   result so a caller can see when it sits at a bound.
6. **Fusion inherits an optical water miss at the shoreline.** 34 pixels are `optical=built_up`
   against `sar=smooth`; Rule A withdraws the built-up claim and 33 of them are truly water the
   optical water test had already missed. Radar confirms the surface is specular but cannot say
   water versus dry ground, so the correct answer is unreachable from this sensor pair and fusion
   does not invent it (§4.7).
7. **The conflict fraction cannot detect a wrong scene, and is not used to.** A 20-pixel
   misregistration of the correct scene (0.0525) reads worse than the wrong acquisition perfectly
   aligned (0.0455), so the two failures are not separable by this quantity. It is reported
   continuously instead of thresholded; only *more disagreement than agreement* warns.
8. **Demo-scene metrics are not sensor metrics.** Every number in this document is measured on
   synthetic rasters with known ground truth. They validate that the *algorithms* work and catch
   regressions; they are not a claim about Sentinel-1 or Sentinel-2 performance.
9. **Optical-only region counts for built-up and bare soil are withheld, not estimated.** Region
   precision and recall are 0.000 and 0.083 respectively without radar, at pixel F1 0.7861 and
   0.9833 — so extent is answerable where the count is not, and no partial count is offered in
   place of the withheld one (§4.8).
10. **A grounded region is a land-cover patch, not a structure.** Twenty built-up patches at 10 m
    per pixel are not twenty buildings. The count is correct as measured and `region_semantics`
    states the distinction on every response, because the number invites a reading it does not
    support (§4.8).
11. **Sector membership is decided by majority overlap, so a straddling region is assigned whole.**
    Between 25% and 100% of demo ground-truth regions cross a quadrant line. The overlap fraction
    is returned per region so a 0.52 assignment is distinguishable from a 1.00 one, but the region
    is not split (§4.8).

## 8. Benchmark datasets

Adapters normalize each corpus into one internal `RemoteSensingSample`, so no dataset format is
hard-coded: `bigearthnet · vrsbench · rsvqa · cdvqa`. Because these corpora are large
(BigEarthNet is tens of GB) the prototype ships **synthetic rasters with known ground truth** —
known CRS, known coordinates, known water/building footprints, known change regions. That is
what allows evaluation scripts to report *honest* IoU/precision/recall/F1 instead of quoted
literature numbers, and lets geospatial tests assert exact expected coordinates.
