# Confidence is not competence

Humans treat confidence as a synonym for competence. A confident-sounding model sounds correct. This is dangerous, especially with small local models whose confidence is famously miscalibrated.

The harness must surface uncertainty as a first-class signal. Every `Variant` carries a rationale and an `Uncertainty` marker, so the human's smell test has something to fire on. Refusing to surface uncertainty is a design defect, not a UX simplification. A confident-sounding output with no uncertainty markers is a primitive-contract violation.

The `Uncertainty` dataclass:

- **`flags`** — load-bearing. Known failure modes the model acknowledges ("citation may be hallucinated", "outside training cutoff"). UI surfaces these prominently.
- **`verification_hooks`** — load-bearing. What the human should check ("verify the price", "confirm the date"). Cheap-verification handoffs.
- **`confidence`** — auxiliary. 0.0–1.0 self-reported. Treated as a research artifact, not a load-bearing signal. Logged but not used to gate decisions, because small-model self-confidence is famously miscalibrated.
- **`relative_rank`** — optional rank within a batch (1 = highest). More informative than absolute confidence when available, because comparative judgments tend to be better calibrated than absolute ones.

Calibration capture starts on day one: every user pick / reject / edit on every variant is logged in machine-readable form. Over time this produces a labeled dataset (model self-confidence claim → human-validated outcome) that a downstream Context-layer study analyses for systematic miscalibration patterns.
