# Creativity needs variance, not precision

Frontier models win on precision tasks: factual recall, math, code correctness, well-defined classification. Small local models lose those head-to-head: confabulation, mode collapse, narrower training distributions.

Frontier models lose on creative tasks: when 50 frontier models write a story about a boy and a dragon, there is a striking sameness across them. They converge to a safe centre. They erode variance, which is the soul of creativity.

Small local models, properly harnessed, can compete on creative tasks for the inverted reason: their imprecision *becomes variance*, and variance is what creativity needs. The design imperative is to stop trying to make small models precise and instead channel their variance through the harness.

This is the project's positioning claim. It justifies why a conversational harness on small local models is worth building rather than a frontier-API wrapper.

How the harness channels variance:

- The E primitive enforces `n >= 2`. Singular outputs are forbidden at the primitive level
- `FrameStrategy` engineers variance deliberately, through identity frames, discipline frames, temperature ladders, and constraint inversions. It does not rely on N samples of the same prompt
- Every `Variant` carries a rationale, so the human can see *why* this variant differs from that one
