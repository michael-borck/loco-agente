# LocoAgente Conversational Harness — Design Spec

**Status:** Draft, awaiting user review
**Date:** 2026-05-07
**Owner:** Michael Borck
**Repo:** loco-agente
**Supersedes:** the existing `track A/B/C/D` framing in `loco-agente/README.md` (which becomes a reorganisation under this design)

---

## 1. Mission and philosophy

### Mission

Build a **conversational harness** — a four-subsystem architecture for small local models — that amplifies human thinking through verified offloading rather than substituting for it. The harness is a research substrate and reference implementation, not a production framework.

This reframes loco-agente's prior mission ("Can small models think in loops?") into a broader question: *Can a well-designed harness make small local models genuinely useful as thinking partners — not by making them act like frontier models, but by channelling their distinct strengths?*

### The three load-bearing principles

These principles drive every design decision in this spec. They are testable claims, not slogans.

#### 1. Creativity does not need precision; it needs ideas

Frontier models win on **precision tasks**: factual recall, math, code correctness, well-defined classification. Small local models lose those head-to-head — confabulation, mode collapse, narrower training distributions.

Frontier models lose on **creative tasks**: when 50 frontier models write a story about a boy and a dragon, there is a striking sameness across them. They converge to a safe centre. Variance — the soul of creativity — is what they erode.

Small local models, properly harnessed, can compete on creative tasks for the inverted reason: their imprecision *becomes variance*, and variance is what creativity needs. The design imperative is to **stop trying to make small models precise** and instead channel their variance through the harness.

This is the project's positioning claim. It justifies why a conversational harness on small local models is worth building rather than a frontier-API wrapper.

#### 2. Conversation, not delegation — but verified offloading is fine

Delegation is not binary. Cognitive offloading is acceptable when there is a verification loop. The cash-register example: when you buy items at a shop, you delegate the addition to the till; when the total seems wrong, you ask to check the prices. That is not surrender; it is amplification with a smell-test boundary.

The harness's job is to **make the verification loop cheap**:

- Surface uncertainty so the human's smell test has something to fire on
- Expose tool outputs so verification doesn't require re-running the work
- Maintain conversation state so the human can push back without re-explaining

The model handles the mechanical (compute, draft, enumerate, summarise). The human keeps the judgment (taste, smell tests, strategic direction, "this feels off"). The harness is the connective tissue that lets the human work at a higher level without losing oversight at any level they choose to inspect.

#### 3. Confidence is not competence

Humans treat confidence as a synonym for competence. A confident-sounding model sounds correct. This is dangerous, especially with small local models whose confidence is famously miscalibrated.

The harness must **surface uncertainty as a first-class signal**. Every variant carries a rationale and an uncertainty marker, so the human's smell test has something to fire on. Refusing to surface uncertainty is a design defect, not a UX simplification. A confident-sounding output with no uncertainty markers is a primitive-contract violation.

### What this changes versus prior loco-agente

- The mission expands from "Can small models think in loops?" to "Can a conversational harness make small local models genuinely useful as thinking partners?"
- The prior `Track A/B/C/D` structure becomes layer studies and application demos under this architecture (see §5)
- The conversational/creativity-augmentation thesis becomes the project's voice; the agentic-loop research stays but in a humbler frame ("delegation with verification at the right boundary") and lives in `applications/delegation/`
- All five existing design principles persist (Local first, Constrained over capable, Measurably better, Comparable experiments, Accumulative)
- LocoLLM connection persists (same base model, adapter pipeline)
- Phased roadmap structure persists, just reorganised under the new architecture (see §6)

---

## 2. The E primitive — the harness's foundational contract

Every layer above this contract must use it. Every demo configuration must respect it. The primitive is what makes the harness conversational-not-delegational at every layer.

### Contract

```python
@dataclass
class Variant:
    """One generated alternative. Always plural at the primitive level."""
    text: str                            # the actual generated content
    rationale: str                       # why this variant — framing, perspective, reasoning
    uncertainty: Uncertainty             # surfaced uncertainty (see below)
    metadata: dict[str, Any]             # frame name, sampling params, model, etc.


@dataclass
class Uncertainty:
    """Surfaced uncertainty signals. Load-bearing fields are flags and verification_hooks."""
    confidence: float                    # 0.0–1.0 self-reported, treated as research artifact
    relative_rank: int | None = None     # rank within batch (1 = highest), if requested
    flags: list[str]                     # known failure modes ("citation may be hallucinated")
    verification_hooks: list[str]        # what the human should check ("verify the price")


def generate_variants(
    *,
    prompt: str,
    n: int,                              # always ≥ 2 — singular outputs are forbidden at this layer
    frame_strategy: FrameStrategy,       # how to deliberately differentiate the variants
    model: ModelHandle,
    context: ContextBundle,              # Context layer's hierarchical instruction set
) -> list[Variant]: ...
```

### What `frame_strategy` does

This is the variance-generation knob. It answers: *how do we keep the variants from collapsing to the average?*

Strategies the harness ships with:

- `IdentityFrames(frames=["CEO", "legal counsel", "marketing"])` — for the Perspective Debate demo
- `DisciplineFrames(disciplines=["systems biology", "operations research", ...])` — for cross-disciplinary synthesis
- `TemperatureLadder(temps=[0.3, 0.7, 1.0, 1.3])` — vanilla diversity sampling
- `ConstraintInversion(base_brief, constraints_to_flip)` — deliberately violate each assumption to surface its load-bearing weight

The `FrameStrategy` interface is extensible. Students and contributors add new strategies as the project matures. **Variance is engineered, not hoped for** — sampling N times and praying does not satisfy the contract; deliberately differentiating the variants does.

### What `verification_hooks` enables

These are the cheap-verification handoffs. The harness produces them; the UI surfaces them; the human checks the ones that fire. Examples generated automatically:

- If the model cited a paper → `"verify citation: <paper title>"` hook
- If the model claimed a number → `"verify quantity"` hook
- If the model proposed a code change → `"smell-test architecture"` hook

Hooks default to **conservative** — better to flag too much than too little, because the cost of an unflagged hallucination is much higher than the cost of a verified-correct claim.

### What the contract rules out

- **Singular outputs are forbidden at this layer.** `n ≥ 2` is enforced. If a use case really wants one answer, it picks-from-N at the orchestration layer, not at the primitive layer.
- **Hidden uncertainty is forbidden.** If the model doesn't know, the variant must say so. A confident-sounding variant with no uncertainty markers is a primitive-contract violation.
- **No variant is final without a rationale.** The rationale is what the human's smell test reads. Bare outputs are not allowed.

### Calibration capture (a Context-layer requirement)

Self-reported `confidence` on small local models is famously miscalibrated. Treat it as a **research artifact**, not a load-bearing signal:

- Load-bearing uncertainty signals: `flags` and `verification_hooks`. UI surfaces these prominently.
- Auxiliary signal: `confidence` (logged for every variant, surfaced subtly in UI).
- More useful comparative signal: `relative_rank` ("of these N variants you produced, rank by your own confidence"). Empirically more meaningful than absolute confidence even when absolute is miscalibrated.

The harness must capture every user pick / reject / edit on every variant in machine-readable form. Over time this produces a labeled dataset (model self-confidence claim → human-validated outcome) that supports a planned **calibration study** in `studies/context/` (see §5). The data capture starts on day one; the analysis is downstream research.

---

## 3. Layer architecture

The harness has four named subsystems. They map to the four-layer enterprise harness model from contemporary literature[^1] but the project uses names rather than numbers because the diagram does not flow numerically and the project's research surfaces are not the enterprise framing's research surfaces.

[^1]: The four-layer harness pattern (orchestration / tools / context / inference) is articulated in industry literature (see notes circulated 2026 on "Harness Engineering"). The numbering varies by source. The project adopts the architecture but uses subsystem names for clarity.

```
┌──────────────────────────────────────────────────────────┐
│  ORCHESTRATION  — patterns that compose the E primitive  │
│                   (Debate, Synthesis, IterativeRefine…)  │
└──────────────────────────────────────────────────────────┘
                          ↓ uses
┌──────────────────────────────────────────────────────────┐
│  CONTEXT        — hierarchical instruction file system   │
│                   (personality, rules, memory, tools)    │
└──────────────────────────────────────────────────────────┘
                          ↓ uses
┌──────────────────────────────────────────────────────────┐
│  TOOLS          — MCP-based external capabilities        │
│                   (deferred to Phase 2; stub in Phase 1) │
└──────────────────────────────────────────────────────────┘
                          ↓ uses
┌──────────────────────────────────────────────────────────┐
│  INFERENCE      — OpenAI-compatible backend (config)     │
│                   (covered by loco-bench / loco-puente)  │
└──────────────────────────────────────────────────────────┘
```

### Asymmetry of layers

The four layers are not equally novel. **Orchestration and Context are where the project's research contribution lives.** Tools is "adopt MCP, do not reinvent." Inference is already covered by loco-bench (which engine to pick) and loco-puente (deployment). The spec is intentionally heaviest on Orchestration and Context.

### Orchestration — the primary research surface

Each pattern is a class implementing a common protocol; new patterns are first-class extension points.

```python
class OrchestrationPattern(Protocol):
    def run(
        self,
        brief: str,
        primitive: GenerateVariantsFn,
        context: ContextBundle,
    ) -> Conversation: ...
```

A `Conversation` is a structured trace: every variant produced, every frame used, every uncertainty surfaced, every user decision (selected / rejected / edited / asked-for-more). This trace is the artifact that downstream studies consume.

**Patterns shipped in Phase 1:**

| Pattern | What it does | First demo using it |
|---|---|---|
| `SinglePass` | One call → N variants → return all | smallest building block |
| `DebatePattern` | N framed agents → R rounds → optional moderator merge | **Perspective Debate demo** |
| `SynthesisPattern` | N source-grounded variants → dedup → attribute → return ranked | **Research Synthesis demo (Phase 2)** |
| `IterativeRefinement` | Variants → human/auto critique → re-variant with feedback | research-track use |

The pattern interface is small enough that students can implement a new one in a few hundred lines. **Orchestration is where contributions accumulate**.

### Context — the secondary research surface

Treats prompts like a file system, not a monolithic string. Modular instruction files loaded by session hooks based on the current task:

```
profiles/<profile_name>/
├── personality/         # one file per frame: ceo.md, legal.md, marketing.md
├── rules/               # hard constraints: never_fabricate.md, cite_or_flag.md
├── memory/              # persistent: user_preferences.md, calibration_history.jsonl
└── tools/               # available MCP tool specs as markdown
```

Session hooks load only what is relevant for the current task. The harness assembles the final prompt by composing files according to declarative rules in the orchestration pattern. **No mega-system-prompt strings are permitted in code.**

The prior Track C ("Scaffolding strategies") is now an explicit Context-layer study: *which file structures, frame counts, rule formulations produce best-calibrated variants?* Concrete experiments, comparable across runs.

### Tools — adopt-not-invent (MCP-based)

The Tools layer provides external capabilities (retrieval, search, file access, code execution) through Model Context Protocol (MCP) servers.

- **Phase 1:** ship with a stub MCP client; tools list is empty for the Perspective Debate demo
- **Phase 2:** wire real MCP servers (Semantic Scholar, arXiv, file system, Zotero) for the Research Synthesis demo
- **Adopt-not-invent rule:** if an MCP server already exists for a need, adopt it. Original MCP servers are written only when none exists for the use case (e.g., a "list papers in my Zotero by discipline tag" server may not exist — that is a writeable contribution).

#### On retrieval and RAG

Retrieval is delegated to the Tools layer. **RAG is a valid implementation strategy *inside* an MCP tool but is not a harness-level concept.** The harness never knows whether a `search_corpus` tool is backed by RAG, BM25, full-text scan, semantic search, or a human typing answers — it just sees the tool's input/output contract.

- We neither build nor forbid RAG infrastructure — we adopt MCP servers that do retrieval well, regardless of internal mechanism.
- If an MCP server does not exist for a needed retrieval type, building a RAG-backed MCP server is a **contribution opportunity**. The harness does not care about its internals — only its tool contract.
- A planned Context-layer study compares RAG-backed vs tool-search retrieval for small local models on the Research Synthesis demo (see §5).

#### On MCP volatility

MCP is the current standard. The industry has shifted positions on MCP architecture more than once and may again. The architectural insulation is intentional: **the Tools layer is the only place the harness needs to change if MCP is replaced.** Orchestration, Context, the E primitive, and Inference are unaffected. The harness is bigger than any one tool standard.

### Inference — covered elsewhere

The harness is **inference-engine-agnostic**. It speaks OpenAI-compatible HTTP. Backends:

- **Ollama** — default for development ergonomics
- **llama.cpp server** — for the squeezed-most-out-of-budget hardware case (the moe-budget work in loco-bench feeds in here)
- **vLLM / SGLang** — for performance on bigger rigs (loco-puente, Hidra)

Inference engine choice is a configuration knob, not architecture. The Inference layer in this spec is one paragraph that says: *use whatever loco-bench says is best for your tier.*

---

## 4. Demo configurations

Two demos, both using the same harness library, different orchestration patterns and context bundles. Both are committed in Phase 1; the Perspective Debate demo runs end-to-end, the Research Synthesis demo is a skeleton with stubbed retrieval against a curated corpus.

### Demo 1: Perspective Debate (`applications/conversation/perspective_debate/`) — Phase 1, end-to-end

**Headline brief:** *"I'm thinking about [X]. Have a CEO, a legal counsel, and a marketing director debate it for 3 rounds. Surface where they disagree. Don't pick a winner — give me a synthesis of where the disagreements actually are."*

**Composition:**
- **Orchestration:** `DebatePattern(frames=["ceo", "legal", "marketing"], rounds=3, moderator="surface_disagreements")`
- **Context:** loads `profiles/business/personality/{ceo,legal,marketing}.md`, `profiles/business/rules/{never_fabricate,disagree_explicitly}.md`
- **Tools:** none (closed-world)
- **Inference:** Ollama default (Qwen3-4B Q4_K_M)
- **Output:** structured `Conversation` trace + a markdown summary showing each round's variants, the disagreements, and the verification hooks ("verify the regulatory claim about X")

**Why this demo first:**
- Closed-world (no retrieval = no Tools layer work)
- Multi-frame variance is exactly what small models are good at when channelled
- Direct test of the "creativity does not need precision" thesis — readers judge by whether the perspectives feel distinct and useful, not by factual precision
- Quick to build (~1–2 weeks once harness library exists)
- Generic pattern — works for product strategy, curriculum design, writing, research planning

**Configurable for live demos.** The `DebatePattern(frames=[...])` resolves frame names against profile bundles. Phase 1 ships at least three profile bundles:

- `profiles/business/` — ceo, legal, marketing (the headline)
- `profiles/academic/` — methodologist, theorist, practitioner
- `profiles/writing/` — protagonist, antagonist, narrator

Audience-appropriate live demos via configuration flag, not code change.

**Phase 1 success criteria for this demo:**
- Three personalities are distinguishable to a reader (blind test: can a human guess which frame produced which variant?)
- Disagreements are surfaced explicitly, not glossed
- Uncertainty hooks fire when the model claims something verifiable
- Runs at reading speed on a consumer GPU

### Demo 2: Cross-Disciplinary Synthesis (`applications/conversation/research_synthesis/`) — Phase 1 skeleton, Phase 2 functional

**Headline brief:** *"I'm working on [problem in field X]. Scan papers from [list of adjacent fields]. For each field, return the 3 most relevant connections, with rationale. Flag where you might be confabulating citations."*

**Composition:**
- **Orchestration:** `SynthesisPattern(disciplines=[...], variants_per_discipline=3, dedupe="semantic", attribute="strict")`
- **Context:** loads `profiles/academic/personality/cross_disciplinary_scout.md`, `profiles/academic/rules/{cite_or_flag,explicit_uncertainty_on_citations}.md`
- **Tools (Phase 1):** **stubbed** — a folder of pre-loaded papers grouped by discipline (`applications/conversation/research_synthesis/sample_papers/{biology,operations,history,…}/*.md`). The synthesis runs against this curated corpus, not live retrieval.
- **Tools (Phase 2):** real MCP servers for paper retrieval (Semantic Scholar / arXiv / Zotero / full-text fetch)
- **Inference:** same Ollama default; bigger context window matters here (32K+)

**Why ship a skeleton in Phase 1:**
- Proves the orchestration pattern works for synthesis even without real retrieval
- Makes Phase 2 work concrete: "wire MCP retrieval into this exact place"
- The folder-of-papers demo is **itself useful** — a researcher with their own corpus performs cross-discipline scans
- Forces designing the Tools layer interface before it's needed
- **Reproducibility:** stubbed corpus = same input → same output → reproducible across runs and across users. Live retrieval makes demos non-deterministic. The stub is a methodological commitment, not a Phase-1 shortcut.

**Phase 1 success criteria for this demo:**
- Given a folder of papers across ≥4 disciplines, produces 3 connections per discipline with cited passages
- Each connection has a rationale + uncertainty markers
- Runs without confabulating citations to papers not in the folder (a real test of the `rules/cite_or_flag.md` constraint)

**Phase 2 success criteria:** real retrieval, but the orchestration code is unchanged from Phase 1.

### What both demos share

- The same `harness.core.generate_variants` primitive
- The same `Conversation` output shape
- The same `Variant` and `Uncertainty` dataclasses
- The same calibration-capture mechanism (every user pick / reject / edit logged)
- Identical inference backend (so comparisons are clean)

That last property is load-bearing: **the demos differ only in orchestration pattern + context bundle.** That is how we know the harness abstraction is real — changing the pattern or the bundle changes the application; changing nothing else does.

### What both demos explicitly skip

- No fancy UI. CLI-only in Phase 1.
- No multi-user / no cloud / no auth. Single-user research tool.
- No fine-tuning during the demo. Adapter training is LocoLLM's job; the harness consumes whatever model LocoLLM produces.

---

## 5. Track reframing and repository layout

### Track reframing

The existing four tracks become studies and applications under the layer architecture. Same research questions; sharper home.

| Old track | New home | What changes |
|---|---|---|
| **A: Karpathy autoresearch** | `applications/delegation/autoresearch/` | Reframed as "delegation-mode harness configuration with verification loop." The action is autonomous, but every step produces a `Variant` with `verification_hooks`, and the human can intervene at any iteration. The research question shifts from "can it loop autonomously" to "what is the right verification cadence?" |
| **B: Task-specific agents** | `applications/delegation/{data_analysis,code_review,documentation}/` | Each becomes a different orchestration-pattern + context-bundle composition. Same harness, different configurations. |
| **C: Scaffolding strategies** | `studies/context/` (the Context layer's research surface) | Now an explicit Context-layer study. Hierarchical-file experiments, prompt-strategy comparisons, calibration-capture analysis all live here. **Calibration-via-user-feedback** sits here. |
| **D: Framework evaluation** | `studies/orchestration/` (the Orchestration layer's research surface) | Now an explicit Orchestration-layer study. Compare hand-rolled `DebatePattern` vs LangGraph-implemented vs CrewAI-implemented for the SAME demo brief. Concrete, apples-to-apples. |

Two new study tracks emerge naturally:

- `studies/orchestration/` — patterns and frameworks (was Track D)
- `studies/context/` — file structures, calibration, frame design (was Track C, plus calibration capture)

Two new application tracks emerge:

- `applications/conversation/` — the conversational/creative demos (Perspective Debate, Research Synthesis)
- `applications/delegation/` — the verified-offloading demos (autoresearch, task agents)

The split between conversation and delegation is subtle but worth maintaining. Some readers will see autoresearch as pure delegation; closer inspection reveals the human agency embedded in the verification loop. The split is honest about that nuance.

### Repository layout

```
loco-agente/
├── README.md                            # the new conversational-harness mission
├── pyproject.toml                       # adds `harness` package
├── docs/
│   ├── superpowers/specs/               # design specs (this file)
│   ├── philosophy/
│   │   ├── conversation-not-delegation.md
│   │   ├── creativity-needs-variance-not-precision.md
│   │   └── confidence-is-not-competence.md
│   ├── architecture/
│   │   └── four-subsystems.md           # the named-not-numbered architecture overview
│   └── tutorials/                       # demo-as-tutorials
├── harness/                             # the library
│   ├── __init__.py
│   ├── core.py                          # the E primitive: generate_variants, Variant, Uncertainty
│   ├── orchestration/                   # Orchestration layer
│   │   ├── pattern.py                   # OrchestrationPattern protocol
│   │   ├── single_pass.py
│   │   ├── debate.py                    # DebatePattern
│   │   ├── synthesis.py                 # SynthesisPattern
│   │   └── iterative.py                 # IterativeRefinement
│   ├── context/                         # Context layer
│   │   ├── bundle.py                    # ContextBundle (loads files, applies session hooks)
│   │   ├── frames.py                    # frame resolution against personality bundles
│   │   └── calibration.py               # captures user picks/rejects/edits
│   ├── tools/                           # Tools layer
│   │   ├── stub.py                      # Phase 1 stub MCP client
│   │   └── mcp.py                       # Phase 2 real MCP client
│   ├── inference/                       # Inference layer (thin wrapper)
│   │   └── openai_compat.py             # talks to Ollama / llama.cpp / vLLM / SGLang
│   └── conversation.py                  # Conversation trace dataclass + serialisation
├── profiles/                            # shipped Context bundles (data, not code)
│   ├── business/
│   │   ├── personality/{ceo,legal,marketing}.md
│   │   └── rules/{never_fabricate,disagree_explicitly}.md
│   ├── academic/
│   │   ├── personality/{methodologist,theorist,practitioner,cross_disciplinary_scout}.md
│   │   └── rules/{cite_or_flag,explicit_uncertainty_on_citations}.md
│   └── writing/
│       ├── personality/{protagonist,antagonist,narrator}.md
│       └── rules/{stay_in_voice,no_self_referential}.md
├── applications/
│   ├── conversation/
│   │   ├── perspective_debate/          # Demo 1 (Phase 1)
│   │   └── research_synthesis/          # Demo 2 skeleton (Phase 1) → real (Phase 2)
│   └── delegation/
│       ├── autoresearch/                # was Track A
│       └── task_agents/                 # was Track B
├── studies/
│   ├── orchestration/                   # was Track D — pattern/framework comparisons
│   ├── context/                         # was Track C — context-layer research, calibration
│   └── inference/                       # cross-link to loco-bench
├── tests/
│   ├── harness/
│   ├── orchestration/
│   ├── context/
│   └── applications/
└── cli/
    ├── locoagente.py                    # main CLI entry point
    └── commands/
        ├── debate.py                    # `locoagente debate --profile business …`
        ├── synthesise.py                # `locoagente synthesise --corpus ./papers …`
        └── log.py                       # inspect/export Conversation traces
```

### What this layout enforces

- **Harness library is small and focused.** Each subdirectory has one responsibility.
- **Profiles are data, not code.** Audience-appropriate demos = swap profile flag, no code changes.
- **Studies and applications are clearly separated.** Studies are research artifacts (papers come from these); applications are demos and tools.
- **No directory called "track."** The old track structure is dead; layer + application/study is the new organising principle.

---

## 6. Success criteria, scope, roadmap

### Phase 1 success — "harness exists and one demo works end-to-end"

- `harness/` library: core primitive + ≥3 orchestration patterns (`SinglePass`, `DebatePattern`, `SynthesisPattern`) + Context loader + stub Tools client + Inference wrapper
- ≥80% unit-test coverage on `harness.core` and `harness.orchestration`
- Three Context profile bundles shipped: `business/`, `academic/`, `writing/`
- **Perspective Debate demo runs end-to-end** on consumer GPU at reading speed, with structured `Conversation` trace and verification hooks
- **Research Synthesis skeleton** runs against a folder-of-papers corpus, produces 3 connections per discipline with rationale + uncertainty
- Calibration capture works: every user pick / reject / edit logged in machine-readable form
- Documentation: 3 philosophy docs (one per principle in §1), 1 architecture overview (§3), 2 tutorials (one per demo), CLI reference

### Phase 2 success — "harness has external evidence and a research outcome"

- Real MCP retrieval wired into Research Synthesis (Semantic Scholar / arXiv / Zotero)
- First Context-layer calibration study published as a technical report — does the calibration capture data show systematic miscalibration patterns in small models? Can a calibration adapter improve confidence reliability?
- ≥1 orchestration pattern contributed by a student or external researcher (proves the extension surface is real)
- Orchestration-layer study: same demo brief run through `DebatePattern` (hand-rolled) vs LangGraph-implemented vs CrewAI-implemented; report on context bloat and quality differences
- Context-layer study: RAG-backed vs tool-search retrieval for small local models on the Research Synthesis demo — when does each fail?

### Phase 3 success — "the delegation applications come back, harness-native"

- Karpathy autoresearch port runs as a `harness/` application with verification hooks at each iteration
- ≥1 `applications/delegation/` task agent (data analysis or code review) running end-to-end with verified-offloading framing

### Project-level meta-success (the thesis test)

- A reader can blind-test the Perspective Debate output and reliably distinguish frames (proving variance is real, not nominal)
- The "creativity does not need precision" thesis has at least one empirical artifact backing it
- Harness runs on consumer hardware (~$500 GPU + 32GB RAM) — anyone can replicate
- Contributions accumulate semester-by-semester (≥1 student-authored pattern, profile, or study per semester)

### Explicitly out of scope

These are not in scope for the loco-agente repo. They keep the project focused.

- **Not a frontier-model wrapper.** Designed for small local models. Frontier APIs are baseline comparisons only.
- **Not a production agent framework.** Research substrate, not enterprise SaaS. No SLAs, no monitoring, no scale guarantees.
- **Not training models.** That's LocoLLM. The harness consumes whatever LocoLLM produces.
- **Not benchmarking inference engines.** That's loco-bench. The harness picks one and works with it.
- **Retrieval is delegated to the Tools layer (MCP).** RAG is a valid implementation strategy inside an MCP tool but is not a harness-level concept. We neither build nor forbid it — we adopt MCP servers that do retrieval well, regardless of internal mechanism.
- **Not multi-user / no auth.** Single-user research tool. Production deployment is loco-puente's problem if it ever needs it.
- **Chatbot UIs (and other rich interfaces) are out of scope as harness components.** The harness produces structured `Conversation` traces; what consumes and renders them is a separate concern. Phase 1 ships a CLI as the reference interface — it is the minimum viable consumer that proves the harness works. Building a chatbot UI, voice interface, IDE extension, or web dashboard is welcomed as a *downstream project* that imports the harness as a library — but it lives outside this repo.

### Connection to the loco fleet

| Project | Relationship |
|---|---|
| **loco-llm** | Provides base model + task-specific adapters. Harness consumes models; does not train them. The adapter pipeline could later produce a "perspective-debate adapter" — Phase 4 work. |
| **loco-bench** | Answers "which inference engine + model fits my VRAM tier" — the harness's Inference layer pulls that answer. Cross-link from `studies/inference/`. The MoE-on-a-budget work feeds in here. |
| **loco-puente** | Deployment substrate. If/when the harness gets a web UI, Puente hosts it. Out of scope for Phase 1–2. |
| **loco-ensayo (WorkReady)** | Potential downstream consumer — character agents in WorkReady could be configured Perspective Debate instances. Worth noting; not a deliverable. |

### Roadmap

Aligned with existing loco-agente phasing (Foundation / Scaffolding / Task Agents / Integration), reorganised under the new architecture.

| Phase | Window | Deliverable |
|---|---|---|
| **Phase 1** | now → end of 2026 (Semester 2 2026) | Harness library + Perspective Debate demo + Research Synthesis skeleton + 3 profiles + calibration capture + tutorials + tests |
| **Phase 2** | Semester 1 2027 | Research Synthesis with real MCP retrieval + first calibration study + ≥1 contributed orchestration pattern + Orchestration framework comparison |
| **Phase 3** | Semester 2 2027 | Delegation applications (autoresearch port, task agents) re-implemented as harness applications |
| **Phase 4** | 2028+ | Adapter training (LocoLLM integration), multi-modal frames, loco-convoy multi-GPU experiments |

### Out-of-band parking lot

Captured for visibility, **not in scope for this spec:**

- **Adapter trained specifically for "produce well-differentiated variants"** — a debate adapter (Phase 4 LocoLLM work)
- **Web UI for conversation traces** (Phase 3+, possibly out-of-repo)
- **PoC chatbot or rich UI as a downstream project.** Once the harness library is stable and the CLI proves the contract, a separate repo (e.g., `loco-conversa-chat`) could import the harness and ship a polished chat experience. That is an ecosystem move, not a loco-agente deliverable. Suggested for someone post-Phase-1.
- **Cross-language profiles** (currently English-only; Spanish for LocoLab voice could come later)
- **Voice interface** (record audio → transcribe → harness → speak response) — downstream consumer
- **Multi-modal frames** (image, audio, video) — needs different inference backend
- **Calibration adapter trained on captured user feedback** — Phase 2–3 research outcome
- **Integration with loco-puente browser portal** — once Phase 3 stabilises the CLI

---

## 7. Open questions and risks

### Open questions

1. **Do small local models actually produce distinguishable frames?** The Perspective Debate demo's success criterion is that humans can blind-test which frame produced which variant. If models cannot maintain frame coherence over 3 rounds at the 4B scale, the demo fails its own test. Mitigation: the calibration-capture mechanism logs which variants users select; we can detect frame collapse empirically.
2. **Is the `FrameStrategy` interface flexible enough for unanticipated study types?** Adding new strategies should be cheap. If the first contributed strategy from a student requires harness changes, the interface needs revision.
3. **Will reproducibility-via-stubbed-corpus translate to live-retrieval credibility?** Reviewers may discount Phase 1 results because they're "just running over a curated corpus." Phase 2's real-retrieval results need to corroborate Phase 1 findings, or we have a credibility gap.
4. **Calibration adapter training data sufficiency.** The calibration-via-user-feedback study requires enough user interactions to be statistically meaningful. If the harness gets few users, the dataset stays small. Mitigation: synthetic interactions for early development; real users come later.

### Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| Frame collapse on small models (variants converge despite different framings) | Medium | Calibration capture detects this; can trial larger models or different frame strategies |
| MCP standard shifts mid-project | Low–medium | Tools-layer insulation; we adopt the successor, harness unchanged |
| Reviewers reject "stubbed corpus" Phase 1 demo as inadequate | Medium | Frame Phase 1 as a methodology + reproducibility commitment, not a shortcut. Phase 2 corroborates with real retrieval. |
| Over-engineering the harness library before the demos validate it | Medium | YAGNI ruthlessly. Implement only what the two demos and the calibration study require. New patterns and bundles come later, contributed by users. |
| Calibration confidence signal too noisy to be useful | High | Treat `confidence` as research artifact, not load-bearing. Lean on `flags` and `verification_hooks`. The research question is whether calibration becomes useful given enough data. |

---

## 8. Reproducibility and integrity

- All `Conversation` traces serialise to JSON with full provenance: model, frame strategy, profile bundle, every variant, every user decision, every uncertainty signal.
- The Phase 1 stubbed corpora for Research Synthesis are checked into the repo with hashed file integrity — same input bytes, same output run by anyone.
- Profile bundles are version-controlled. Any change to a bundle is a commit; runs can be replayed against historical bundle versions.
- Every CLI invocation logs the harness version, profile version, model name, and inference backend in the resulting `Conversation` trace.
- The first contribution from each new participant (student, researcher) is to replicate the Perspective Debate demo end-to-end on their own hardware. That is the integration test for "harness runs on consumer hardware."

---

## 9. References

- LocoAgente prior README — `loco-agente/README.md` (the four-track structure that this spec reorganises)
- LocoLLM README — `loco-llm/README.md` — adapter training thesis
- LocoBench MoE-on-a-Budget design spec — `loco-bench/docs/superpowers/specs/2026-05-06-locobench-moe-budget-design.md` — Inference layer choices
- Andrej Karpathy — *Context Engineering* (mid-2025) — the precursor frame to harness engineering
- Industry literature on the four-layer harness architecture, "Harness Engineering" notes circulated 2026 (LLM as CPU, harness as OS)
- Anthropic — Model Context Protocol (MCP) standard
- Stanford — DSPy programmatic prompt optimisation
- UC Berkeley — Gorilla LLM (function-calling specialist)

---

## Appendix A: Glossary

- **The E primitive** — `generate_variants(prompt, n, frame_strategy, model, context) -> list[Variant]`. The single contract every layer of the harness builds on. Always plural; always carries rationale; always surfaces uncertainty.
- **Frame strategy** — the deliberate variance-generation mechanism. Identity frames, discipline frames, temperature ladders, constraint inversions. Variance is engineered, not hoped for.
- **Verification hook** — a structured signal in a `Variant` indicating something the human should check. Cheap-verification handoff; the foundation of "verified offloading."
- **Profile bundle** — a directory of Context-layer files (personality, rules, memory, tools) that configures the harness for a particular audience or use case. Profiles are data, not code.
- **Conversation trace** — the structured record of every interaction with the harness. Variants, frames, uncertainty, user decisions. The artifact that downstream studies consume.
- **Conversational harness** — the four-subsystem architecture that supports verified offloading; the project's primary deliverable.
- **Verified offloading** — cognitive offloading with a verification loop. The harness's design principle: do the mechanical work, surface uncertainty, expose tool outputs, maintain conversation state for cheap human pushback.
