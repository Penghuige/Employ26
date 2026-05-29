# Skill Dictionary Workflow SOP

## Goal

This SOP standardizes iteration for `src/skill_extraction` so each round follows
the same control flow:

1. Run baseline dictionary matching.
2. Run regression evaluation.
3. Classify the current error mode.
4. Apply only conservative changes.
5. Re-run matching and evaluation.
6. Stop when metrics meet targets or when the workflow is blocked.

The fixed framework does not change. Only the repair action inside a round can
change, based on the observed error pattern.

## Entry Points

### 1. Baseline only

```bash
python -m src.skill_extraction.skill_dictionary_workflow baseline
```

Use this to refresh the current baseline and write workflow state without
changing the dictionary.

### 2. Full normalized workflow

```bash
python -m src.skill_extraction.skill_dictionary_workflow run
```

This executes the configured iterative workflow and writes a persistent state
file.

### 3. Workflow status

```bash
python -m src.skill_extraction.skill_dictionary_workflow status
```

This reads the latest workflow state and shows the current stage, metrics, and
next action.

## State And Artifacts

- Workflow state:
  `output/skill_extraction/reports/dictionary_iteration/workflow_state.json`
- Per-round iteration reports:
  `output/skill_extraction/reports/dictionary_iteration/`
- Regression summaries:
  `output/skill_extraction/reports/regression_eval/`
- Dictionary rules:
  `config/skill_dictionary_iteration.json`

## Fixed Stages

### Stage A: Baseline Evaluation

Always run:

- `match_flat_skills_to_duckdb.py`
- `regression_eval.py`

Outputs:

- baseline precision / recall / F1
- top false positives
- top false negatives
- error mode classification

### Stage B: Error Mode Classification

The workflow classifies the round into one of these modes:

- `precision_first`
  False positives dominate. Prefer filters, alias cleanup, and contextual
  constraints.
- `recall_first`
  False negatives dominate. Prefer conservative additions, canonical merges,
  and controlled aliases.
- `balanced`
  Both sides are similar. Prefer minimal mixed fixes.

### Stage C: Conservative Repair

Allowed actions:

- add high-confidence candidate skills
- merge canonical synonyms
- add short-term allowlist entries
- add contextual matching rules
- block generic skill names
- block unsafe aliases

Disallowed actions:

- bulk rewriting the whole dictionary
- sending the full dictionary to API models
- adding low-confidence candidates directly to the main dictionary
- repeating a previously rejected error pattern without a new rule

### Stage D: Re-evaluation

After every repair round, always re-run:

- dictionary match
- regression eval

Then compare:

- precision delta
- recall delta
- F1 delta

## Model Responsibilities

### Local LLM

Use for:

- candidate discovery
- low-cost extraction probes

Do not use local LLM as the final arbiter when structured output quality is
unstable.

### `gpt-5.4-mini`

Use only for:

- high-uncertainty candidate adjudication
- boundary-case review
- small-sample quality audit

Constraints:

- never review the full dictionary
- default cap is `max_api_reviews`
- prompts must contain only the current sample, candidate list, and minimal
  evidence

## Stop Conditions

The workflow stops when one of these is true:

1. `precision >= precision_target`
2. `recall >= recall_target`
3. `f1 >= f1_target`
4. no API-reviewed candidate is kept and workflow policy says stop
5. metric gain is below workflow thresholds and workflow policy says stop
6. max rounds reached

Targets and stopping thresholds live in:

- `config/skill_dictionary_iteration.json`

## Why Repair Actions Can Change

The framework is fixed, but the repair action cannot be a single repeated
template because late-round errors are different from early-round errors.

Typical sequence:

- early rounds: generic false positives, synonym normalization, obvious missing
  tools
- middle rounds: long-tail missing technical terms
- late rounds: short acronyms, context-dependent aliases, 2-character Chinese
  terms

These require different repair methods. Repeating the same “extract candidates
+ review + merge” action every round would overfit recall and reintroduce
precision failures.

## Recommended Operating Rule

Use this decision policy:

- If the issue is a generic false positive, add a filter or contextual rule.
- If the issue is a canonical synonym miss, add a merge or alias.
- If the issue is a clear long-tail hard skill, add conservatively.
- If the issue is an ambiguous short acronym, keep it in review unless context
  can constrain it.
- If the issue remains ambiguous after one review round, stop and escalate to
  manual confirmation.

## Current Controller

The normalized controller is implemented in:

- [skill_dictionary_workflow.py](/D:/PythonProjects/Employ26/src/skill_extraction/skill_dictionary_workflow.py)

It provides:

- baseline execution
- persistent workflow state
- round history
- target-based stop conditions
- explicit next-action decisions
