# AGENTS.md

## Project Purpose

This directory is not a finished code repository yet. It is a documentation-driven workspace for building a research codebase with AI coding agents.

The target project is:

**Non-IID-aware, low-ambiguity, adaptive black-box copyright verification for federated learning**

In practical terms, the future repository should support:

1. Federated learning training.
2. Watermark or fingerprint generation.
3. Adaptive watermark allocation under non-IID client data.
4. Black-box ownership verification with score, threshold, and margin logic.
5. Attack evaluation such as fine-tuning, pruning, quantization, distillation, or extraction.
6. Evaluation, reporting, and paper-ready figure or table export.

This workspace currently contains the planning and execution documents that tell an AI agent how to build that repository correctly.

## Your Role As An Agent

If you are an AI coding agent entering this directory, your job is **not** to improvise a brand new architecture.

Your job is to:

1. Read the existing guidance files first.
2. Follow the documented architecture and workflow.
3. Make minimal, staged, verifiable progress.
4. Keep the repository structure, experiment protocol, and implementation boundaries clean.

Do not skip the reading order below.

## Required Reading Order

Before making any change, read files in this order.

### 1. Research Plan

Read first:

`deep-research-report-2.md`

Why:

1. It defines the research problem.
2. It explains the intended method direction.
3. It gives the module decomposition rationale.
4. It explains the expected datasets, baselines, attacks, metrics, and experimental phases.

### 2. Main Build Guide

Read second:

`ai-code-generation-guide.md`

Why:

1. This is the main operating manual for AI code generation.
2. It defines the architecture, module boundaries, implementation phases, quality rules, Git rules, and prompt templates.
3. If there is any uncertainty about what to build or how to build it, this file is the primary source of truth.

### 3. One-Page Summary

Read third:

`ai-code-generation-guide-one-page-checklist.md`

Why:

1. It is the compressed operational checklist.
2. Use it to quickly verify you are following the rules from the main guide.
3. Use it before and after each implementation step.

### 4. Phase Index

Read fourth:

`ai-code-generation-phase-checklists-index.md`

Why:

1. It tells you which phase checklist to use.
2. It enforces stage-by-stage development.
3. It prevents starting later-stage work too early.

### 5. Current Phase Checklist

Then read the relevant current phase file:

1. `ai-code-generation-phase-0-checklist.md`
2. `ai-code-generation-phase-1-checklist.md`
3. `ai-code-generation-phase-2-checklist.md`
4. `ai-code-generation-phase-3-checklist.md`
5. `ai-code-generation-phase-4-checklist.md`
6. `ai-code-generation-phase-5-checklist.md`
7. `ai-code-generation-phase-6-checklist.md`
8. `ai-code-generation-phase-7-checklist.md`

Why:

1. Each phase has a specific target.
2. Each phase defines what outputs must exist before moving on.
3. You should only work on one phase at a time.

## File Roles

### `deep-research-report-2.md`

Role:

1. Research vision document.
2. Defines the scientific problem, candidate direction, technical route, experiments, and publication path.

Use it when:

1. You need to understand what the project is really about.
2. You need to map code to research questions.
3. You need to decide whether a proposed implementation matches the intended paper direction.

### `ai-code-generation-guide-design.md`

Role:

1. Design document for the main code-generation guide.
2. Historical planning artifact.

Use it when:

1. You need to understand why the guide was structured this way.
2. You are editing the main guide itself.

Normally, prefer `ai-code-generation-guide.md` over this file for active implementation decisions.

### `ai-code-generation-guide-writing-plan.md`

Role:

1. Writing plan used to create the main guide.
2. Historical execution artifact.

Use it when:

1. You are editing or extending the guide documents themselves.

Normally, you do not need this file for repository implementation work.

### `ai-code-generation-guide.md`

Role:

1. Primary build manual.
2. Source of truth for architecture, workflow, module boundaries, prompts, and acceptance criteria.

Use it when:

1. You are planning or implementing code.
2. You need to know what module should exist.
3. You need to know how to stage development.
4. You need to know what not to do.

### `ai-code-generation-guide-one-page-checklist.md`

Role:

1. Quick operating checklist.
2. Fast compliance check before and after edits.

Use it when:

1. You want a short reminder of hard rules.
2. You are about to start a task or verify a task is done.

### `ai-code-generation-phase-checklists-index.md`

Role:

1. Phase navigation file.
2. Tells you which phase checklist to load.

Use it when:

1. You need to know where the project currently is.
2. You need to choose the next checklist.

### `ai-code-generation-phase-*-checklist.md`

Role:

1. Stage-specific execution checklists.
2. Defines required outputs, constraints, verification, and exit conditions for each phase.

Use them when:

1. You are actively building the repository.
2. You need to know exactly what must exist before moving to the next phase.

## Working Rules

You must follow these rules in this directory.

### 1. Read Before Edit

Before making any change:

1. Read the research plan.
2. Read the main guide.
3. Read the current phase checklist.
4. Read any file you are about to modify.

Do not edit first and rationalize later.

### 2. One Focused Task At A Time

Do not mix these in one step:

1. New module implementation.
2. Large refactor.
3. Experiment protocol redesign.
4. Output format redesign.

Pick one focused task, complete it, verify it, then move on.

### 3. Minimal Correct Changes

Prefer:

1. Small diffs.
2. Clear interfaces.
3. Minimal abstraction.
4. Reusing existing structure.

Avoid:

1. Speculative plugin systems.
2. Parallel architectures.
3. “Future-proofing” that increases complexity now.

### 4. Respect Module Boundaries

Do not:

1. Put algorithm logic into `scripts/`.
2. Put plotting logic into core training code.
3. Put evaluation logic directly inside training loops.
4. Mix query generation with verification decision logic.
5. Couple attack execution to the main training path.

### 5. Verify Every Step

Every non-trivial change should end with:

1. A listed verification command.
2. The actual result.
3. A statement of whether the phase exit condition is met.

If verification fails, stop and debug. Do not claim completion.

### 6. Phase Discipline

This project must be built in phases.

Rules:

1. Do not start Phase 3 if Phase 2 is not actually working.
2. Do not start attacks before verification outputs exist.
3. Do not start reporting before evaluation outputs are standardized.
4. Do not skip exit conditions.

## Expected Future Repository Shape

When code generation begins, the target repository should roughly follow this shape:

```text
repo/
  configs/
  data/
  scripts/
  src/
    federated/
    datasets/
    models/
    watermarking/
    allocation/
    verification/
    attacks/
    evaluation/
    reporting/
  tests/
  outputs/
  docs/
```

Core intended module responsibilities:

1. `federated`: FL loop and lifecycle.
2. `datasets`: loading and non-IID split generation.
3. `models`: task backbones and model registry.
4. `watermarking`: owner codebooks and query artifacts.
5. `allocation`: adaptability and budget assignment.
6. `verification`: black-box scoring and decisions.
7. `attacks`: post-training attacks.
8. `evaluation`: metrics and experiment summaries.
9. `reporting`: figures, tables, reports.

## What This Project Is Not

Do not misread the goal.

This is not primarily:

1. A generic FL demo.
2. A production deployment system.
3. A blockchain project.
4. A cryptography-first implementation benchmark.
5. A notebook-only experiment dump.

It is a research-code project for a paper-oriented federated learning copyright-verification method.

## Task Start Template For Agents

When starting any task in this directory, begin with this structure:

1. State the current phase.
2. State the exact goal of this task.
3. List the files you will read first.
4. List the files you expect to modify.
5. State the verification command you will run.

Example:

```md
Current phase: Phase 2
Task goal: Implement the minimal FedAvg training loop for CIFAR-10.
Files to read first:
- deep-research-report-2.md
- ai-code-generation-guide.md
- ai-code-generation-phase-2-checklist.md
- existing training/config files
Files to modify:
- <file list>
Verification:
- <command>
```

## If You Need To Modify Documentation

If your task is to update the guide documents themselves:

1. Prefer editing `ai-code-generation-guide.md` only if the main operating rules are changing.
2. Update the one-page checklist if the high-level rules changed.
3. Update phase checklists if the phase boundaries or exit conditions changed.
4. Keep consistency across all guidance files.

## Final Reminder

This directory is guidance-first.

Before writing code, understand:

1. What the research problem is.
2. What phase the project is in.
3. What files define the rules.
4. What exactly must be verified before moving on.

If you skip the documentation and improvise, you are likely to damage the intended architecture.
