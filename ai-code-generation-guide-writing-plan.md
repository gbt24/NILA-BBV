# AI Code Generation Guide Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Write a complete, executable AI code generation guide that can drive OpenCode/Claude Code to produce a full, runnable, submission-grade research repository for the federated learning copyright-verification project.

**Architecture:** Build one primary Markdown handbook in the current directory from the approved design spec. Write the guide section-by-section so each block includes goals, hard rules, agent operating instructions, reusable prompts, and acceptance criteria. Finish with a self-review pass to remove ambiguity and ensure the guide is directly executable by code agents.

**Tech Stack:** Markdown, existing design spec, current research plan

---

### Task 1: Establish Final Guide Skeleton

**Files:**
- Create: `/Users/gbt24/Documents/paper/2026上半年讨论会/bbv/ai-code-generation-guide.md`
- Read: `/Users/gbt24/Documents/paper/2026上半年讨论会/bbv/ai-code-generation-guide-design.md`
- Read: `/Users/gbt24/Documents/paper/2026上半年讨论会/bbv/deep-research-report-2.md`

- [ ] **Step 1: Re-read the approved design and research plan**

Read these files before drafting:

`/Users/gbt24/Documents/paper/2026上半年讨论会/bbv/ai-code-generation-guide-design.md`

`/Users/gbt24/Documents/paper/2026上半年讨论会/bbv/deep-research-report-2.md`

Expected: The guide structure, target audience, and research-module mapping are clear.

- [ ] **Step 2: Create the final guide scaffold**

Create `ai-code-generation-guide.md` with these top-level sections:

```md
# AI代码生成指南

## 1. 指南目标与适用范围
## 2. 研究问题到代码系统的映射
## 3. 技术栈与总体架构
## 4. 仓库目录规范
## 5. 模块级代码生成指南
## 6. 分阶段实现路线
## 7. AI代理代码生成规范
## 8. 代码质量与实验规范
## 9. Git管理规范
## 10. 代理提示模板库
## 11. 失败恢复与重试策略
## 12. 最终验收清单
```

Expected: File exists with all target sections in order.

- [ ] **Step 3: Verify the scaffold file exists and is readable**

Read: `/Users/gbt24/Documents/paper/2026上半年讨论会/bbv/ai-code-generation-guide.md`

Expected: The file contains the exact 12 sections listed above.


### Task 2: Write Guide Foundations

**Files:**
- Modify: `/Users/gbt24/Documents/paper/2026上半年讨论会/bbv/ai-code-generation-guide.md`

- [ ] **Step 1: Draft Sections 1-3**

Write complete content for:

1. `指南目标与适用范围`
2. `研究问题到代码系统的映射`
3. `技术栈与总体架构`

The content must explicitly define:

```md
- The guide is for OpenCode/Claude Code style code agents.
- The target output is a submission-grade research repository.
- The repository serves the non-IID low-ambiguity adaptive black-box copyright verification project.
- The code system is decomposed into federated core, data, models, watermark/fingerprint, adaptive allocation, verification, attacks, evaluation, and reporting.
- PyTorch is the base layer, FedLab is the primary research engine, and Flower is the later system-extension layer.
```

Expected: A reader can understand what the guide is for and how the research problem maps to code.

- [ ] **Step 2: Draft Section 4**

Write the repository structure section with a concrete directory tree similar to:

```md
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

For each directory, explain a single responsibility and what must not be stored there.

Expected: The guide prevents future repository drift.

- [ ] **Step 3: Read back Sections 1-4 and check continuity**

Read the updated guide and verify:

1. Section 1 defines audience and scope.
2. Section 2 maps research to modules.
3. Section 3 matches the module design.
4. Section 4 reflects the same architecture in file layout.

Expected: No contradiction between architecture and directory structure.


### Task 3: Write Module Generation Rules

**Files:**
- Modify: `/Users/gbt24/Documents/paper/2026上半年讨论会/bbv/ai-code-generation-guide.md`

- [ ] **Step 1: Draft Section 5 with a repeated per-module template**

For each module below, write subsections using the same structure:

1. Module purpose
2. Inputs and outputs
3. Key interfaces
4. File split rules
5. Agent generation order
6. Testing requirements
7. Common failure modes
8. Copyable agent prompt template
9. Definition of done

Modules to cover:

```md
- federated core
- data
- models
- watermark/fingerprint
- adaptive allocation
- verification
- attacks
- evaluation
- reporting
```

Expected: The guide can be used module-by-module rather than only top-down.

- [ ] **Step 2: Make the prompt templates concrete**

Each module must include a prompt block like:

```md
请阅读现有仓库中与 <module-name> 相关的文件，只做最小正确改动，实现 <target>。
要求：
1. 先给出将修改/新增的文件列表。
2. 保持现有目录结构，不新增不必要抽象。
3. 先实现接口和最小测试，再补完整实现。
4. 完成后运行指定验证命令并报告结果。
```

Expected: A user can copy prompts directly into OpenCode/Claude Code.

- [ ] **Step 3: Verify module coverage**

Read Section 5 and confirm each module from the design doc appears exactly once as a first-level subsection inside the module chapter.

Expected: No module is missing, duplicated, or underspecified.


### Task 4: Write Phase Plan and Agent Rules

**Files:**
- Modify: `/Users/gbt24/Documents/paper/2026上半年讨论会/bbv/ai-code-generation-guide.md`

- [ ] **Step 1: Draft Section 6 with explicit phase gates**

Write phases `0` through `7`. For each phase, include:

1. Objective
2. Preconditions
3. Work items
4. Expected files or outputs
5. Verification commands
6. Exit criteria

Expected: The reader knows exactly when a phase is complete and can move on.

- [ ] **Step 2: Draft Sections 7 and 8**

Section 7 must define hard rules for agent-generated code, including:

```md
- read before edit
- minimal diffs
- one focused task per iteration
- no speculative abstractions
- mandatory verification after each change
- ask when research assumptions are unclear
```

Section 8 must define quality and experiment norms, including:

```md
- config naming
- seeds and reproducibility
- checkpoint and metrics storage
- test expectations
- plotting and table export rules
- experiment registry / run naming
```

Expected: The guide constrains both code shape and experiment discipline.

- [ ] **Step 3: Read Sections 6-8 and check for execution gaps**

Verify that every phase has at least one matching rule from Sections 7-8 that constrains how the phase should be executed.

Expected: The phase plan and agent rules reinforce each other.


### Task 5: Write Git, Templates, Recovery, and Acceptance

**Files:**
- Modify: `/Users/gbt24/Documents/paper/2026上半年讨论会/bbv/ai-code-generation-guide.md`

- [ ] **Step 1: Draft Section 9**

Write Git rules covering:

```md
- branch naming
- commit size
- commit message style
- separating algorithm changes from experiment-result updates
- pre-commit verification requirements
- PR summary expectations
```

Expected: Git usage is consistent with research-code iteration.

- [ ] **Step 2: Draft Section 10**

Create a prompt template library with at least these templates:

1. repository bootstrap
2. module implementation
3. refactor under constraints
4. add tests only
5. debug failing experiment
6. run evaluation and summarize outputs
7. pre-release verification

Expected: The user has reusable prompts for common agent workflows.

- [ ] **Step 3: Draft Sections 11-12**

Section 11 must define failure recovery paths for:

```md
- test failure
- architecture drift
- duplicate code
- broken experiment script
- unstable results
- mismatch between implementation and research plan
```

Section 12 must define a final acceptance checklist for a submission-grade repository.

Expected: The guide ends with actionable recovery and release criteria.


### Task 6: Final Review and Publish Readiness

**Files:**
- Read: `/Users/gbt24/Documents/paper/2026上半年讨论会/bbv/ai-code-generation-guide.md`
- Read: `/Users/gbt24/Documents/paper/2026上半年讨论会/bbv/ai-code-generation-guide-design.md`

- [ ] **Step 1: Run a spec coverage review**

Check every major promise from `ai-code-generation-guide-design.md` against the final guide:

1. audience
2. module mapping
3. dual-framework architecture
4. repository structure
5. module generation guidance
6. phase roadmap
7. AI coding rules
8. quality rules
9. Git rules
10. prompt templates
11. recovery paths
12. acceptance checklist

Expected: Every design commitment has a corresponding section in the guide.

- [ ] **Step 2: Run a placeholder and ambiguity scan**

Read the guide and remove:

```md
TBD
TODO
future work
later add
appropriate handling
similar to above
```

Also rewrite any vague sentence that a code agent could interpret in multiple ways.

Expected: The guide is directly executable without hidden assumptions.

- [ ] **Step 3: Confirm final file locations**

Verify that these files exist in the current directory:

1. `/Users/gbt24/Documents/paper/2026上半年讨论会/bbv/ai-code-generation-guide-design.md`
2. `/Users/gbt24/Documents/paper/2026上半年讨论会/bbv/ai-code-generation-guide-writing-plan.md`
3. `/Users/gbt24/Documents/paper/2026上半年讨论会/bbv/ai-code-generation-guide.md`

Expected: The design, plan, and final guide are co-located and ready for continued iteration.
