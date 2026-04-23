# LEAF Dataset Download Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 自动下载并预处理 FEMNIST 与 Sent140 到 `data/raw/`，然后让这两个数据集的训练命令和完整 final gate 可在本地通过。

**Architecture:** 保持现有 loader 只读本地数据，新增独立数据准备脚本处理下载、解压、转换和分片。脚本输出严格对齐当前 `leaf_femnist.py` 和 `text.py` 的读取格式，不修改训练主路径。

**Tech Stack:** Python standard library, urllib, zipfile/tarfile/csv/json, pytest, existing `bbv` scripts.

---

### Task 1: Implement FEMNIST preparation script

**Files:**
- Create: `scripts/data/prepare_femnist.py`
- Test: `tests/smoke/test_prepare_leaf_datasets.py`

- [ ] **Step 1: Write the failing test**
- [ ] **Step 2: Run test to verify it fails**
- [ ] **Step 3: Implement download, extract, convert to `data/raw/femnist/{train,test}`**
- [ ] **Step 4: Run test to verify it passes**
- [ ] **Step 5: Commit**

### Task 2: Implement Sent140 preparation script

**Files:**
- Create: `scripts/data/prepare_sent140.py`
- Test: `tests/smoke/test_prepare_leaf_datasets.py`

- [ ] **Step 1: Write the failing test**
- [ ] **Step 2: Run test to verify it fails**
- [ ] **Step 3: Implement download, extract, convert to `data/raw/sent140/{train,test}`**
- [ ] **Step 4: Run test to verify it passes**
- [ ] **Step 5: Commit**

### Task 3: Implement unified entrypoint and rerun verification gate

**Files:**
- Create: `scripts/data/prepare_leaf_datasets.py`
- Modify: `README.md`
- Test: `tests/smoke/test_prepare_leaf_datasets.py`

- [ ] **Step 1: Write the failing test**
- [ ] **Step 2: Run test to verify it fails**
- [ ] **Step 3: Implement unified entrypoint and docs**
- [ ] **Step 4: Run smoke tests plus FEMNIST/Sent140 training commands**
- [ ] **Step 5: Re-run full final verification gate and commit**
