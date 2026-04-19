# AI代码生成指南

## 1. 指南目标与适用范围

### 1.1 目标

这份指南用于指导 `OpenCode`、`Claude Code` 一类具备读写代码、运行命令、分阶段修改仓库能力的代码代理，围绕当前研究计划生成一个完整、可运行、可复现、可持续扩展的联邦学习版权验证研究仓库。

本指南的直接目标不是“帮代理写几个模块”，而是让代理在统一规则下完成以下工作：

1. 搭建可运行的联邦学习研究仓库。
2. 实现面向 non-IID 场景的 watermark/fingerprint 与 black-box verification 主流程。
3. 支撑主实验、消融实验、攻击评测、统计分析和图表导出。
4. 在整个过程中维持清晰目录、稳定接口、可追踪实验和可审计 Git 历史。

### 1.2 适用对象

适用对象只有两类：

1. 使用代码代理驱动研究代码生成的研究者。
2. 需要将研究计划稳定转译为可执行仓库的代码代理。

### 1.3 目标仓库级别

本指南默认目标是“投稿级完整仓库”，至少应达到以下标准：

1. 新环境可安装并跑通最小训练闭环。
2. 主任务、联邦训练、版权验证、攻击评测、统计分析可分别执行。
3. 主结果、消融、鲁棒性评测和图表可以复现。
4. 代码、配置、日志、结果和文档边界清晰。

### 1.4 非目标范围

本指南首版不追求以下内容：

1. 生产级多机部署与运维。
2. 重密码学协议或区块链系统实现。
3. 任意 ML 项目的通用生成框架。
4. 代替论文正文或开题报告写作。

## 2. 研究问题到代码系统的映射

当前研究计划围绕“面向非IID联邦学习的低歧义自适应黑盒版权验证”，代码系统必须围绕研究问题组织，而不是围绕通用技术层堆砌。

### 2.1 研究问题映射

1. `RQ1: 如何估计客户端 watermark adaptability`
对应模块：`allocation`、`federated`、`datasets`。

2. `RQ2: 如何把黑盒验证从 trigger 命中升级为 owner codebook 匹配`
对应模块：`watermarking`、`verification`。

3. `RQ3: 如何统一 ambiguity/FPR/FNR/utility`
对应模块：`evaluation`、`reporting`。

4. `RQ4-RQ5: 查询成本、攻击鲁棒性与迁移行为`
对应模块：`attacks`、`verification`、`evaluation`。

### 2.2 代码系统划分

仓库必须至少拆为 9 个核心代码域：

1. `federated core`
职责：联邦训练循环、client/server 生命周期、聚合策略、仿真运行入口。

2. `data`
职责：数据下载、预处理、联邦划分、non-IID 生成、缓存与 split 元信息保存。

3. `models`
职责：视觉与文本任务 backbone、分类头、统一模型注册。

4. `watermark/fingerprint`
职责：owner codebook、正证据集、负证据集、查询样本、基线嵌入逻辑。

5. `adaptive allocation`
职责：适配度估计、客户端打分、watermark depth 与 loss 权重调度。

6. `verification`
职责：黑盒查询、码字恢复、owner score、margin 判决、阈值校准。

7. `attacks`
职责：fine-tuning、pruning、quantization、distillation、model extraction。

8. `evaluation`
职责：实验汇总、AUC/FPR/FNR/ambiguity 统计、消融与对比实验。

9. `reporting`
职责：图表、表格、结果摘要、复现实验导出。

### 2.3 系统边界原则

代理在实现时必须遵守：

1. 训练逻辑不能直接耦合图表生成。
2. watermark/fingerprint 逻辑不能直接写在 federated loop 里。
3. evaluation 只消费标准化输出，不直接依赖训练过程中的临时变量。
4. attacks 必须可在训练后单独运行。

## 3. 技术栈与总体架构

### 3.1 基础选型

本指南采用双框架策略：

1. `PyTorch`：唯一深度学习基础框架。
2. `FedLab`：研究实验主框架，优先承担联邦模拟、non-IID 划分与复现实验。
3. `Flower`：扩展接口层，用于后续更接近真实 client-server 的实验形态。
4. `Hydra`：配置中心。
5. `pytest`：测试框架。
6. `ruff`、`black`、`mypy`：格式化、静态检查、基础类型检查。
7. `pandas`、`scipy`、`matplotlib`、`seaborn`：统计与可视化。

### 3.2 总体架构原则

1. 所有运行入口必须走配置，不允许把实验参数散落在脚本内部。
2. 所有模块必须有清晰输入输出对象，避免“训练函数返回 12 个杂项”。
3. 所有实验结果必须序列化到标准结果目录，供 evaluation 与 reporting 重用。
4. Flower 适配层不能反过来污染研究主代码；研究主代码以 FedLab 风格抽象为主。

### 3.3 运行流

建议统一为以下主流程：

1. 读取配置。
2. 解析数据集与 split。
3. 初始化任务模型与联邦训练器。
4. 初始化 watermark/fingerprint 或 adaptive allocation 组件。
5. 执行训练。
6. 导出 checkpoint、训练日志、verification 结果。
7. 运行 attacks。
8. 运行 evaluation 与 reporting。

### 3.4 代理执行要求

本节对应的代理执行块：

```md
请先阅读现有仓库的配置系统、运行入口和模块注册方式。不要直接新增第二套实验框架。
要求：
1. 以 PyTorch 为唯一底层实现。
2. 以 FedLab 风格组织主研究代码。
3. Flower 只作为扩展接口，不改写研究主流程。
4. 所有新增功能必须接入统一配置与输出目录。
```

### 3.5 验收标准

1. 能说明每一类库的唯一职责。
2. 能从配置驱动一次最小实验。
3. 不存在互相竞争的第二套运行逻辑。

## 4. 仓库目录规范

推荐仓库目录如下：

```text
repo/
  README.md
  pyproject.toml
  configs/
    defaults/
    datasets/
    models/
    federated/
    watermarking/
    attacks/
    evaluation/
  data/
    raw/
    processed/
    splits/
  scripts/
    train/
    attacks/
    eval/
    report/
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
    utils/
  tests/
    unit/
    integration/
    smoke/
  outputs/
    runs/
    figures/
    tables/
    summaries/
  docs/
    specs/
    plans/
    notes/
```

### 4.1 目录职责

1. `configs/`
只放配置，不放运行脚本、分析结果或 Python 逻辑。

2. `data/`
只放数据、缓存与 split 元信息，不放人工编辑代码。

3. `scripts/`
只放薄运行入口，不放核心算法实现。

4. `src/`
只放核心实现代码，不放临时 notebook 输出。

5. `tests/`
只放测试，不复制主实现。

6. `outputs/`
只放实验产物，不提交大模型权重或未筛选临时垃圾文件。

7. `docs/`
只放设计、计划、记录与研究文档。

### 4.2 目录边界硬规则

1. 不允许把论文图表逻辑混进 `src/evaluation/` 的核心统计函数里。
2. 不允许把数据切分代码写进训练脚本。
3. 不允许在 `scripts/` 中复制 `src/` 逻辑。
4. 不允许让 `outputs/` 成为隐式配置源。

### 4.3 代理执行块

```md
请先阅读现有目录，再进行最小增量修改。
要求：
1. 所有新增文件必须放在现有职责最匹配的目录。
2. 不允许为了省事把算法写进 scripts。
3. 不允许创建含义重复的新顶层目录。
4. 如果发现目录职责不清，先提出重构建议，再修改。
```

### 4.4 验收标准

1. 任意新文件都能解释为什么在该目录。
2. 同类职责没有分散到多个路径。
3. 训练、评测、输出、文档四类内容边界清晰。

## 5. 模块级代码生成指南

本节是代理生成代码的核心部分。每个模块都必须按同一模板推进：目的、输入输出、关键接口、文件拆分、生成顺序、测试要求、常见失败、提示模板、完成定义。

### 5.1 Federated Core

目的：提供统一联邦训练抽象与最小可运行训练闭环。

输入输出：输入为模型、客户端数据、训练配置；输出为全局模型、轮次指标、checkpoint。

关键接口：`build_server()`、`build_client()`、`run_round()`、`train_federated()`。

文件拆分：客户端逻辑、服务端逻辑、聚合器、训练入口分离。

代理生成顺序：
1. 先建最小训练循环。
2. 再加聚合策略与日志。
3. 最后接 watermark 与 allocation 钩子。

测试要求：
1. 单轮联邦训练 smoke test。
2. 聚合结果 shape test。
3. 配置加载 integration test。

常见失败：
1. 把 watermark 逻辑硬编码进训练循环。
2. 把数据切分与联邦训练耦合。
3. server/client 状态对象过大。

提示模板：

```md
请阅读 federated core 相关文件，只做最小正确改动，实现 <目标功能>。
要求：
1. 先列出将修改/新增的文件。
2. 保持训练循环与 watermark 逻辑解耦。
3. 先补最小 smoke test，再补实现。
4. 完成后运行联邦训练最小验证命令并报告结果。
```

完成定义：最小联邦训练在单数据集单配置下可运行，并为后续模块留出明确钩子。

### 5.2 Data

目的：统一数据集加载、联邦划分与 non-IID split 生成。

输入输出：输入为数据集名、split 配置；输出为标准化 client partitions 与 split metadata。

关键接口：`load_dataset()`、`build_partition()`、`save_split_metadata()`、`load_split_metadata()`。

文件拆分：数据下载、预处理、split 生成、split 注册分开。

代理生成顺序：
1. 先支持一个视觉数据集。
2. 再加 Dirichlet/shard split。
3. 再加 natural split 和 feature skew。

测试要求：分区样本数守恒、split 可复现、metadata 可读回。

常见失败：把 split 写死在 notebook；不保存 seed；split 无法复用。

提示模板：

```md
请阅读 data 模块相关文件，只做最小正确改动，实现 <dataset/split 功能>。
要求：
1. 先说明新增的 split 类型和涉及文件。
2. 不要把划分逻辑写进训练脚本。
3. 保存 split metadata 和随机种子。
4. 完成后运行 split smoke test 并报告结果。
```

完成定义：至少一种数据集和一种 non-IID 划分可稳定复现。

### 5.3 Models

目的：提供可替换 backbone 和统一模型注册接口。

输入输出：输入为模型名与任务配置；输出为标准化模型实例。

关键接口：`build_model()`、`register_model()`、`forward_for_task()`。

文件拆分：视觉模型、文本模型、注册表分开。

代理生成顺序：先一个视觉 baseline，再文本 baseline，再扩展注册表。

测试要求：前向 shape、分类头输出维度、注册表解析。

常见失败：任务逻辑直接写入 backbone；模型选择依赖 if-else 污染训练入口。

提示模板：

```md
请阅读 models 相关文件，只做最小正确改动，实现 <模型/注册能力>。
要求：
1. 保持统一 build_model 接口。
2. 不把任务专用逻辑耦合进基础 backbone。
3. 先补前向与注册测试，再补实现。
4. 完成后报告最小模型构建验证结果。
```

完成定义：模型可通过统一配置构建并进入训练流程。

### 5.4 Watermark/Fingerprint

目的：实现 owner codebook、查询样本、正负证据集和基线嵌入逻辑。

输入输出：输入为 owner 配置、辅助数据池、seed；输出为 codebook、query sets、embedding artifacts。

关键接口：`generate_codebook()`、`build_positive_queries()`、`build_negative_queries()`、`prepare_owner_artifacts()`。

文件拆分：codebook、query construction、embedding hooks、baselines 分开。

代理生成顺序：
1. 先 zero-bit 或最小 codebook baseline。
2. 再加正负证据集。
3. 再加多比特与可扩展接口。

测试要求：码长正确、查询集大小符合配置、owner 间互异性检查。

常见失败：把 query generation 与 verification 混在一起；无 owner metadata；负证据集缺失。

提示模板：

```md
请阅读 watermark/fingerprint 相关文件，只做最小正确改动，实现 <codebook 或 query 功能>。
要求：
1. 先列出涉及的 owner metadata、query artifacts 和接口。
2. 不要把 verification 判决写到该模块。
3. 先补码长/集合大小/唯一性测试，再补实现。
4. 完成后报告生成样例和验证命令结果。
```

完成定义：能为至少一个 owner 生成可复用的查询工件，并供 verification 消费。

### 5.5 Adaptive Allocation

目的：根据客户端适配度动态分配 watermark depth 与 loss 权重。

输入输出：输入为客户端统计量、梯度或代理指标；输出为 allocation scores 与 per-client assignment。

关键接口：`estimate_adaptability()`、`allocate_watermark_budget()`、`update_assignment()`。

文件拆分：指标计算、分配策略、训练钩子分开。

代理生成顺序：先静态评分，再动态更新，再接 federated loop。

测试要求：分数范围检查、预算守恒、assignment 可解释性。

常见失败：直接读取过多私有信息；分配逻辑散落在训练循环；无法关闭该模块做消融。

提示模板：

```md
请阅读 adaptive allocation 相关文件，只做最小正确改动，实现 <适配度估计或分配策略>。
要求：
1. 先给出输入统计量、输出 assignment 和涉及文件。
2. 保持该模块可单独开关，便于消融。
3. 先补预算守恒和分数范围测试，再补实现。
4. 完成后运行最小 allocation 验证并报告结果。
```

完成定义：allocation 策略可独立启停，并能稳定接入训练过程。

### 5.6 Verification

目的：完成黑盒查询、码字恢复、owner score、margin 判决和阈值校准。

输入输出：输入为模型查询结果与 owner artifacts；输出为 recovered codeword、owner scores、accept/reject 决策。

关键接口：`query_model()`、`recover_codeword()`、`compute_owner_score()`、`verify_owner()`、`calibrate_threshold()`。

文件拆分：query execution、score computation、calibration、decision logic 分开。

代理生成顺序：先恢复与评分，再阈值，再 margin，再批量评测入口。

测试要求：score 单调性、阈值输出有效、non-owner 被拒概率基本合理。

常见失败：把 calibration 写死；同时依赖训练内部状态；正证据和负证据未统一入分数。

提示模板：

```md
请阅读 verification 相关文件，只做最小正确改动，实现 <评分/校准/判决功能>。
要求：
1. 先列出输入的 query 结果、owner artifacts 和输出指标。
2. 不要依赖训练过程中的临时变量。
3. 先补 score 与 calibration 测试，再补实现。
4. 完成后运行最小 verification 命令并报告结果。
```

完成定义：给定可疑模型和 owner artifacts，可以输出稳定的 accept/reject 与相关指标。

### 5.7 Attacks

目的：实现训练后攻击与鲁棒性评测入口。

输入输出：输入为 checkpoint、攻击配置；输出为 attacked checkpoints 与攻击日志。

关键接口：`run_finetune_attack()`、`run_pruning_attack()`、`run_quantization_attack()`、`run_distillation_attack()`。

文件拆分：每种攻击独立文件，共享公共工具。

代理生成顺序：先 fine-tuning，再 pruning/quantization，再 distillation/extraction。

测试要求：攻击输出 checkpoint 存在、主任务性能变化在可解释范围、日志完整。

常见失败：攻击脚本直接复用训练脚本副作用；不保存攻击配置；结果无法回溯。

提示模板：

```md
请阅读 attacks 相关文件，只做最小正确改动，实现 <攻击类型>。
要求：
1. 先说明输入 checkpoint、输出目录和日志格式。
2. 不要把攻击逻辑写回主训练入口。
3. 先补输出存在性和日志测试，再补实现。
4. 完成后报告最小攻击运行结果。
```

完成定义：攻击流程可在训练后独立运行，并产出标准化结果。

### 5.8 Evaluation

目的：统一主结果、消融、攻击结果和统计指标计算。

输入输出：输入为训练结果、verification 输出、attack 输出；输出为结构化指标表。

关键接口：`summarize_main_results()`、`compute_ambiguity_metrics()`、`compute_robustness_metrics()`、`export_metrics_table()`。

文件拆分：指标计算、实验汇总、统计检验分开。

代理生成顺序：先汇总，再指标，再统计检验。

测试要求：指标列完整、输入缺失时有明确错误、导出表结构稳定。

常见失败：evaluation 直接重新跑训练；输出 schema 漂移；统计逻辑混入绘图代码。

提示模板：

```md
请阅读 evaluation 相关文件，只做最小正确改动，实现 <指标汇总/统计功能>。
要求：
1. 先列出输入结果文件和输出表 schema。
2. 不要在 evaluation 中重写训练逻辑。
3. 先补 schema 与缺失输入测试，再补实现。
4. 完成后运行最小结果汇总命令并报告输出路径。
```

完成定义：evaluation 可以稳定消费标准结果并产出主表。

### 5.9 Reporting

目的：生成论文图表、结果摘要和复现报告。

输入输出：输入为 evaluation 输出；输出为 figures、tables、summaries。

关键接口：`plot_main_results()`、`plot_tradeoff_curve()`、`render_summary_report()`。

文件拆分：绘图函数、表格导出、报告模板分开。

代理生成顺序：先表格导出，再主图，再 summary。

测试要求：图文件存在、尺寸和命名稳定、表格可读取。

常见失败：绘图直接读取原始训练日志；图表风格不可重复；输出命名随意。

提示模板：

```md
请阅读 reporting 相关文件，只做最小正确改动，实现 <图表/摘要功能>。
要求：
1. 先说明输入表、输出图和命名规则。
2. 只消费 evaluation 的标准输出，不直接解析训练过程临时文件。
3. 先补文件生成测试，再补实现。
4. 完成后报告输出目录和样例文件名。
```

完成定义：论文主图主表可从 evaluation 结果自动生成。

## 6. 分阶段实现路线

每个阶段必须满足“目标、前置条件、工作项、产物、验证命令、退出条件”六要素。未通过退出条件，禁止进入下一阶段。

### Phase 0 仓库初始化与最小闭环

目标：建立基础仓库、依赖、配置系统和最小训练入口。

前置条件：无。

工作项：初始化目录、依赖文件、README、最小脚本、最小测试。

产物：`pyproject.toml`、基础目录、`scripts/train/run_smoke.py`、最小 smoke test。

验证命令：`pytest tests/smoke -q`。

退出条件：新环境可安装并跑通最小 smoke test。

### Phase 1 数据层与 non-IID 划分

目标：支持至少一个视觉数据集和一种 non-IID 划分。

前置条件：Phase 0 通过。

工作项：数据加载、split 生成、metadata 缓存、split 测试。

产物：数据模块、split 配置、split metadata。

验证命令：`pytest tests/unit/test_splits.py -q`。

退出条件：同一 seed 下 split 可复现，样本数守恒。

### Phase 2 主任务训练与联邦基线

目标：跑通 FedAvg 或同类联邦基线。

前置条件：Phase 1 通过。

工作项：模型注册、联邦训练循环、日志记录。

产物：训练结果目录、checkpoint、基线指标。

验证命令：`python scripts/train/run_fedavg.py dataset=cifar10 federated=fedavg model=resnet18 seed=0`。

退出条件：基线任务在一个数据集上稳定收敛。

### Phase 3 Watermark/Fingerprint 基线

目标：实现最小版权验证基线。

前置条件：Phase 2 通过。

工作项：owner artifacts、query generation、最小 verification 路径。

产物：codebook/query 工件、verification 输出。

验证命令：`python scripts/train/run_watermark_baseline.py dataset=cifar10 watermarking=baseline owner.id=owner0 seed=0`。

退出条件：给定 owner 与模型可输出稳定验证结果。

### Phase 4 Adaptive Allocation

目标：引入 non-IID 自适应分配。

前置条件：Phase 3 通过。

工作项：适配度估计、分配策略、训练钩子、消融开关。

产物：assignment 日志、allocation 配置、对照结果。

验证命令：`pytest tests/integration/test_allocation_pipeline.py -q`。

退出条件：allocation 可启停，结果可记录并进入后续评测。

### Phase 5 Black-Box Verification 与校准

目标：实现 score、margin、threshold calibration。

前置条件：Phase 4 通过。

工作项：恢复码字、评分函数、阈值校准、ambiguity 指标。

产物：verification summary、calibration artifacts、accept/reject 结果。

验证命令：`python scripts/eval/run_verification.py dataset=cifar10 verification=margin owner.id=owner0 seed=0`。

退出条件：owner/non-owner 可以被区分，并能导出 calibration 结果。

### Phase 6 攻击与鲁棒性评测

目标：支持常见模型攻击并评估鲁棒性。

前置条件：Phase 5 通过。

工作项：fine-tuning、pruning、quantization、至少一种蒸馏或提取。

产物：attacked checkpoints、攻击日志、鲁棒性指标。

验证命令：`python scripts/attacks/run_attack_suite.py attack=finetune dataset=cifar10 checkpoint=outputs/runs/cifar10-fedavg-seed0/checkpoints/best.pt seed=0`。

退出条件：攻击后仍能跑完整 verification-evaluation 流程。

### Phase 7 统计分析、图表与复现实验导出

目标：形成投稿级输出。

前置条件：Phase 6 通过。

工作项：主表、主图、消融图、tradeoff 曲线、summary report。

产物：`outputs/figures/`、`outputs/tables/`、`outputs/summaries/`。

验证命令：`python scripts/report/build_report.py dataset=cifar10 study=main outputs_dir=outputs/runs/cifar10-main-seed0`。

退出条件：主图主表可一键导出，README 提供复现实验命令。

## 7. AI代理代码生成规范

### 7.1 硬规则

1. 先读相关文件，再修改。
2. 每次只做一个聚焦目标。
3. 优先最小正确实现，不引入猜测性抽象。
4. 先接口与测试，再补实现。
5. 每次修改后必须运行最小验证命令。
6. 不清楚研究设定时必须提问，不得自行发明。

### 7.2 禁止行为

1. 未读现有模块就新建平行实现。
2. 同时改训练主循环、目录结构、实验协议三件事。
3. 为“以后可能会用”而提前设计复杂插件系统。
4. 把验证失败解释为“应该没问题”而不跑命令。

### 7.3 推荐单轮交互格式

代理每轮输出应包含：

1. 本轮目标。
2. 将修改/新增的文件。
3. 最小实现思路。
4. 验证命令。
5. 实际结果或阻塞点。

### 7.4 验收标准

任意一轮任务结束后，用户应能看到明确的文件变更边界和验证证据。

## 8. 代码质量与实验规范

### 8.1 配置规范

1. 所有实验必须可由配置复现。
2. 配置命名采用 `<task>/<dataset>/<variant>` 风格。
3. 同一实验不得把一半参数放命令行、一半写死在代码里。

### 8.2 复现规范

1. 记录随机种子。
2. 保存 split metadata。
3. 保存模型配置、训练配置、攻击配置。
4. 每次实验运行必须有唯一 run id。

### 8.3 存储规范

1. checkpoint、metrics、logs、tables、figures 分目录保存。
2. 输出目录命名要含日期、数据集、方法名、seed。
3. evaluation 只读取标准化结果文件。

### 8.4 测试规范

1. 新模块至少有一个 unit test。
2. 新流程至少有一个 smoke test。
3. 核心训练与验证链路至少有 integration test。

### 8.5 图表与表格规范

1. 图表从标准 evaluation 输出生成。
2. 图名、表名、坐标轴和 legend 使用固定命名。
3. 同一指标在不同脚本中不得使用不同列名。

### 8.6 验收标准

任一实验都应可通过配置、run id 和输出目录被完整回溯。

## 9. Git管理规范

### 9.1 分支规范

1. `feat/<module-name>`：新增模块。
2. `exp/<study-name>`：实验编排或结果整理。
3. `fix/<bug-name>`：缺陷修复。
4. `docs/<topic>`：文档更新。

### 9.2 提交粒度

1. 一个提交只做一类事。
2. 算法实现与实验结果更新不能混在同一提交。
3. 大规模格式化不得与逻辑修改混提。

### 9.3 提交信息

建议使用：

1. `feat: add federated training smoke pipeline`
2. `feat: implement codebook query generation`
3. `fix: correct verification score normalization`
4. `test: add allocation budget conservation tests`
5. `docs: add reproduction instructions`

### 9.4 提交前验证

每次提交前至少运行与本改动相关的最小验证命令，并在提交说明或 PR 描述中写明。

### 9.5 PR 说明要求

每个 PR 应说明：

1. 解决的问题。
2. 修改的模块。
3. 验证命令和结果。
4. 是否影响实验协议。
5. 是否需要补跑结果。

### 9.6 验收标准

读 Git 历史时，可以清楚区分“算法实现”“测试补充”“实验结果更新”“文档补充”。

## 10. 代理提示模板库

### 10.1 仓库初始化模板

```md
请基于当前研究计划，为联邦学习版权验证项目搭建最小可运行研究仓库。
要求：
1. 先阅读现有设计文档和目录。
2. 只创建必要目录、配置和最小运行入口。
3. 先实现 smoke test，再补脚手架代码。
4. 输出修改文件列表、验证命令和结果。
```

### 10.2 模块实现模板

```md
请阅读与 <module-name> 相关的现有文件，只做最小正确改动，实现 <target>。
要求：
1. 先说明输入、输出、接口和涉及文件。
2. 不新增不必要抽象。
3. 先写最小测试，再补实现。
4. 完成后运行指定验证命令并报告结果。
```

### 10.3 受约束重构模板

```md
请在不改变外部行为和目录职责的前提下，对 <module-name> 做最小重构。
要求：
1. 先说明当前问题和最小拆分方案。
2. 不顺手做无关清理。
3. 保留原有接口，除非明确列出迁移方案。
4. 完成后跑回归测试并报告结果。
```

### 10.4 仅补测试模板

```md
请不要修改业务逻辑，只为 <module-name> 补充测试。
要求：
1. 先分析现有接口和关键路径。
2. 至少补一个 unit test 和一个边界条件测试。
3. 不引入不必要 mock。
4. 运行相关 pytest 命令并报告结果。
```

### 10.5 调试失败实验模板

```md
当前实验在 <command> 失败，请先定位根因，再做最小修复。
要求：
1. 先复述失败现象和可能根因。
2. 只修改与根因直接相关的文件。
3. 修复后先跑失败用例，再跑最小回归验证。
4. 报告修复内容和验证结果。
```

### 10.6 评测与结果汇总模板

```md
请基于现有训练结果和 verification 输出，完成 <evaluation target>。
要求：
1. 先说明输入结果文件和目标输出表/图。
2. 不重新训练模型。
3. 保持输出 schema 稳定。
4. 完成后报告输出路径和关键指标摘要。
```

### 10.7 发布前自检模板

```md
请对当前仓库做发布前自检。
要求：
1. 检查安装、训练、verification、attacks、evaluation、reporting 入口是否完整。
2. 检查 README、配置、结果目录和测试覆盖。
3. 输出发现的问题，按严重程度排序。
4. 不直接修复，先给出修复建议清单。
```

## 11. 失败恢复与重试策略

### 11.1 测试失败

先缩小到最小失败用例，再只修该路径。禁止在未理解错误前进行大面积重构。

### 11.2 架构漂移

如果发现同类逻辑在多个目录重复出现，先停止新增代码，整理重复点，再做最小合并。

### 11.3 重复实现

若代理新建了与现有模块同职责的第二套实现，优先回收新实现到原模块，而不是长期并存。

### 11.4 实验脚本损坏

先确认是配置问题、输入数据问题还是代码问题；修复后必须重跑最小 smoke experiment。

### 11.5 结果不稳定

优先检查 seed、split metadata、checkpoint 恢复逻辑、aggregation 随机性和 evaluation schema。

### 11.6 偏离研究计划

若实现与研究计划不一致，优先回到研究计划和本指南，不允许为了代码方便重写研究问题。

### 11.7 重试原则

1. 每次只修一个根因。
2. 每次修复后都跑对应最小验证。
3. 连续两次修复失败时，必须重新分析而不是继续试错。

## 12. 最终验收清单

发布或投稿前，仓库应满足以下清单：

1. 可在新环境完成安装。
2. 至少一个数据集的最小联邦训练可运行。
3. 最小 watermark/fingerprint 基线可运行。
4. adaptive allocation 可启停并有消融开关。
5. black-box verification 能输出 owner score、threshold 或 margin 相关结果。
6. 至少一种攻击流程可独立运行。
7. evaluation 能导出主结果表。
8. reporting 能导出至少一张主图和一张主表。
9. 所有关键入口都有配置和 README 指令。
10. 结果目录、日志、checkpoint 与图表命名统一。
11. Git 历史能分辨算法、测试、实验和文档改动。
12. 代理可基于本指南继续扩展，而不会因目录或接口混乱失控。

如果以上 12 项中有任何一项不能满足，就不要把仓库视为“投稿级完整仓库”。
