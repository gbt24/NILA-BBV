# AI代码生成一页式执行清单

适用对象：`OpenCode`、`Claude Code` 类代码代理  
目标：生成一个面向“非IID低歧义自适应黑盒版权验证”研究的投稿级完整仓库

## 1. 开始前

- [ ] 先读这 3 份文件：`deep-research-report-2.md`、`ai-code-generation-guide.md`、当前仓库已有代码
- [ ] 复述本轮目标，不要同时做多个大任务
- [ ] 列出本轮将修改/新增的文件
- [ ] 说明验证命令，再开始改代码

## 2. 总体架构

- [ ] 只用 `PyTorch` 作为底层框架
- [ ] 主研究实验以 `FedLab` 风格组织
- [ ] `Flower` 只做后续扩展接口，不重写主流程
- [ ] 所有实验都通过统一配置驱动
- [ ] 所有结果都写入标准输出目录

## 3. 仓库边界

- [ ] `configs/` 只放配置
- [ ] `src/` 只放核心实现
- [ ] `scripts/` 只放薄入口
- [ ] `tests/` 只放测试
- [ ] `outputs/` 只放实验产物
- [ ] `docs/` 只放文档与计划
- [ ] 不要把算法写进 `scripts/`
- [ ] 不要把评测、绘图、训练逻辑混在一个文件

## 4. 核心模块

- [ ] `federated`：联邦训练闭环
- [ ] `datasets`：数据加载与 non-IID 划分
- [ ] `models`：视觉/文本模型注册
- [ ] `watermarking`：codebook、正负证据集、query 工件
- [ ] `allocation`：客户端适配度与分配策略
- [ ] `verification`：黑盒查询、score、margin、threshold
- [ ] `attacks`：fine-tuning、pruning、quantization、distillation
- [ ] `evaluation`：主结果、消融、鲁棒性指标
- [ ] `reporting`：图表、表格、摘要报告

## 5. 实现顺序

- [ ] Phase 0：仓库初始化与 smoke test
- [ ] Phase 1：数据层与 non-IID 划分
- [ ] Phase 2：主任务训练与联邦基线
- [ ] Phase 3：watermark/fingerprint 基线
- [ ] Phase 4：adaptive allocation
- [ ] Phase 5：black-box verification 与校准
- [ ] Phase 6：攻击与鲁棒性评测
- [ ] Phase 7：统计分析、图表、复现实验导出
- [ ] 上一阶段未通过，不进入下一阶段

## 6. 代理硬规则

- [ ] 先读相关文件，再改
- [ ] 每次只做一个聚焦模块
- [ ] 优先最小正确实现，不提前抽象
- [ ] 先接口和测试，再补实现
- [ ] 每次改动后必须运行最小验证命令
- [ ] 研究设定不清楚时先提问
- [ ] 不要新建与现有模块职责重复的第二套实现

## 7. 测试与复现

- [ ] 新模块至少有一个 unit test
- [ ] 新流程至少有一个 smoke test
- [ ] 核心链路至少有一个 integration test
- [ ] 保存随机种子、split metadata、模型配置、攻击配置
- [ ] 每次运行必须有唯一 run id
- [ ] checkpoint、metrics、figures、tables 分目录保存

## 8. Git规范

- [ ] 一个提交只做一类事
- [ ] 算法改动和实验结果更新不要混提
- [ ] 大规模格式化不要和逻辑改动混提
- [ ] commit message 清晰表达模块和目的
- [ ] 提交前先跑最小验证命令

## 9. 常用提示词骨架

```md
请阅读与 <module-name> 相关的现有文件，只做最小正确改动，实现 <target>。
要求：
1. 先列出将修改/新增的文件。
2. 保持现有目录结构，不新增不必要抽象。
3. 先实现接口和最小测试，再补完整实现。
4. 完成后运行指定验证命令并报告结果。
```

## 10. 失败时怎么做

- [ ] 测试失败：先缩小到最小失败用例
- [ ] 架构漂移：先停止新增代码，整理重复职责
- [ ] 重复实现：优先合并回原模块，不长期并存
- [ ] 实验跑不通：先区分配置、数据、代码三类问题
- [ ] 结果不稳：先查 seed、split、checkpoint、evaluation schema
- [ ] 连续两次修复失败：停下来重新分析根因

## 11. 最终验收

- [ ] 新环境可安装
- [ ] 最小联邦训练可运行
- [ ] watermark/fingerprint 基线可运行
- [ ] adaptive allocation 可启停
- [ ] verification 可输出 score / threshold / margin
- [ ] 至少一种攻击可运行
- [ ] evaluation 可导出主结果表
- [ ] reporting 可导出主图主表
- [ ] README 有复现实验命令
- [ ] Git 历史能区分算法、测试、实验、文档改动

如果以上有任一项未满足，就不要把仓库视为“投稿级完整仓库”。
