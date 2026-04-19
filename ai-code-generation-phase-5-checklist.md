# Phase 5 Checklist

目标：black-box verification 与校准

## 本阶段产物

- [ ] 黑盒查询执行逻辑
- [ ] 码字恢复或最小判决逻辑
- [ ] owner score
- [ ] threshold calibration
- [ ] margin 或 ambiguity 相关指标

## 代理执行要求

- [ ] verification 不依赖训练期临时变量
- [ ] calibration 不能写死在脚本里
- [ ] 正证据和负证据要统一进入分数定义

## 必做任务

- [ ] 实现 `query_model()`
- [ ] 实现 `recover_codeword()` 或等价恢复逻辑
- [ ] 实现 `compute_owner_score()`
- [ ] 实现 `calibrate_threshold()`
- [ ] 实现 `verify_owner()`

## 验证

- [ ] 运行 verification 命令
- [ ] 检查 owner/non-owner 结果可区分
- [ ] 检查 calibration artifact 是否输出

## 退出条件

- [ ] verification 结果稳定可导出
- [ ] 后续攻击与评测阶段可以直接消费这些输出
