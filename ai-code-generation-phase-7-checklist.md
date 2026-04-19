# Phase 7 Checklist

目标：统计分析、图表与复现实验导出

## 本阶段产物

- [ ] 主结果表
- [ ] 消融结果表
- [ ] 鲁棒性结果表
- [ ] 主图、tradeoff 曲线、摘要报告
- [ ] README 复现实验命令

## 代理执行要求

- [ ] evaluation 只消费标准结果文件
- [ ] reporting 只消费 evaluation 输出
- [ ] 图表和表格命名规则固定
- [ ] 不要在 reporting 中重写训练逻辑

## 必做任务

- [ ] 实现主结果汇总
- [ ] 实现 ambiguity/FPR/FNR/robustness 指标汇总
- [ ] 实现主图和主表导出
- [ ] 实现 summary report
- [ ] 更新 README 的复现实验入口

## 验证

- [ ] 运行 report 构建命令
- [ ] 检查 `outputs/figures/`、`outputs/tables/`、`outputs/summaries/`
- [ ] 手动确认至少一张主图和一张主表成功导出

## 退出条件

- [ ] 主图主表可一键导出
- [ ] 仓库达到投稿级输出要求
