# Phase 3 Checklist

目标：watermark/fingerprint 基线

## 本阶段产物

- [ ] owner codebook 或最小 watermark 基线
- [ ] 正证据查询集
- [ ] 可选负证据查询集
- [ ] 最小 verification 通路

## 代理执行要求

- [ ] 不要把 verification 判决逻辑写进 query 生成模块
- [ ] 所有 owner artifacts 必须可持久化保存
- [ ] 先做最小 baseline，再扩展多比特设计

## 必做任务

- [ ] 实现 `generate_codebook()` 或最小 trigger 生成逻辑
- [ ] 实现 `build_positive_queries()`
- [ ] 实现 owner metadata 保存
- [ ] 将工件接入最小训练或验证流程
- [ ] 增加码长、集合大小、唯一性测试

## 验证

- [ ] 运行 watermark baseline 命令
- [ ] 检查 owner artifacts 输出
- [ ] 检查最小 verification 结果是否生成

## 退出条件

- [ ] 至少一个 owner 的工件可生成并复用
- [ ] 最小版权验证链路可运行
