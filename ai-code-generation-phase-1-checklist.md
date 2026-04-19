# Phase 1 Checklist

目标：数据层与 non-IID 划分

## 本阶段产物

- [ ] 至少一个视觉数据集加载器
- [ ] 至少一种 non-IID 划分方法
- [ ] split metadata 保存与读取逻辑
- [ ] 数据层测试

## 代理执行要求

- [ ] 不要把划分逻辑写进训练脚本
- [ ] 保存随机种子与 split metadata
- [ ] split 逻辑必须可复用
- [ ] 先做 CIFAR-10 或同等轻量数据集

## 必做任务

- [ ] 实现 `load_dataset()`
- [ ] 实现 `build_partition()`
- [ ] 支持 Dirichlet 或 shard split
- [ ] 保存 split 元信息到标准路径
- [ ] 增加分区样本守恒和复现性测试

## 验证

- [ ] 运行 `pytest tests/unit/test_splits.py -q`
- [ ] 同一 seed 下重复生成 split，结果一致
- [ ] 样本总数与原数据集一致

## 退出条件

- [ ] 至少一个数据集可用
- [ ] 至少一种 non-IID 划分稳定可复现
- [ ] split 输出可被训练阶段直接消费
