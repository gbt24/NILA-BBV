# Phase 2 Checklist

目标：主任务训练与联邦基线

## 本阶段产物

- [ ] 模型注册机制
- [ ] 联邦训练主循环
- [ ] 基线方法，如 `FedAvg`
- [ ] 训练日志与 checkpoint 输出

## 代理执行要求

- [ ] 不要把 watermark/fingerprint 逻辑提前混入训练主循环
- [ ] 先保证最小联邦训练稳定，再考虑扩展
- [ ] 使用统一配置与输出目录

## 必做任务

- [ ] 实现 `build_model()`
- [ ] 实现 `build_client()` / `build_server()` 或等价接口
- [ ] 实现 `train_federated()`
- [ ] 保存训练指标、checkpoint、run metadata
- [ ] 增加 smoke test 和关键 integration test

## 验证

- [ ] 运行最小基线命令，如 `python scripts/train/run_fedavg.py dataset=cifar10 federated=fedavg model=resnet18 seed=0`
- [ ] 检查 checkpoint 与指标文件生成
- [ ] 确认日志中包含轮次指标

## 退出条件

- [ ] 一个联邦基线在一个数据集上可稳定运行
- [ ] 训练结果可供 watermark/fingerprint 阶段复用
