# Phase 6 Checklist

目标：攻击与鲁棒性评测

## 本阶段产物

- [ ] fine-tuning attack
- [ ] pruning 或 quantization attack
- [ ] 至少一种 distillation 或 extraction attack
- [ ] 攻击日志与 attacked checkpoint

## 代理执行要求

- [ ] 攻击逻辑独立于主训练入口
- [ ] 每种攻击保存自身配置和输出
- [ ] 不要把攻击副作用写回原始 checkpoint

## 必做任务

- [ ] 实现攻击入口脚本
- [ ] 为每种攻击定义标准输出目录
- [ ] 保存 attacked checkpoint 和日志
- [ ] 增加输出存在性与最小回归测试

## 验证

- [ ] 运行攻击套件命令
- [ ] 检查 attacked checkpoint 输出
- [ ] 检查攻击后仍可继续跑 verification

## 退出条件

- [ ] 至少一种攻击完整可运行
- [ ] 攻击输出可被 evaluation 消费
