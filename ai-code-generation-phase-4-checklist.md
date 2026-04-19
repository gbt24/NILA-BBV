# Phase 4 Checklist

目标：adaptive allocation

## 本阶段产物

- [ ] 客户端适配度估计逻辑
- [ ] watermark budget 分配策略
- [ ] 联邦训练中的 allocation 钩子
- [ ] 消融开关

## 代理执行要求

- [ ] allocation 模块必须可单独启停
- [ ] 不要直接读取不必要的私有统计量
- [ ] 分配逻辑不要散落在训练循环多个位置

## 必做任务

- [ ] 实现 `estimate_adaptability()`
- [ ] 实现 `allocate_watermark_budget()`
- [ ] 将 assignment 接入训练流程
- [ ] 保存 assignment 日志与配置
- [ ] 增加预算守恒和分数范围测试

## 验证

- [ ] 运行 allocation integration test
- [ ] 检查 assignment 输出是否存在
- [ ] 检查关闭该模块后基线流程仍可运行

## 退出条件

- [ ] allocation 可启停
- [ ] allocation 输出可进入后续 verification 和 evaluation
