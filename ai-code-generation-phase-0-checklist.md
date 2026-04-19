# Phase 0 Checklist

目标：仓库初始化与最小闭环

## 本阶段产物

- [ ] 基础目录结构
- [ ] `pyproject.toml` 或等价依赖定义
- [ ] `README.md` 初版
- [ ] 最小训练入口脚本
- [ ] 最小 smoke test

## 代理执行要求

- [ ] 先读取 `ai-code-generation-guide.md`
- [ ] 只搭脚手架，不提前实现 watermark、verification、attacks
- [ ] 所有入口通过统一配置驱动
- [ ] `scripts/` 只放薄入口
- [ ] `src/` 只放核心实现

## 必做任务

- [ ] 创建 `configs/`、`src/`、`scripts/`、`tests/`、`outputs/`、`docs/`
- [ ] 建立最小配置加载链路
- [ ] 建立最小模型构建链路
- [ ] 建立最小单机或伪联邦 smoke 运行脚本
- [ ] 建立最小测试用例

## 验证

- [ ] 运行 `pytest tests/smoke -q`
- [ ] 运行最小训练命令并确认不报错
- [ ] 检查输出目录是否生成基础日志或运行记录

## 退出条件

- [ ] 新环境可安装
- [ ] 最小闭环可运行
- [ ] 后续阶段可在此基础上增量开发
