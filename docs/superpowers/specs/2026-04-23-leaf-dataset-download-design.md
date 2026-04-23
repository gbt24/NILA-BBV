# LEAF Dataset Download Design

## Goal

新增独立脚本，自动下载并预处理 `FEMNIST` 与 `Sent140`，把数据写入当前项目已经支持的本地 LEAF 风格目录结构，使以下命令在本地数据存在时可直接运行：

- `uv run python scripts/train/run_watermark_baseline.py dataset=femnist seed=0`
- `uv run python scripts/train/run_watermark_baseline.py dataset=sent140 seed=0`

## Scope

本设计只覆盖：

1. 官方或原始来源下载
2. 本地解压与预处理
3. 转换为项目现有 loader 读取的 LEAF 风格 shard json
4. 一个统一入口脚本

不覆盖：

1. 修改现有训练 loader 为自动下载
2. 修改 federated / verification / reporting 主逻辑
3. 新增第三方数据平台依赖

## Constraints

1. 默认输出目录固定为仓库内 `data/raw/`
2. `src/bbv/datasets/leaf.py`、`leaf_femnist.py`、`text.py` 继续只负责读本地数据
3. 脚本必须可重复执行，已完成数据默认跳过
4. 不把下载缓存、压缩包或中间文件提交到 git
5. 尽量使用 Python 标准库完成下载、解压和转换

## Recommended Approach

推荐做法是三个脚本：

1. `scripts/data/prepare_femnist.py`
2. `scripts/data/prepare_sent140.py`
3. `scripts/data/prepare_leaf_datasets.py`

原因：

1. `FEMNIST` 与 `Sent140` 的原始来源、预处理步骤、数据结构完全不同，拆开更清晰
2. 统一入口保留给用户和 CI，单数据集脚本保留给调试和局部重跑
3. 不污染现有核心代码路径，失败时也更容易定位

## Data Layout

脚本最终必须生成：

```text
data/raw/
  femnist/
    train/
      all_data_0.json
      ...
    test/
      all_data_0.json
      ...
  sent140/
    train/
      all_data_0.json
      ...
    test/
      all_data_0.json
      ...
```

每个 shard 文件都必须满足当前代码已经实现的 LEAF 风格：

```json
{
  "users": ["user0", "user1"],
  "num_samples": [2, 2],
  "user_data": {
    "user0": {"x": [...], "y": [...]},
    "user1": {"x": [...], "y": [...]}
  }
}
```

## FEMNIST Design

### Input Source

优先使用官方或原始可公开获取的数据源，并在脚本中显式记录下载 URL。脚本需要把下载、解压、转换流程全部自动化，不要求用户手工准备中间文件。

### Transformation

1. 以 writer 为 client 边界
2. 保留 natural non-IID 划分
3. 每条样本写成长度为 `784` 的扁平灰度向量
4. 标签写成整数类别 id
5. `train/` 与 `test/` 分开写 shard 文件

### Output Compatibility

输出必须能被 `src/bbv/datasets/leaf_femnist.py` 直接读取，不再需要额外转换。

## Sent140 Design

### Input Source

优先使用官方 Sentiment140 原始 csv 或同等官方来源数据。

### Transformation

1. 按用户 id 分组形成 client
2. 文本样本保留为当前项目可兼容格式：列表或字符串，且文本内容可由现有 `text.py` 的 `_extract_text()` 提取
3. 标签保留原始情感标签值，由现有 `_normalize_sent140_label()` 统一转成二分类
4. `train/` 与 `test/` 分别写 shard 文件

### Output Compatibility

输出必须能被 `src/bbv/datasets/text.py` 中的 `load_sent140_split()` 直接读取。

## Script Interfaces

### `scripts/data/prepare_femnist.py`

支持：

- `--output-root data/raw`
- `--force`
- `--shard-size <int>`

行为：

1. 若目标目录已完整存在且未指定 `--force`，直接退出并提示复用已有数据
2. 否则执行下载、解压、转换、写 shard

### `scripts/data/prepare_sent140.py`

支持：

- `--output-root data/raw`
- `--force`
- `--shard-size <int>`

### `scripts/data/prepare_leaf_datasets.py`

支持：

- `--dataset femnist|sent140|all`
- `--output-root data/raw`
- `--force`

行为：

1. 分发到对应子脚本
2. 汇总打印最终生成目录

## Error Handling

脚本必须对以下情况报清晰错误：

1. 下载失败
2. 解压失败
3. 原始文件格式与预期不匹配
4. 生成后 `train/` 或 `test/` 缺失
5. 生成后 shard 内部不满足 `users / num_samples / user_data` 结构

## Verification Plan

实现后必须按顺序验证：

1. `uv run python scripts/data/prepare_femnist.py`
2. `uv run python scripts/data/prepare_sent140.py`
3. `uv run python scripts/train/run_watermark_baseline.py dataset=femnist seed=0`
4. `uv run python scripts/train/run_watermark_baseline.py dataset=sent140 seed=0`
5. 最后重跑完整 final verification gate

## Files To Modify

预计新增：

- `scripts/data/prepare_femnist.py`
- `scripts/data/prepare_sent140.py`
- `scripts/data/prepare_leaf_datasets.py`
- `tests/smoke/test_prepare_leaf_datasets.py`

预计小幅修改：

- `README.md`
- 必要时补充 `tests/integration` 中针对真实数据存在条件的 smoke/integration coverage

## Acceptance Criteria

满足以下条件才算完成：

1. 两个数据集都能一键下载和预处理到 `data/raw/`
2. 生成数据能被现有 loader 直接读取
3. `dataset=femnist` 与 `dataset=sent140` 训练命令能在本地跑通
4. 最终验证 gate 中原先因缺数据失败的两条命令通过
