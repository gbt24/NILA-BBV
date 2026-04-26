# Design: MNIST Dataset Support

**Date:** 2026-04-26  
**Status:** Approved  
**Scope:** Add MNIST (10-class digit recognition) as a supported dataset in the BBV federated learning pipeline.

## Motivation

BBV currently supports CIFAR-10, CIFAR-100, FEMNIST (62-class EMNIST via LEAF), Sent140, and Shakespeare. Adding MNIST provides a lightweight, well-studied baseline for quick experiments.

## Design

### Key Constraint: Input Shape Compatibility

MNIST images are **1-channel 28x28** grayscale. Existing vision models (ResNet18) expect **3-channel 32x32**. Following the FEMNIST pattern (`leaf_femnist.py:12-18`), MNIST images are transformed: `Pad(2) -> repeat(3,1,1)` yielding `(3, 32, 32)`.

### Changes (4 files)

#### 1. `src/bbv/datasets/transforms.py`

- `build_image_transform(train, dataset_name)` — accept optional `dataset_name` parameter.
- For `mnist`: `ToTensor → Pad(2) → Lambda(repeat 3 channels)`. No `RandomHorizontalFlip` (meaningless for digits).
- For CIFAR datasets: behavior unchanged.

```python
def build_image_transform(train: bool, dataset_name: str = "cifar10"):
    steps = [transforms.ToTensor()]
    if dataset_name == "mnist":
        steps.append(transforms.Pad(2))
        steps.append(transforms.Lambda(lambda x: x.repeat(3, 1, 1)))
    elif train:
        steps.insert(0, transforms.RandomHorizontalFlip(p=0.5))
    return transforms.Compose(steps)
```

#### 2. `src/bbv/datasets/loaders.py`

- `_VISION_DATASETS`: add `"mnist": "MNIST"`.
- Pass `dataset_name` to `build_image_transform`.

#### 3. `src/bbv/federated/fedavg.py`

- `build_model_input_shape()`: add `"mnist"` returning `(3, 32, 32)` (post-transform shape).

#### 4. `configs/train/dataset/mnist.yaml` (new)

```yaml
name: mnist
num_classes: 10
samples_per_client: 24
partition_type: dirichlet
concentration: 1.0
shards_per_client: 2
quantity_sigma: 0.0
```

## Verification

- `torchvision.datasets.MNIST` auto-downloads on first use; no data preparation script needed.
- MLP model (input `(3, 32, 32)`) and ResNet18 both work after the channel transform.
- Existing tests (unit/integration/smoke) remain passing.
