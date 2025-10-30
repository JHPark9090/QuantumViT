# Quantum Vision Transformer

Implementation of Quantum Vision Transformers based on:

**Paper**: Cherrat et al. (2024) - "Quantum Vision Transformers"
**arXiv**: https://arxiv.org/abs/2209.08167

## Overview

This implementation provides three types of quantum vision transformers for image classification:

1. **Orthogonal Patch-wise Network** - Processes each patch independently with quantum circuits (trivial attention)
2. **Quantum Orthogonal Transformer** - Classical-like attention mechanism with quantum Q/K/V projections
3. **Compound Transformer** - Novel quantum attention using compound matrices (polynomial speedup)

### Key Features

- RBS (Reconfigurable Beam Splitter) gates for quantum operations
- Matrix data loader with amplitude encoding
- Three quantum orthogonal circuit types:
  - **Butterfly**: O(log d) depth, most efficient
  - **Pyramid**: Alternative hierarchical structure
  - **X**: Cross-pattern connections
- Hamming-weight preserving operations
- PennyLane + PyTorch integration
- Full training and evaluation pipeline

---

## Installation

### Requirements

```bash
# Core dependencies
pip install pennylane torch torchvision numpy matplotlib seaborn scikit-learn tqdm

# Or use existing conda environment
conda activate ./conda-envs/qml_eeg
```

### File Structure

```
.
├── QuantumVisionTransformer.py          # Model implementation
├── test_quantum_vision_transformer.py   # Training/testing script
└── QUANTUM_VISION_TRANSFORMER_README.md # This file
```

---

## Quick Start

### 1. Test Installation

```bash
# Quick test (should print output shape)
python QuantumVisionTransformer.py
```

### 2. Train on MNIST (Patchwise Model)

```bash
python test_quantum_vision_transformer.py \
    --dataset mnist \
    --model patchwise \
    --n-layers 2 \
    --n-epochs 50 \
    --batch-size 32
```

### 3. Train on CIFAR-10 (Compound Model)

```bash
python test_quantum_vision_transformer.py \
    --dataset cifar10 \
    --model compound \
    --n-layers 2 \
    --circuit-type butterfly \
    --n-epochs 50
```

### 4. Quick Test with Mini Dataset

```bash
python test_quantum_vision_transformer.py \
    --dataset mnist \
    --model patchwise \
    --mini \
    --n-epochs 5 \
    --batch-size 16
```

---

## Model Architectures

### 1. Orthogonal Patch-wise Network

**Architecture**:
```
Input Image (28×28)
    ↓
Extract Patches (16 patches of 7×7)
    ↓
For each patch:
    ├─ Amplitude Encoding → Quantum State
    ├─ Quantum Orthogonal Layers (RBS gates)
    └─ Measurement (PauliZ expectation values)
    ↓
Concatenate features
    ↓
Fully Connected → Classification
```

**Characteristics**:
- No attention between patches (trivial attention)
- Independent processing of each patch
- Most efficient for quantum hardware (smallest circuit)
- Best for: Quick testing, hardware experiments

**Usage**:
```bash
python test_quantum_vision_transformer.py --model patchwise --n-layers 2
```

### 2. Quantum Orthogonal Transformer

**Architecture**:
```
Input Image (28×28)
    ↓
Extract Patches (16 patches of 7×7)
    ↓
For each patch:
    ├─ Quantum Circuit → Q (Query)
    ├─ Quantum Circuit → K (Key)
    └─ Quantum Circuit → V (Value)
    ↓
Scaled Dot-Product Attention:
    Attention(Q, K, V) = softmax(Q·K^T / √d) · V
    ↓
Fully Connected → Classification
```

**Characteristics**:
- Classical-like attention with quantum projections
- Separate quantum circuits for Q, K, V
- More expressive than patchwise
- Best for: Standard vision tasks

**Usage**:
```bash
python test_quantum_vision_transformer.py --model orthogonal --n-layers 2
```

### 3. Compound Transformer

**Architecture**:
```
Input Image (28×28)
    ↓
Extract Patches (16 patches of 7×7)
    ↓
For each patch pair:
    ├─ Compute Compound Matrix (2nd order)
    ├─ Amplitude Encoding → Quantum State
    ├─ Quantum Orthogonal Layers
    └─ Multi-observable Measurement (PauliX, Y, Z)
    ↓
Aggregate features
    ↓
Fully Connected → Classification
```

**Characteristics**:
- Novel quantum attention mechanism
- Polynomial speedup over classical transformers
- Uses compound matrices for pairwise relationships
- Measures multiple observables for richer features
- Best for: Research, theoretical advantage

**Usage**:
```bash
python test_quantum_vision_transformer.py --model compound --n-layers 2
```

---

## Quantum Circuit Types

All models support three types of quantum orthogonal circuits:

### Butterfly Circuit (Recommended)

- **Depth**: O(log d) - Most efficient
- **Structure**: Hierarchical pairing of qubits
- **Best for**: Hardware experiments, scalability

```bash
--circuit-type butterfly
```

### Pyramid Circuit

- **Depth**: O(d) - Linear depth
- **Structure**: Sequential connections
- **Best for**: Exploratory experiments

```bash
--circuit-type pyramid
```

### X Circuit

- **Depth**: O(d/2) - Medium depth
- **Structure**: Cross-pattern connections
- **Best for**: Alternative architectures

```bash
--circuit-type x
```

---

## Command Line Arguments

### Dataset Options

```bash
--dataset [mnist|cifar10]    # Dataset to use (default: mnist)
--mini                       # Use mini dataset (500 train, 100 val, 200 test)
```

### Model Options

```bash
--model [patchwise|orthogonal|compound]  # Model type (default: patchwise)
--circuit-type [butterfly|pyramid|x]     # Quantum circuit (default: butterfly)
--n-layers INT                           # Number of quantum layers (default: 2)
--patch-size INT                         # Patch size (default: 7)
```

### Training Options

```bash
--n-epochs INT        # Number of epochs (default: 50)
--batch-size INT      # Batch size (default: 32)
--lr FLOAT            # Learning rate (default: 1e-3)
--weight-decay FLOAT  # Weight decay (default: 1e-4)
```

### System Options

```bash
--seed INT            # Random seed (default: 2025)
--cpu                 # Force CPU usage
--output-dir PATH     # Output directory (default: ./qvit_results)
```

---

## Example Experiments

### Experiment 1: Compare All Three Models on MNIST

```bash
# Patchwise
python test_quantum_vision_transformer.py \
    --dataset mnist --model patchwise --n-layers 2 --n-epochs 30

# Orthogonal
python test_quantum_vision_transformer.py \
    --dataset mnist --model orthogonal --n-layers 2 --n-epochs 30

# Compound
python test_quantum_vision_transformer.py \
    --dataset mnist --model compound --n-layers 2 --n-epochs 30
```

### Experiment 2: Compare Circuit Types (Butterfly vs Pyramid vs X)

```bash
# Butterfly (most efficient)
python test_quantum_vision_transformer.py \
    --dataset mnist --model patchwise --circuit-type butterfly --n-layers 2

# Pyramid
python test_quantum_vision_transformer.py \
    --dataset mnist --model patchwise --circuit-type pyramid --n-layers 2

# X-pattern
python test_quantum_vision_transformer.py \
    --dataset mnist --model patchwise --circuit-type x --n-layers 2
```

### Experiment 3: Depth Study (1, 2, 3, 4 layers)

```bash
for layers in 1 2 3 4; do
    python test_quantum_vision_transformer.py \
        --dataset mnist --model patchwise --n-layers $layers --n-epochs 30
done
```

### Experiment 4: CIFAR-10 Benchmark

```bash
python test_quantum_vision_transformer.py \
    --dataset cifar10 \
    --model compound \
    --circuit-type butterfly \
    --n-layers 3 \
    --n-epochs 100 \
    --batch-size 64 \
    --lr 1e-3
```

---

## Output Files

After training, the following files are saved to `--output-dir`:

1. **`best_model_{dataset}_{model}.pt`**
   - PyTorch checkpoint with best validation accuracy
   - Contains: model state, optimizer state, hyperparameters

2. **`training_curves_{dataset}_{model}.png`**
   - Loss and accuracy curves over epochs
   - Shows train vs validation performance

3. **`confusion_matrix_{dataset}_{model}.png`**
   - Confusion matrix heatmap for test set
   - Shows per-class performance

4. **`results_{dataset}_{model}.txt`**
   - Summary of experiment results
   - Includes: accuracy, loss, training time, hyperparameters

---

## Performance Benchmarks

### MNIST (28×28, 10 classes)

| Model         | Circuit   | Layers | Test Acc | Notes                     |
|---------------|-----------|--------|----------|---------------------------|
| Patchwise     | Butterfly | 2      | ~95%*    | Fastest, hardware-friendly|
| Orthogonal    | Butterfly | 2      | ~96%*    | Better than patchwise     |
| Compound      | Butterfly | 2      | ~97%*    | Best performance          |

*Expected results based on paper (actual may vary)

### CIFAR-10 (28×28 resized, 10 classes)

| Model         | Circuit   | Layers | Test Acc | Notes                     |
|---------------|-----------|--------|----------|---------------------------|
| Patchwise     | Butterfly | 3      | ~55%*    | Baseline                  |
| Orthogonal    | Butterfly | 3      | ~58%*    | Attention helps           |
| Compound      | Butterfly | 3      | ~60%*    | Best quantum performance  |

*Expected results (CIFAR-10 is more challenging)

---

## Technical Details

### RBS Gate Implementation

The Reconfigurable Beam Splitter (RBS) gate is implemented as:

```python
def RBS_gate(theta, wire1, wire2):
    qml.RY(2 * theta, wires=wire1)
    qml.CNOT(wires=[wire1, wire2])
    qml.RY(-2 * theta, wires=wire1)
    qml.CNOT(wires=[wire1, wire2])
```

This achieves the RBS unitary:
```
RBS(θ) = |cos(θ)  -sin(θ)|
         |sin(θ)   cos(θ)|
```

### Data Encoding

Matrix data is encoded using amplitude encoding:

1. **Flatten**: Matrix X ∈ ℝ^(n×d) → vector ∈ ℝ^(nd)
2. **Normalize**: Σᵢ|xᵢ|² = 1
3. **Pad**: Extend to 2^n qubits
4. **Encode**: |ψ⟩ = Σᵢ xᵢ|i⟩

This requires log₂(nd) qubits.

### Quantum Measurements

Different models measure different observables:

- **Patchwise**: PauliZ on all qubits
- **Orthogonal**: PauliZ on all qubits (Q, K, V)
- **Compound**: PauliX, PauliY, PauliZ (multi-observable)

---

## Troubleshooting

### Issue: Out of Memory

**Solution 1**: Reduce batch size
```bash
--batch-size 16  # or even 8
```

**Solution 2**: Use CPU
```bash
--cpu
```

**Solution 3**: Reduce number of layers
```bash
--n-layers 1
```

### Issue: Training Too Slow

**Solution 1**: Use mini dataset for testing
```bash
--mini --n-epochs 5
```

**Solution 2**: Use patchwise model (fastest)
```bash
--model patchwise
```

**Solution 3**: Use butterfly circuit (most efficient)
```bash
--circuit-type butterfly
```

### Issue: Low Accuracy

**Possible causes**:
1. Learning rate too high/low → Try `--lr 1e-4` or `--lr 1e-2`
2. Too few epochs → Increase `--n-epochs 100`
3. Need more quantum layers → Try `--n-layers 3`
4. Dataset too challenging → Test on MNIST first

---

## Comparison with Classical Vision Transformer

| Feature                  | Quantum ViT (This Work)  | Classical ViT         |
|--------------------------|--------------------------|----------------------|
| Attention mechanism      | Quantum/Compound         | Multi-head attention |
| Complexity (attention)   | O(poly(log d))          | O(d²)                |
| Circuit depth            | O(log d) (butterfly)    | N/A                  |
| Hardware requirements    | Quantum computer        | Classical computer   |
| Scalability             | Limited by qubits       | High                 |
| Theoretical speedup     | Polynomial (compound)   | -                    |
| Current performance     | ~60% (CIFAR-10)         | ~90% (CIFAR-10)      |

**Key takeaway**: Quantum ViT offers theoretical speedup but current NISQ hardware limits practical performance.

---

## Citations

If you use this implementation, please cite:

```bibtex
@article{cherrat2024quantum,
  title={Quantum Vision Transformers},
  author={Cherrat, El Amine and others},
  journal={arXiv preprint arXiv:2209.08167},
  year={2024}
}
```

---

## Future Work

1. **Hardware Experiments**: Test on IBM Quantum, IonQ, or Rigetti devices
2. **Larger Models**: Scale to more qubits (12-16) for better performance
3. **Hybrid Models**: Combine with classical transformers
4. **Medical Imaging**: Apply to MedMNIST datasets (as in paper)
5. **Error Mitigation**: Add ZNE, measurement error mitigation
6. **Quantum Advantage**: Benchmark compound transformer speedup

---

## License

This implementation is provided for research and educational purposes.

---

## Contact

For questions or issues, please refer to the original paper or contact the repository maintainer.

---

**Last Updated**: October 30, 2025
**Version**: 1.0
**Based on**: Cherrat et al. (2024) - Quantum Vision Transformers
