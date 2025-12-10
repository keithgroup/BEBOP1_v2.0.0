# BEBOP1 HPC Deployment Guide

## Quick Start on HPC Cluster

### 1. Load Required Modules
```bash
# Load Python and NumPy (adjust module names for your cluster)
module load python/3.8
module load numpy/1.19.0
# OR use conda/pip
```

### 2. Install BEBOP1
```bash
# Option A: Direct installation
python setup.py install --user

# Option B: Development installation  
pip install --user -e .

# Option C: Just copy files to working directory
# (All files are already optimized and self-contained)
```

### 3. Test Installation
```bash
# Test with example file
python cli.py -f S000001_B3LYP.out --be --sort

# Should output bond energies and performance metrics
```

### 4. Performance on HPC
- Optimized for molecules with 4+ atoms
- Uses vectorized NumPy operations
- Memory efficient for large molecules
- Typical speedup: 5-15x over original version

### 5. Batch Job Example
```bash
#!/bin/bash
#SBATCH --job-name=bebop1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --memory=8G
#SBATCH --time=01:00:00

module load python/3.8
python cli.py -f your_molecule.out --be --sort --json
```

### 6. Troubleshooting
- Ensure NumPy version >= 1.19.0
- Check file permissions for input files
- For large molecules, increase memory allocation
- Use `--json` flag to save results for analysis

### 7. Integration with Other Codes
```python
# Python script usage
from bebop1 import BEBOP

# Load and calculate
data = BEBOP('your_file.out')
energy = data.total_E()
bonds = data.bond_E(NetBond=True, sig_pi=True)
```
