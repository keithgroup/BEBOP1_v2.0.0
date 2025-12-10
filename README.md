# BEBOP1

The bond energy/bond order population ([BEBOP](https://doi.org/10.1021/acs.jctc.2c00334)) program is a computational chemistry algorithm that computes accurate molecular energies at equilibrium and bond energies using well-conditioned Hartree-Fock orbital populations and bond orders from approximate quantum chemistry methods.

## Installation

```bash
git clone https://github.com/keithgroup/BEBOP1_v2.0.0
cd BEBOP1_v2.0.0
pip install .
```

## Preparing BEBOP1 Input Files

BEBOP-1 requires output from the MinPop algorithm when running Hartree-Fock. Below is an example of how to prepare input files using Gaussian 16.

1. Optimize your molecular structure using your preferred level of theory (e.g., B3LYP/CBSB7, with or without dispersion).

```
# Opt B3LYP/CBSB7
```

2. Run Restricted Openshell Hartree-Fock on the optimized structure in Gaussian, using a well-conditioned basis set: CBSB3. 

Note: The code processes alpha and beta electron populations so RHF wavefunction outputs will not work. UHF wavefuction outputs should work with this code, but ROHF is currently preferred to eliminate spin contamination and to be consistent with our calculations.   

```
# SP ROHF/CBSB3 Pop=(Full) IOp(6/27=122,6/12=3)
```
Note: 'Pop=(Full)' runs the full population analysis needed by the code.  IOp(6/27=122) calls the MinPop procedure needed to obtain well-conditioned minimum basis set orbital populations.  IOp(6/12=3) ensures printing of orbitals for systems with more than 50 atoms.  


## Usage

Execute `bebop1` in the command line:

```bash
bebop1 -f {name_file} --be --sort --json > {name_file}.bop
```

where `{name_file}` is the Gaussian output file that contains the HF MinPop calculation.  

Note: the code should be able to parse through a multistep job where multiple single-point energy, geometry optimization, and/or frequency calculations are run, as long as one MinPop calculation output block exists in the output file.  

### Parser Options

```bash
$ bebop1 -h
usage: bebop1 [-h] -f F [--be] [--sort] [--json]

compute BEBOP atomization energies and bond energies (i.e., gross and net)

optional arguments:
  -h, --help  show this help message and exit
  -f F        name of the Gaussian Hartree-Fock output file
  --be        compute BEBOP bond energies (net and gross bond energies)
  --sort      sort the net BEBOP bond energies (from lowest to highest in energy)
  --json      save the job output into JSON
```

## Citation

If this code is found useful, please cite:

**BEBOP-1**: Barbaro Zulueta, Sonia V. Tulyani, Phillip R. Westmoreland, Michael J. Frisch, E. James Petersson, George A. Petersson, and John A. Keith
Journal of Chemical Theory and Computation **2022** _18_ (8), 4774-4794
DOI: [10.1021/acs.jctc.2c00334](https://doi.org/10.1021/acs.jctc.2c00334)

## License

Distributed under the MIT License.
See `LICENSE` for more information.


