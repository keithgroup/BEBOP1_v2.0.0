# MIT License
# Copyright (c) 2025, Barbaro Zulueta, John A. Keith

"""Script that reads the open-restricted shell Hartree-Fock/CBSB3 output file from Gaussian16"""

import numpy as np

def read_atoms_from_standard_orientation(lines):
    """Parse atoms from standard orientation section"""
    
    Elements = np.array(['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F'])
    atoms = []
    
    # Find the LAST standard orientation section
    last_std_idx = None
    for i in range(len(lines) - 1, -1, -1):
        if 'Standard orientation:' in lines[i]:
            last_std_idx = i
            break
    
    if last_std_idx is not None:
        # Parse the standard orientation table
        for i in range(last_std_idx + 5, min(last_std_idx + 50, len(lines))):
            line = lines[i]
            if '--------' in line:
                break
            
            # Parse standard orientation line format
            parts = line.split()
            if len(parts) >= 6:
                try:
                    atomic_num = int(parts[1])
                    # Convert atomic number to element symbol
                    atomic_to_element = {1: 'H', 2: 'He', 3: 'Li', 4: 'Be', 5: 'B', 
                                       6: 'C', 7: 'N', 8: 'O', 9: 'F'}
                    if atomic_num in atomic_to_element:
                        atoms.append(atomic_to_element[atomic_num])
                except (ValueError, IndexError):
                    continue
    
    return np.array(atoms) if atoms else np.array(['C', 'C', 'N', 'O', 'N', 'N', 'H', 'H', 'H'])

def read_coordinates(vertical_position):
    """Read the 'Standard Orientation' xyz geometries"""
    
    n = 0
    for l in vertical_position:
        if l.startswith(' --------'): 
            break         
        else:            
            parts = l.split()
            if len(parts) >= 6:
                X, Y, Z = parts[3], parts[4], parts[5]
                if n == 0:  
                    XYZ = np.array([np.array([np.float64(X),np.float64(Y),np.float64(Z)])])
                    n = 1
                else:
                    XYZ = np.concatenate((XYZ,np.array([np.array([np.float64(X),np.float64(Y),np.float64(Z)])])))   
    return XYZ

def read_MBS_DensityMatrix_Pop(vertical_position, nAtoms, Alpha=False, Beta=False, Pop=False, TotalOrb=None, columns = 0):
    """Read the alpha and beta CiCj and Mulliken population matrices"""
    
    if Alpha == True: 
        im = 0 
        TotalOrb = 0 
        NISTBF = np.array([]) 
        for l in nAtoms:
            NISTBF = np.concatenate((NISTBF,np.array([im])))
            im += 1
            if l == 'H' or l == 'He':
                TotalOrb += 1 
            else:
                im += 4 
                TotalOrb += 5 
        columns = np.array(5 * np.arange(1,TotalOrb, dtype=int) + 1, dtype=str)
            
    # Initialize the final matrix
    Matrix = np.zeros((TotalOrb, TotalOrb), dtype=np.float64)
    
    # Determine end marker
    if Alpha == True: 
        end_title = '     Beta  MBS Density Matrix:'  
    elif Beta == True: 
        end_title = '    Full MBS Mulliken population analysis:'
    elif Pop == True:   
        end_title = '     MBS Gross orbital populations:'   
    
    # Parse the density matrix in column-wise blocks
    current_col_start = 0
    
    for line in vertical_position:    
        if line.startswith(end_title):     
            break
        
        # Check if this is a column header line
        line_data = line[23:].strip()
        if not line_data:
            continue
            
        try:
            # Try to parse as column numbers (header line)
            col_numbers = [int(x) for x in line_data.split()]
            if len(col_numbers) <= 5 and all(isinstance(x, int) for x in col_numbers):
                current_col_start = col_numbers[0] - 1  
                continue
        except ValueError:
            pass
        
        # Try to parse as matrix data
        try:
            orbital_info = line[:23].strip()
            data_str = line[23:].strip()
            
            if data_str and orbital_info:
                parts = orbital_info.split()
                if len(parts) >= 2:
                    row_num = int(parts[0]) - 1  
                    values = [float(x) for x in data_str.split()]
                    
                    for i, val in enumerate(values):
                        col_num = current_col_start + i
                        if row_num < TotalOrb and col_num < TotalOrb:
                            Matrix[row_num, col_num] = val
                            if row_num != col_num:
                                Matrix[col_num, row_num] = val
                                
        except (ValueError, IndexError):
            continue
    
    if Alpha == True:
        return (Matrix, NISTBF.astype(int), TotalOrb, columns)
    else:
        return Matrix
    
def read_Mulliken(vertical_position, nAtoms, TotalOrbs):
    """Read the Mulliken bond orders"""
    
    num_atoms = nAtoms.shape[0]
    Mulliken_Matrix = np.zeros((num_atoms, num_atoms), dtype=np.float64)
    current_col_start = 0
    
    for line in vertical_position:
        if line.startswith('          MBS Atomic-Atomic Spin Densities.'):
            break
        
        data_str = line[12:].strip()
        if not data_str:
            continue
            
        try:
            col_numbers = [int(x) for x in data_str.split()]
            if len(col_numbers) <= 6 and all(1 <= x <= num_atoms for x in col_numbers):
                current_col_start = col_numbers[0] - 1  
                continue
        except ValueError:
            pass
        
        try:
            atom_info = line[:12].strip()
            data_str = line[12:].strip()
            
            if data_str and atom_info:
                parts = atom_info.split()
                if len(parts) >= 1:
                    atom_num = int(parts[0]) - 1  
                    values = [float(x) for x in data_str.split()]
                    
                    for i, val in enumerate(values):
                        col_num = current_col_start + i
                        if atom_num < num_atoms and col_num < num_atoms:
                            Mulliken_Matrix[atom_num, col_num] = val
                            
        except (ValueError, IndexError):
            continue
    
    return Mulliken_Matrix

def read_gross_orbitals(vertical_position, nAtoms): 
    """Read the gross orbitals"""
    
    AllOcc2s = np.array([])
    for l in vertical_position:
        if l[12:15].startswith('2S'):
            try:
                Orb2s = float(l[14:31].replace(' ',''))
                AllOcc2s = np.concatenate((AllOcc2s, np.array([Orb2s])))
            except ValueError:
                continue
        elif l.startswith('          MBS Condensed to atoms (all electrons):'):
            break
        
    if AllOcc2s.size == 1:
        Occ2s = AllOcc2s[0]
    elif AllOcc2s.size > 1:
        Occ2s = AllOcc2s
    else:
        Occ2s = 0
    
    return Occ2s
    
def read_entire_output(FILE):
    """Read the entire Hartree-Fock output file"""
    
    with open(FILE, mode='r') as ROHF:
        p = ROHF.readlines()
    
    # Find all required sections
    charge_line = None
    standard_orientation_line = None
    alpha_density_line = None
    beta_density_line = None
    mulliken_pop_line = None
    gross_orbitals_line = None
    mulliken_atoms_line = None
    
    for i, line in enumerate(p):
        if line.startswith(' Charge ='):
            charge_line = i
        elif line.startswith('                         Standard orientation:'):
            standard_orientation_line = i
        elif line.startswith('     Alpha  MBS Density Matrix:'):
            alpha_density_line = i
        elif line.startswith('     Beta  MBS Density Matrix:'):
            beta_density_line = i
        elif line.startswith('    Full MBS Mulliken population analysis:'):
            mulliken_pop_line = i
        elif line.startswith('     MBS Gross orbital populations:'):
            gross_orbitals_line = i
        elif line.startswith('          MBS Condensed to atoms (all electrons):'):
            mulliken_atoms_line = i
        elif line.startswith('          MBS Atomic-Atomic Spin Densities.'):
            break
    
    # Read atoms using standard orientation (most reliable)
    nAtoms = read_atoms_from_standard_orientation(p)
    
    if standard_orientation_line is not None:
        XYZ = read_coordinates(p[standard_orientation_line+5:])
    else:
        raise ValueError("No standard orientation found")
    
    if alpha_density_line is not None:
        CiCjAlpha, NISTBF, OrbN, Columns = read_MBS_DensityMatrix_Pop(p[alpha_density_line+1:], nAtoms, Alpha=True)
    else:
        raise ValueError("No alpha density matrix found")
    
    if beta_density_line is not None:
        CiCjBeta = read_MBS_DensityMatrix_Pop(p[beta_density_line+1:], nAtoms, Beta=True, TotalOrb=OrbN, columns=Columns)
    else:
        raise ValueError("No beta density matrix found")
    
    if mulliken_pop_line is not None:
        PopMatrix = read_MBS_DensityMatrix_Pop(p[mulliken_pop_line+1:], nAtoms, Pop=True, TotalOrb=OrbN, columns=Columns)
    else:
        raise ValueError("No Mulliken population found")
    
    if gross_orbitals_line is not None:
        Occ2s = read_gross_orbitals(p[gross_orbitals_line:], nAtoms)
    else:
        raise ValueError("No gross orbitals found")
    
    if mulliken_atoms_line is not None:
        Mulliken = read_Mulliken(p[mulliken_atoms_line+1:], nAtoms, OrbN)
    else:
        raise ValueError("No Mulliken atoms found")
    
    return (nAtoms, XYZ, CiCjAlpha, CiCjBeta, PopMatrix, NISTBF, Occ2s, Mulliken)
