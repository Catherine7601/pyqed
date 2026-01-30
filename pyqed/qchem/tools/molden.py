# pyqed/qchem/tools/molden.py
"""
Molden tools for PyQED

Features:
- Store molden file as input

Author: Ruoxi
"""

import numpy as np

ANGULAR = ['s', 'p', 'd', 'f', 'g', 'h', 'i']


def order_ao_index(bas):
    cart = (bas[0].coord_type == 'cartesian')
    
    idx = []
    off = 0
    
    if cart:
        for shell in bas:
            l = shell.angmom
            nctr = shell.coeffs.shape[1]
            for n in range(nctr):
                if l == 2:
                    idx.extend([off+0, off+3, off+5, off+1, off+2, off+4])
                elif l == 3:
                    idx.extend([off+0, off+6, off+9, off+3, off+1,
                                off+2, off+5, off+8, off+7, off+4])
                elif l == 4:
                    idx.extend([off+0, off+10, off+14, off+1, off+2,
                                off+6, off+11, off+9, off+13, off+3,
                                off+5, off+12, off+4, off+7, off+8])
                elif l > 4:
                    raise RuntimeError('l > 4 is not supported')
                else:
                    idx.extend(range(off, off + (l+1)*(l+2)//2))
                off += (l+1)*(l+2)//2
    else:
        for shell in bas:
            l = shell.angmom
            nctr = shell.coeffs.shape[1]
            for n in range(nctr):
                if l == 2:
                    idx.extend([off+2, off+3, off+1, off+4, off+0])
                elif l == 3:
                    idx.extend([off+3, off+4, off+2, off+5, off+1, off+6, off+0])
                elif l == 4:
                    idx.extend([off+4, off+5, off+3, off+6, off+2,
                                off+7, off+1, off+8, off+0])
                elif l > 4:
                    raise RuntimeError('l > 4 is not supported')
                else:
                    idx.extend(range(off, off + l*2+1))
                off += l*2 + 1
    
    return idx


def dump_molden(mol, filename, mo_coeff, mo_energy=None, mo_occ=None):
    """MO--->molden"""
    bas = mol._bas
    mo_coeff = mo_coeff.copy()
    nmo = mo_coeff.shape[1]
    
    if mo_energy is None:
        mo_energy = np.arange(nmo, dtype=float)
    if mo_occ is None:
        mo_occ = np.zeros(nmo)
    
    cart = (bas[0].coord_type == 'cartesian')
    
    with open(filename, 'w') as f:
        f.write('[Molden Format]\n')
        f.write('made by PyQED\n')
        
        f.write('[Atoms] (AU)\n')
        for ia in range(mol.natom):
            symb = mol.atom_symbol(ia)
            chg = mol.atom_charge(ia)
            coord = mol.atom_coord(ia)
            f.write(f'{symb}   {ia+1}   {chg}   ')
            f.write(f'{coord[0]:18.14f}   {coord[1]:18.14f}   {coord[2]:18.14f}\n')
        
        f.write('[GTO]\n')
        for ia in range(mol.natom):
            f.write(f'{ia+1} 0\n')
            for shell in bas:
                if shell.icenter != ia:
                    continue
                l = shell.angmom
                exps = shell.exps
                coeffs = shell.coeffs
                nprim = len(exps)
                nctr = coeffs.shape[1]
                for ic in range(nctr):
                    f.write(f' {ANGULAR[l]}   {nprim:2d} 1.00\n')
                    for ip in range(nprim):
                        f.write(f'    {exps[ip]:18.14g}  {coeffs[ip, ic]:18.14g}\n')
            f.write('\n')
        
        if cart:
            f.write('[6d]\n[10f]\n[15g]\n')
        else:
            f.write('[5d]\n[7f]\n[9g]\n')
        f.write('\n')
        
        if cart:
            norm = mol.overlap.diagonal() ** 0.5
            mo_coeff = np.einsum('i,ij->ij', norm, mo_coeff)
        
        aoidx = order_ao_index(bas)
        
        f.write('[MO]\n')
        for imo in range(nmo):
            f.write(f' Sym= A\n')
            f.write(f' Ene= {mo_energy[imo]:15.10g}\n')
            f.write(f' Spin= Alpha\n')
            f.write(f' Occup= {mo_occ[imo]:10.5f}\n')
            for i, j in enumerate(aoidx):
                f.write(f' {i+1:3d}    {mo_coeff[j, imo]:18.14g}\n')
    
    print(f'Molden file saved to {filename}')


def _is_casscf(obj):
    return hasattr(obj, 'ncas') and hasattr(obj, 'nelecas') and hasattr(obj, 'ncore')


def to_molden(obj, filename, state=0):
    """
    Molden output
    
    Parameters
    ----------
    obj : RHF or CASSCF object
    filename : str
    state : int, optional
    
    Examples
    --------
    >>> from pyqed.qchem.tools.molden import to_molden
    >>> 
    >>> mf = mol.RHF().run()
    >>> to_molden(mf, 'rhf.molden')
    >>> 
    >>> mc = CASSCF(mf, ncas=4, nelecas=2).run()
    >>> to_molden(mc, 'casscf.molden')
    """
    if _is_casscf(obj):
        
        mol = obj.mol
        mo_coeff = obj.mo_coeff
        
        if mol._bas is None:
            raise RuntimeError("mol._bas is None. use mol.build(driver='gbasis')")
        
        ncore = obj.ncore
        ncas = obj.ncas
        nao = mol.nao
        nmo_cas = mo_coeff.shape[1]
        
        if nmo_cas < nao:
            S = mol.overlap
            mo_vir = obj.mf.mo_coeff[:, nmo_cas:].copy()
            overlap_ov = mo_coeff.T @ S @ mo_vir
            mo_vir = mo_vir - mo_coeff @ overlap_ov
            
            S_vir = mo_vir.T @ S @ mo_vir
            eigval, eigvec = np.linalg.eigh(S_vir)
            keep = eigval > 1e-10
            mo_vir = mo_vir @ eigvec[:, keep] @ np.diag(1.0 / np.sqrt(eigval[keep]))
            mo_coeff = np.hstack([mo_coeff, mo_vir])
        
        nmo = mo_coeff.shape[1]
        mo_energy = np.arange(nmo, dtype=float)
        mo_occ = np.zeros(nmo)
        mo_occ[:ncore] = 2.0
        
        if hasattr(obj, 'SC1') and obj.SC1 is not None:
            dm1_cas = obj.make_rdm1(state)
            mo_occ[ncore:ncore+ncas] = np.diag(dm1_cas)
        else:
            mo_occ[ncore:ncore+ncas] = obj.nelecas / ncas
        
        dump_molden(mol, filename, mo_coeff, mo_energy, mo_occ)
    
    else:
        dump_molden(obj.mol, filename, obj.mo_coeff, obj.mo_energy, obj.mo_occ)