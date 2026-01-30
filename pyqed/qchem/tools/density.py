#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Real-space electron density from CASCI reduced density matrix

σ_e(r; R) = Σ_{μν} D_{μν} χ_μ(r) χ_ν(r)

where D is the 1-RDM in AO basis and χ are atomic orbital basis functions.

Author: Ruoxi
"""

import numpy as np
from opt_einsum import contract
from scipy.special import factorial2


# ============================================================================
# Gaussian Type Orbital (GTO) evaluation
# ============================================================================

ANGULAR_MOMENTUM = {
    0: [(0, 0, 0)],  # s
    1: [(1, 0, 0), (0, 1, 0), (0, 0, 1)],  # p: px, py, pz
    2: [(2, 0, 0), (1, 1, 0), (1, 0, 1),   # d: dxx, dxy, dxz
        (0, 2, 0), (0, 1, 1), (0, 0, 2)],  #    dyy, dyz, dzz
    3: [(3, 0, 0), (2, 1, 0), (2, 0, 1),   # f
        (1, 2, 0), (1, 1, 1), (1, 0, 2),
        (0, 3, 0), (0, 2, 1), (0, 1, 2), (0, 0, 3)],
}


def gto_norm(l, alpha):
    """
    Normalization constant for a primitive Cartesian GTO.
    """
    double_fact = factorial2(2*l - 1, exact=True) if l > 0 else 1
    norm = (2*alpha/np.pi)**(3/4) * (4*alpha)**(l/2) / np.sqrt(float(double_fact))
    return norm


def eval_gto_primitive(coords, center, alpha, lx, ly, lz):
    """
    Evaluate a single primitive Cartesian GTO on a grid.
    
    χ(r) = N * (x-Ax)^lx * (y-Ay)^ly * (z-Az)^lz * exp(-α|r-A|²)
    """
    dx = coords[:, 0] - center[0]
    dy = coords[:, 1] - center[1]
    dz = coords[:, 2] - center[2]
    
    r2 = dx**2 + dy**2 + dz**2
    angular = (dx**lx) * (dy**ly) * (dz**lz)
    
    l = lx + ly + lz
    norm = gto_norm(l, alpha)
    radial = norm * np.exp(-alpha * r2)
    
    return angular * radial


def eval_contracted_gto(coords, center, exponents, coeffs, l):
    """
    Evaluate a contracted GTO shell on a grid.
    
    χ_μ(r) = Σ_i c_i * g_i(r)
    """
    npoints = coords.shape[0]
    ang_components = ANGULAR_MOMENTUM[l]
    nfunc = len(ang_components)
    
    values = np.zeros((npoints, nfunc))
    
    for ifunc, (lx, ly, lz) in enumerate(ang_components):
        for alpha, coef in zip(exponents, coeffs):
            values[:, ifunc] += coef * eval_gto_primitive(
                coords, center, alpha, lx, ly, lz
            )
    
    return values


def eval_ao(mol, coords):
    """
    Evaluate all AOs on a grid of points.
    
    Uses gbasis GeneralizedContractionShell objects from PyQED.
    
    Parameters
    ----------
    mol : PyQED Molecule object
        Must have _bas attribute containing gbasis shell objects
    coords : ndarray (npoints, 3)
        Grid coordinates in Bohr
    
    Returns
    -------
    ao_values : ndarray (npoints, nao)
        AO values at each grid point
    """
    npoints = coords.shape[0]
    shells = mol._bas
    nao = mol.nao
    
    ao_values = np.zeros((npoints, nao))
    iao = 0
    
    for shell in shells:
        # gbasis shell attributes:
        # - coord: atom center (3,)
        # - exps: exponents array
        # - coeffs: contraction coefficients (nprim,) or (nprim, ncon)
        # - angmom: angular momentum (int)
        
        center = shell.coord
        exponents = shell.exps
        l = shell.angmom
        
        coeffs = shell.coeffs
        if coeffs.ndim == 1:
            coeffs = coeffs.reshape(-1, 1)
        
        # For each contraction in this shell
        for icon in range(coeffs.shape[1]):
            coef = coeffs[:, icon]
            shell_values = eval_contracted_gto(coords, center, exponents, coef, l)
            nfunc = shell_values.shape[1]
            ao_values[:, iao:iao+nfunc] = shell_values
            iao += nfunc
    
    return ao_values


# ============================================================================
# Density matrix and electron density
# ============================================================================

def make_rdm1_ao(casci, state_id=0):
    """
    Get the 1-RDM in AO basis from CASCI calculation.
    
    Parameters
    ----------
    casci : CASCI object
        A converged CASCI object from PyQED
    state_id : int
        Which electronic state (0 = ground state)
    
    Returns
    -------
    dm_ao : ndarray (nao, nao)
        1-RDM in AO basis
    """
    ncore = casci.ncore
    ncas = casci.ncas
    
    # MO coefficients from mf
    mo_coeff = casci.mf.mo_coeff  # (nao, nmo)
    
    # RDM in active MO basis make_rdm1
    dm_cas = casci.make_rdm1(state_id)  # (ncas, ncas)
    
    # Build full MO-basis RDM
    nmo = mo_coeff.shape[1]
    dm_mo = np.zeros((nmo, nmo))
    
    # Core orbitals doubly occupied
    for i in range(ncore):
        dm_mo[i, i] = 2.0
    
    # Active space contribution
    dm_mo[ncore:ncore+ncas, ncore:ncore+ncas] = dm_cas
    
    # Transform to AO basis: D_AO = C @ D_MO @ C^T
    dm_ao = contract('pi, ij, qj -> pq', mo_coeff, dm_mo, mo_coeff)
    
    return dm_ao


def electron_density_on_grid(casci, coords, state_id=0):
    """
    Calculate electron density σ_e(r) on a grid of points.
    
    σ_e(r) = Σ_{μν} D_{μν} χ_μ(r) χ_ν(r)
    
    Parameters
    ----------
    casci : CASCI object
        Converged CASCI calculation
    coords : ndarray (npoints, 3)
        Grid coordinates in Bohr
    state_id : int
        Electronic state index
    
    Returns
    -------
    rho : ndarray (npoints,)
        Electron density at each grid point
    """
    dm_ao = make_rdm1_ao(casci, state_id)
    mol = casci.mol
    ao = eval_ao(mol, coords)  # (npoints, nao)
    rho = contract('pm, mn, pn -> p', ao, dm_ao, ao)
    return rho


def make_cubic_grid(mol, margin=5.0, npoints=50):
    """
    Generate a cubic grid around the molecule.
    """
    atom_coords = mol.atom_coords()
    
    min_coords = atom_coords.min(axis=0) - margin
    max_coords = atom_coords.max(axis=0) + margin
    
    if isinstance(npoints, int):
        npoints = (npoints, npoints, npoints)
    
    x = np.linspace(min_coords[0], max_coords[0], npoints[0])
    y = np.linspace(min_coords[1], max_coords[1], npoints[1])
    z = np.linspace(min_coords[2], max_coords[2], npoints[2])
    
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    coords = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    
    return coords, npoints, (x, y, z)


def electron_density_cube(casci, state_id=0, margin=5.0, npoints=50, filename=None):
    """
    Calculate electron density on a cubic grid and optionally save to cube file.
    """
    mol = casci.mol
    coords, grid_shape, grid_axes = make_cubic_grid(mol, margin, npoints)
    rho_flat = electron_density_on_grid(casci, coords, state_id)
    rho = rho_flat.reshape(grid_shape)
    
    if filename is not None:
        write_cube(filename, mol, rho, grid_axes)
    
    return rho, grid_axes


def write_cube(filename, mol, data, grid_axes, comment='Electron density'):
    """
    Write data to Gaussian cube format.
    """
    x, y, z = grid_axes
    nx, ny, nz = len(x), len(y), len(z)
    
    dx = x[1] - x[0] if nx > 1 else 1.0
    dy = y[1] - y[0] if ny > 1 else 1.0
    dz = z[1] - z[0] if nz > 1 else 1.0
    
    origin = np.array([x[0], y[0], z[0]])
    atom_coords = mol.atom_coords()
    atom_charges = mol.atom_charges()
    natom = len(atom_coords)
    
    with open(filename, 'w') as f:
        f.write(f'{comment}\n')
        f.write('Generated by PyQED electron_density module\n')
        f.write(f'{natom:5d} {origin[0]:12.6f} {origin[1]:12.6f} {origin[2]:12.6f}\n')
        f.write(f'{nx:5d} {dx:12.6f} {0.0:12.6f} {0.0:12.6f}\n')
        f.write(f'{ny:5d} {0.0:12.6f} {dy:12.6f} {0.0:12.6f}\n')
        f.write(f'{nz:5d} {0.0:12.6f} {0.0:12.6f} {dz:12.6f}\n')
        
        for i in range(natom):
            charge = atom_charges[i]
            coord = atom_coords[i]
            f.write(f'{int(charge):5d} {charge:12.6f} {coord[0]:12.6f} {coord[1]:12.6f} {coord[2]:12.6f}\n')
        
        count = 0
        for ix in range(nx):
            for iy in range(ny):
                for iz in range(nz):
                    f.write(f'{data[ix, iy, iz]:13.5E}')
                    count += 1
                    if count % 6 == 0:
                        f.write('\n')
                if count % 6 != 0:
                    f.write('\n')
                    count = 0
    
    print(f'Cube file written to {filename}')


def transition_density_on_grid(casci, coords, bra_id, ket_id=0):
    """
    Calculate transition density on a grid.
    
    γ^{IJ}(r) = Σ_{μν} D^{IJ}_{μν} χ_μ(r) χ_ν(r)
    """
    tdm_cas = casci.make_tdm1(bra_id, ket_id)
    
    ncore = casci.ncore
    ncas = casci.ncas
    mo_coeff = casci.mf.mo_coeff
    nmo = mo_coeff.shape[1]
    
    tdm_mo = np.zeros((nmo, nmo))
    tdm_mo[ncore:ncore+ncas, ncore:ncore+ncas] = tdm_cas
    
    tdm_ao = contract('pi, ij, qj -> pq', mo_coeff, tdm_mo, mo_coeff)
    
    mol = casci.mol
    ao = eval_ao(mol, coords)
    rho_trans = contract('pm, mn, pn -> p', ao, tdm_ao, ao)
    
    return rho_trans

def plot_density_3d_interactive(casci, state_id=0, margin=3.0, npoints=30, 
                                 isovalues=None, opacity=0.6):
   
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection  
    from skimage import measure
    
    mol = casci.mol
    atom_coords = mol.atom_coords()
    
    min_c = atom_coords.min(axis=0) - margin
    max_c = atom_coords.max(axis=0) + margin
    
    x = np.linspace(min_c[0], max_c[0], npoints)
    y = np.linspace(min_c[1], max_c[1], npoints)
    z = np.linspace(min_c[2], max_c[2], npoints)
    
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    coords = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    
    print("Calculating density on 3D grid...")
    rho = electron_density_on_grid(casci, coords, state_id)
    rho_3d = rho.reshape(npoints, npoints, npoints)
    
    if isovalues is None:
        rho_max = rho_3d.max()
        isovalues = [0.02 * rho_max, 0.1 * rho_max, 0.3 * rho_max]
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    colors = ['blue', 'green', 'red']
    
    for i, iso in enumerate(isovalues):
        try:
            verts, faces, _, _ = measure.marching_cubes(rho_3d, iso)
            
            
            verts[:, 0] = verts[:, 0] * (x[1] - x[0]) + x[0]
            verts[:, 1] = verts[:, 1] * (y[1] - y[0]) + y[0]
            verts[:, 2] = verts[:, 2] * (z[1] - z[0]) + z[0]
            
            
            mesh = Poly3DCollection(verts[faces],
                                    alpha=opacity,
                                    facecolor=colors[i % len(colors)],
                                    edgecolor='none')
            ax.add_collection3d(mesh)
            print(f"Isosurface ρ = {iso:.4f}: {len(faces)} triangles")
            
        except ValueError:
            print(f"Warning: No surface found for isovalue {iso:.4f}")
    
    
    ax.scatter(atom_coords[:, 0], atom_coords[:, 1], atom_coords[:, 2],
               c='black', s=200, marker='o')
    
    for i, sym in enumerate(mol.atom_symbols()):
        ax.text(atom_coords[i, 0], atom_coords[i, 1], atom_coords[i, 2] + 0.3,
                sym, fontsize=14, ha='center', color='black')
    
    
    ax.set_xlim(min_c[0], max_c[0])
    ax.set_ylim(min_c[1], max_c[1])
    ax.set_zlim(min_c[2], max_c[2])
    
    ax.set_xlabel('x (Bohr)')
    ax.set_ylabel('y (Bohr)')
    ax.set_zlabel('z (Bohr)')
    ax.set_title(f'Electron density isosurfaces (state {state_id})')
    
    plt.tight_layout()
    plt.show()


# def plot_density_3d_scatter(casci, state_id=0, margin=3.0, npoints=20, 
#                             threshold=0.01):
#     """    
#     Parameters
#     ----------
#     casci : CASCI object
#     state_id : int
#     margin : float
#     npoints : int
#     threshold : float
#     """
#     import matplotlib
#     matplotlib.use('TkAgg')
#     import matplotlib.pyplot as plt
#     from mpl_toolkits.mplot3d import Axes3D
    
#     mol = casci.mol
#     atom_coords = mol.atom_coords()
    
#     min_c = atom_coords.min(axis=0) - margin
#     max_c = atom_coords.max(axis=0) + margin
    
#     x = np.linspace(min_c[0], max_c[0], npoints)
#     y = np.linspace(min_c[1], max_c[1], npoints)
#     z = np.linspace(min_c[2], max_c[2], npoints)
    
#     X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
#     coords = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    
#     print("Calculating density...")
#     rho = electron_density_on_grid(casci, coords, state_id)
    
#     mask = rho > threshold * rho.max()
    
#     fig = plt.figure(figsize=(10, 8))
#     ax = fig.add_subplot(111, projection='3d')
    
#     scatter = ax.scatter(coords[mask, 0], coords[mask, 1], coords[mask, 2],
#                         c=rho[mask], cmap='viridis', s=50*rho[mask]/rho.max(),
#                         alpha=0.6)
    
#     #
#     ax.scatter(atom_coords[:, 0], atom_coords[:, 1], atom_coords[:, 2],
#                c='red', s=300, marker='o', edgecolors='black', linewidths=2)
    
#     for i, sym in enumerate(mol.atom_symbols()):
#         ax.text(atom_coords[i, 0], atom_coords[i, 1], atom_coords[i, 2] + 0.5,
#                 sym, fontsize=14, ha='center', fontweight='bold')
    
#     ax.set_xlabel('x (Bohr)')
#     ax.set_ylabel('y (Bohr)')
#     ax.set_zlabel('z (Bohr)')
#     ax.set_title(f'Electron density 3D (state {state_id})')
#     plt.colorbar(scatter, ax=ax, label=r'$\rho$ (a.u.)', shrink=0.6)
    
#     plt.tight_layout()
#     plt.show()


# Test


if __name__ == "__main__":
    
    from pyqed import Molecule
    from pyqed.qchem.mcscf.casci import CASCI
    
    # H2 molecule at equilibrium
    mol = Molecule(atom=[
        ['H', (0., 0., 0.)],
        ['H', (0., 0., 1.4)],
    ], basis='6-31g', unit='bohr')
    mol.build()
    
    print("Number of AOs:", mol.nao)
    print("Basis name:", mol.basis)

    mf = mol.RHF()
    mf.run()
    
    
    ncas, nelecas = (2, 2)
    mc = CASCI(mf, ncas, nelecas)
    mc.run(nstates=2)
    
    print("\n" + "="*60)
    print("Electron Density Calculation")
    print("="*60)

    z_points = np.linspace(-2, 4, 200)
    coords = np.column_stack([
        np.zeros_like(z_points),
        np.zeros_like(z_points),
        z_points
    ])
    
    rho_gs = electron_density_on_grid(mc, coords, state_id=0)
    
    try:
        import matplotlib.pyplot as plt
        
        fig, axes_plot = plt.subplots(1, 2, figsize=(12, 5))
        
        axes_plot[0].plot(z_points, rho_gs, 'b-', linewidth=2, label='CASCI')
        axes_plot[0].axvline(0, color='red', linestyle='--', alpha=0.7, label='H1')
        axes_plot[0].axvline(1.4, color='red', linestyle='--', alpha=0.7, label='H2')
        axes_plot[0].set_xlabel('z (Bohr)')
        axes_plot[0].set_ylabel(r'$\rho(0,0,z)$ (a.u.)')
        axes_plot[0].set_title('Electron density along bond axis (H₂)')
        axes_plot[0].legend()
        axes_plot[0].grid(True, alpha=0.3)
        
        x_pts = np.linspace(-3, 3, 60)
        z_pts = np.linspace(-2, 4, 80)
        X, Z = np.meshgrid(x_pts, z_pts)
        coords_2d = np.column_stack([X.ravel(), np.zeros(X.size), Z.ravel()])
        rho_2d = electron_density_on_grid(mc, coords_2d, state_id=0).reshape(Z.shape)
        
        im = axes_plot[1].contourf(X, Z, rho_2d, levels=30, cmap='viridis')
        axes_plot[1].scatter([0, 0], [0, 1.4], c='red', s=100, marker='o', label='H atoms')
        axes_plot[1].set_xlabel('x (Bohr)')
        axes_plot[1].set_ylabel('z (Bohr)')
        axes_plot[1].set_title('Electron density in xz-plane (H₂)')
        axes_plot[1].set_aspect('equal')
        plt.colorbar(im, ax=axes_plot[1], label=r'$\rho$ (a.u.)')
        
        plt.tight_layout()
        plt.savefig('electron_density_h2.png', dpi=150)
        print("\nPlot saved to electron_density_h2.png")
        
    except ImportError:
        print("\nMatplotlib not available, skipping plot")
    
    electron_density_cube(mc, state_id=0, margin=5.0, npoints=50, filename='h2_density.cube')
    