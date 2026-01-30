#!/usr/bin/env python
"""
Molecular orbital visualization tools for PyQED

Features:
- MO energy level diagram
- 3D orbital isosurface visualization

Author: Ruoxi
"""

import numpy as np


class View:
    """
    Molecular orbital visualization class
    
    Parameters
    ----------
    mf_or_mc : RHF, UHF, or CASSCF object
        PyQED calculation object with mo_coeff and mo_energy
    
    Examples
    --------
    >>> from pyqed import Molecule
    >>> mol = Molecule(atom=[['H', (0, 0, 0)], ['F', (0, 0, 1.1)]], basis='sto6g')
    >>> mol.build()
    >>> mf = mol.RHF().run()
    >>> 
    >>> viz = View(mf)
    >>> viz.mo_energy()  
    >>> viz.orbital(0)   
    >>> viz.orbital(1)   
    """
    
    def __init__(self, mf_or_mc):
        self.obj = mf_or_mc
        
        if hasattr(mf_or_mc, 'mol'):
            self.mol = mf_or_mc.mol
        else:
            raise ValueError("Input object must have 'mol' attribute")
        
        if hasattr(mf_or_mc, 'mo_coeff'):
            self.mo_coeff = mf_or_mc.mo_coeff
        else:
            raise ValueError("Input object must have 'mo_coeff' attribute")

        self.mo_energy = getattr(mf_or_mc, 'mo_energy', None)
        self.mo_occ = getattr(mf_or_mc, 'mo_occ', None)
    
    def mo_energy_diagram(self, figsize=(10, 6), show=True):
        """
        Plot MO energy level diagram
        
        Parameters
        ----------
        figsize : tuple
            Figure size
        show : bool
            Whether to call plt.show()
        
        Returns
        -------
        fig, ax : matplotlib figure and axes
        """
        import matplotlib.pyplot as plt
        from matplotlib import cm
        
        if self.mo_energy is None:
            raise ValueError("MO energies not available")
        
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
        
        n_mo = len(self.mo_energy)
        cmap = cm.get_cmap("tab20")
        colors = cmap(np.linspace(0, 1, n_mo))
        
        pos = []
        for i, energy in enumerate(self.mo_energy):
            left = 3 * i
            right = 3 * i + 2.5
            length = right - left
            
            line, = ax.plot([left, right], [energy, energy], color=colors[i], lw=3)
            if self.mo_occ is not None:
                occ = self.mo_occ[i]
                electron_x, electron_y = None, None
                if occ >= 1.99:  # doubly occupied
                    electron_x = [left + 0.25 * length, left + 0.75 * length]
                    electron_y = [energy, energy]
                elif occ >= 0.5:  # singly occupied
                    electron_x = [left + 0.5 * length]
                    electron_y = [energy]
                
                if electron_x and electron_y:
                    ax.scatter(electron_x, electron_y, color=line.get_color(), s=50, 
                              marker='o', zorder=5)
            
            pos.append(left + 0.5 * length)
        
        ax.axhline(y=0, ls=":", color="k", alpha=0.5)
        ax.set_xticks(pos)
        ax.set_xticklabels([f"{i}" for i in range(n_mo)])
        ax.set_xlabel("MO index")
        ax.set_ylabel("Energy (Hartree)")
        ax.set_title("Molecular Orbital Energy Levels")
        
        if show:
            plt.show()
        
        return fig, ax
    
    mo_energy = mo_energy_diagram
    
    def _compute_ao_on_grid(self, coords):
        """
        Compute AO values on a grid of points using gbasis
        
        Parameters
        ----------
        coords : np.ndarray (N, 3)
            Grid coordinates
        
        Returns
        -------
        ao_values : np.ndarray (N, nao)
            AO values at each grid point
        """
        mol = self.mol
        bas = mol._bas
        
        if bas is None:
            raise RuntimeError("mol._bas is None. Please use mol.build()")
        
        try:
            from gbasis.evals.eval import evaluate_basis
            ao_values = evaluate_basis(bas, coords)
            if ao_values.shape[0] != coords.shape[0]:
                ao_values = ao_values.T
            return ao_values
        except (ImportError, Exception) as e:
            print(f"gbasis evaluate_basis failed: {e}, using manual calculation")
        
        #####manual#####
        npts = coords.shape[0]
        nao = mol.nao
        ao_values = np.zeros((npts, nao))
        
        ao_idx = 0
        for shell in bas:
            l = shell.angmom
            center = np.array(shell.coord)
            exps = shell.exps
            coeffs = shell.coeffs  # shape: (nprim, nctr)
            nctr = coeffs.shape[1]
            
            dx = coords[:, 0] - center[0]
            dy = coords[:, 1] - center[1]
            dz = coords[:, 2] - center[2]
            r2 = dx**2 + dy**2 + dz**2
            
            for ictr in range(nctr):
                
                radial = np.zeros(npts)
                for ip in range(len(exps)):
                    alpha = exps[ip]
                    norm_prim = (2 * alpha / np.pi) ** 0.75
                    if l == 1:
                        norm_prim *= (4 * alpha) ** 0.5
                    elif l == 2:
                        norm_prim *= (16 * alpha**2 / 3) ** 0.5
                    
                    radial += coeffs[ip, ictr] * norm_prim * np.exp(-alpha * r2)
                
                if l == 0:  # s
                    ao_values[:, ao_idx] = radial
                    ao_idx += 1
                elif l == 1:  #p_{-1}, p_0, p_{+1}  y, z, x
                    ao_values[:, ao_idx] = dy * radial      # p_y
                    ao_values[:, ao_idx + 1] = dz * radial  # p_z
                    ao_values[:, ao_idx + 2] = dx * radial  # p_x
                    ao_idx += 3
                elif l == 2:  # d: 5 functions
                    # d_{-2}, d_{-1}, d_0, d_{+1}, d_{+2}
                    # xy, yz, (3z²-r²)/2, xz, (x²-y²)/2
                    ao_values[:, ao_idx] = dx * dy * radial * np.sqrt(3)
                    ao_values[:, ao_idx + 1] = dy * dz * radial * np.sqrt(3)
                    ao_values[:, ao_idx + 2] = (3 * dz**2 - r2) * radial * 0.5
                    ao_values[:, ao_idx + 3] = dx * dz * radial * np.sqrt(3)
                    ao_values[:, ao_idx + 4] = (dx**2 - dy**2) * radial * np.sqrt(3) * 0.5
                    ao_idx += 5
                else:
                    raise NotImplementedError(f"l={l} not implemented")
        
        return ao_values
    
    def _compute_mo_on_grid(self, mo_idx, coords):
        """
        Compute MO values on a grid
        
        Parameters
        ----------
        mo_idx : int
            MO index
        coords : np.ndarray (N, 3)
            Grid coordinates
        
        Returns
        -------
        mo_values : np.ndarray (N,)
            MO values at each grid point
        """
        ao_values = self._compute_ao_on_grid(coords)
        mo_coeff = self.mo_coeff[:, mo_idx]
        return ao_values @ mo_coeff
    
    def cube_data(self, mo_idx, nx=60, ny=60, nz=60, margin=3.0):
        """
        Generate cube file data for a molecular orbital
        
        Parameters
        ----------
        mo_idx : int
            MO index (0-based)
        nx, ny, nz : int
            Number of grid points in each direction
        margin : float
            Margin around molecule (in Bohr)
        
        Returns
        -------
        grid_data : dict
            Dictionary containing origin, axes, and orbital values
        """
        mol = self.mol
        coords = mol.atom_coords()
        
        # Determine box
        min_coords = np.min(coords, axis=0) - margin
        max_coords = np.max(coords, axis=0) + margin
        box = max_coords - min_coords
        
        # Create grid
        xs = np.linspace(0, box[0], nx)
        ys = np.linspace(0, box[1], ny)
        zs = np.linspace(0, box[2], nz)
        
        # Grid spacing
        dx = box[0] / (nx - 1) if nx > 1 else box[0]
        dy = box[1] / (ny - 1) if ny > 1 else box[1]
        dz = box[2] / (nz - 1) if nz > 1 else box[2]
        
        # Generate all grid points
        grid_points = []
        for ix in range(nx):
            for iy in range(ny):
                for iz in range(nz):
                    grid_points.append([
                        min_coords[0] + ix * dx,
                        min_coords[1] + iy * dy,
                        min_coords[2] + iz * dz
                    ])
        grid_points = np.array(grid_points)
        
        # Compute MO values
        mo_values = self._compute_mo_on_grid(mo_idx, grid_points)
        mo_values = mo_values.reshape(nx, ny, nz)
        
        return {
            'origin': min_coords,
            'axes': np.array([[dx, 0, 0], [0, dy, 0], [0, 0, dz]]),
            'nx': nx, 'ny': ny, 'nz': nz,
            'values': mo_values
        }
    
    def write_cube(self, mo_idx, filename, nx=60, ny=60, nz=60, margin=3.0):
        """
        Write MO to a Gaussian cube file
        
        Parameters
        ----------
        mo_idx : int
            MO index (0-based)
        filename : str
            Output filename
        nx, ny, nz : int
            Number of grid points
        margin : float
            Margin around molecule (in Bohr)
        """
        mol = self.mol
        data = self.cube_data(mo_idx, nx, ny, nz, margin)
        
        with open(filename, 'w') as f:
            f.write(f'MO {mo_idx} generated by PyQED\n')
            f.write('Cube file\n')
            
            # Number of atoms and origin
            f.write(f'{mol.natom:5d} {data["origin"][0]:12.6f} {data["origin"][1]:12.6f} {data["origin"][2]:12.6f}\n')
            
            # Grid specification
            f.write(f'{data["nx"]:5d} {data["axes"][0, 0]:12.6f} {data["axes"][0, 1]:12.6f} {data["axes"][0, 2]:12.6f}\n')
            f.write(f'{data["ny"]:5d} {data["axes"][1, 0]:12.6f} {data["axes"][1, 1]:12.6f} {data["axes"][1, 2]:12.6f}\n')
            f.write(f'{data["nz"]:5d} {data["axes"][2, 0]:12.6f} {data["axes"][2, 1]:12.6f} {data["axes"][2, 2]:12.6f}\n')
            
            # Atoms
            coords = mol.atom_coords()
            for ia in range(mol.natom):
                chg = mol.atom_charge(ia)
                f.write(f'{chg:5d} {float(chg):12.6f} {coords[ia, 0]:12.6f} {coords[ia, 1]:12.6f} {coords[ia, 2]:12.6f}\n')
            
            # Orbital values
            values = data['values']
            for ix in range(data['nx']):
                for iy in range(data['ny']):
                    for iz in range(data['nz']):
                        f.write(f' {values[ix, iy, iz]:12.5E}')
                        if (iz + 1) % 6 == 0:
                            f.write('\n')
                    if data['nz'] % 6 != 0:
                        f.write('\n')
        
        print(f'Cube file saved to {filename}')
    
    def orbital(self, mo_idx, isovalue=0.02, nx=50, ny=50, nz=50, margin=3.0,
                colors=('blue', 'red'), opacity=0.6, show_atoms=True, 
                backend='matplotlib', figsize=(8, 8), show=True):
        """
        Visualize a molecular orbital as 3D isosurface
        
        Parameters
        ----------
        mo_idx : int
            MO index (0-based)
        isovalue : float
            Isosurface value (default 0.02)
        nx, ny, nz : int
            Number of grid points
        margin : float
            Margin around molecule (in Bohr)
        colors : tuple
            Colors for positive and negative lobes
        opacity : float
            Surface opacity (0-1)
        show_atoms : bool
            Whether to show atoms
        backend : str
            'matplotlib' or 'plotly'
        figsize : tuple
            Figure size for matplotlib
        show : bool
            Whether to display the plot
        
        Returns
        -------
        fig : figure object
        """
        data = self.cube_data(mo_idx, nx, ny, nz, margin)
        
        if backend == 'matplotlib':
            return self._plot_orbital_matplotlib(
                data, mo_idx, isovalue, colors, opacity, show_atoms, figsize, show, margin
            )
        elif backend == 'plotly':
            return self._plot_orbital_plotly(
                data, mo_idx, isovalue, colors, opacity, show_atoms, show
            )
        else:
            raise ValueError(f"Unknown backend: {backend}")
    
    def _plot_orbital_matplotlib(self, data, mo_idx, isovalue, colors, opacity, 
                                  show_atoms, figsize, show, margin=3.0):
        """Plot orbital using matplotlib"""
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        try:
            from skimage import measure
        except ImportError:
            raise ImportError("Please install scikit-image: pip install scikit-image")
        
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        values = data['values']
        origin = data['origin']
        dx = data['axes'][0, 0]
        dy = data['axes'][1, 1]
        dz = data['axes'][2, 2]
        
        # Plot positive isosurface
        try:
            verts, faces, _, _ = measure.marching_cubes(values, isovalue)
            # Scale vertices to real coordinates
            verts[:, 0] = verts[:, 0] * dx + origin[0]
            verts[:, 1] = verts[:, 1] * dy + origin[1]
            verts[:, 2] = verts[:, 2] * dz + origin[2]
            ax.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2],
                           color=colors[0], alpha=opacity, shade=True)
        except ValueError:
            pass  # No isosurface at this level
        
        # Plot negative isosurface
        try:
            verts, faces, _, _ = measure.marching_cubes(values, -isovalue)
            verts[:, 0] = verts[:, 0] * dx + origin[0]
            verts[:, 1] = verts[:, 1] * dy + origin[1]
            verts[:, 2] = verts[:, 2] * dz + origin[2]
            ax.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2],
                           color=colors[1], alpha=opacity, shade=True)
        except ValueError:
            pass
        
        # Plot atoms
        if show_atoms:
            mol = self.mol
            coords = mol.atom_coords()
            
            # Atom colors and sizes
            atom_colors = {
                'H': 'white', 'C': 'gray', 'N': 'blue', 'O': 'red',
                'F': 'green', 'S': 'yellow', 'Li': 'purple', 'Na': 'purple',
                'Cl': 'green', 'Br': 'brown', 'P': 'orange'
            }
            atom_sizes = {
                'H': 100, 'C': 200, 'N': 200, 'O': 200, 'F': 180,
                'S': 250, 'Li': 220, 'Na': 280, 'Cl': 230, 'Br': 260, 'P': 240
            }
            
            for ia in range(mol.natom):
                symb = mol.atom_symbol(ia)
                color = atom_colors.get(symb, 'gray')
                size = atom_sizes.get(symb, 200)
                ax.scatter(coords[ia, 0], coords[ia, 1], coords[ia, 2],
                          c=color, s=size, edgecolors='black', linewidths=0.5)
        
        ax.set_xlabel('X (Bohr)')
        ax.set_ylabel('Y (Bohr)')
        ax.set_zlabel('Z (Bohr)')
        ax.set_title(f'MO {mo_idx} (isovalue = {isovalue})')
        
        # Equal aspect ratio
        coords = self.mol.atom_coords()
        max_range = np.max(np.ptp(coords, axis=0)) / 2 + margin
        mid = np.mean(coords, axis=0)
        ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
        ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
        ax.set_zlim(mid[2] - max_range, mid[2] + max_range)
        
        if show:
            plt.show()
        
        return fig
    
    def _plot_orbital_plotly(self, data, mo_idx, isovalue, colors, opacity, 
                             show_atoms, show):
        """Plot orbital using plotly (interactive)"""
        try:
            import plotly.graph_objects as go
        except ImportError:
            raise ImportError("Please install plotly: pip install plotly")
        
        values = data['values']
        origin = data['origin']
        
        # create coordinate arrays
        x = np.linspace(origin[0], origin[0] + data['axes'][0, 0] * (data['nx'] - 1), data['nx'])
        y = np.linspace(origin[1], origin[1] + data['axes'][1, 1] * (data['ny'] - 1), data['ny'])
        z = np.linspace(origin[2], origin[2] + data['axes'][2, 2] * (data['nz'] - 1), data['nz'])
        
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        fig = go.Figure()
        
        # Positive isosurface
        fig.add_trace(go.Isosurface(
            x=X.flatten(), y=Y.flatten(), z=Z.flatten(),
            value=values.flatten(),
            isomin=isovalue, isomax=isovalue,
            surface_count=1,
            colorscale=[[0, colors[0]], [1, colors[0]]],
            showscale=False,
            opacity=opacity,
            caps=dict(x_show=False, y_show=False, z_show=False)
        ))
        
        # Negative isosurface
        fig.add_trace(go.Isosurface(
            x=X.flatten(), y=Y.flatten(), z=Z.flatten(),
            value=values.flatten(),
            isomin=-isovalue, isomax=-isovalue,
            surface_count=1,
            colorscale=[[0, colors[1]], [1, colors[1]]],
            showscale=False,
            opacity=opacity,
            caps=dict(x_show=False, y_show=False, z_show=False)
        ))
        
        # Plot atoms
        if show_atoms:
            mol = self.mol
            coords = mol.atom_coords()
            symbols = mol.atom_symbols()
            
            atom_colors_map = {
                'H': 'white', 'C': 'gray', 'N': 'blue', 'O': 'red',
                'F': 'green', 'S': 'yellow', 'Li': 'purple'
            }
            
            for ia in range(mol.natom):
                symb = symbols[ia]
                color = atom_colors_map.get(symb, 'gray')
                fig.add_trace(go.Scatter3d(
                    x=[coords[ia, 0]], y=[coords[ia, 1]], z=[coords[ia, 2]],
                    mode='markers+text',
                    marker=dict(size=10, color=color, line=dict(width=1, color='black')),
                    text=[symb],
                    textposition='top center',
                    showlegend=False
                ))
        
        fig.update_layout(
            title=f'MO {mo_idx} (isovalue = {isovalue})',
            scene=dict(
                xaxis_title='X (Bohr)',
                yaxis_title='Y (Bohr)',
                zaxis_title='Z (Bohr)',
                aspectmode='data'
            )
        )
        
        if show:
            fig.show()
        
        return fig
    
def orbital(mf, mo_idx, **kwargs):
    """
    Convenience function to visualize an orbital
    
    Parameters
    ----------
    mf : RHF or CASSCF object
        PyQED calculation object
    mo_idx : int
        MO index
    **kwargs : dict
        Arguments passed to View.orbital()
    
    Examples
    --------
    >>> from pyqed import Molecule
    >>> from pyqed.qchem.tools.view import orbital
    >>> 
    >>> mol = Molecule(atom=[['H', (0, 0, 0)], ['F', (0, 0, 1.1)]], basis='sto6g')
    >>> mol.build()
    >>> mf = mol.RHF().run()
    >>> orbital(mf, 0)  
    """
    viz = View(mf)
    return viz.orbital(mo_idx, **kwargs)


def write_cube(mf, mo_idx, filename, **kwargs):
    """
    Convenience function to write a cube file
    
    Parameters
    ----------
    mf : RHF or CASSCF object
        PyQED calculation object
    mo_idx : int
        MO index
    filename : str
        Output filename
    **kwargs : dict
        Arguments passed to View.write_cube()
    """
    viz = View(mf)
    return viz.write_cube(mo_idx, filename, **kwargs)


# ============================================================
# Test
# ============================================================
if __name__ == '__main__':
    from pyqed import Molecule
    
    # 使用列表格式的原子坐标（避免字符串解析的bug）
    mol = Molecule(atom=[['H', (0., 0., 0.)], ['F', (0., 0., 2.0)]], 
                   basis='sto6g', unit='bohr')
    mol.build()
    
    mf = mol.RHF()
    mf.run()
    
    print("MO energies:", mf.mo_energy)
    print("MO occupations:", mf.mo_occ)
    
    # Create visualization object
    viz = View(mf)
    
    # Plot MO energy diagram
    print("\nPlotting MO energy diagram...")
    viz.mo_energy_diagram(show=False)
    
    # Write cube file
    print("\nWriting cube file for MO 0...")
    viz.write_cube(0, 'mo_0.cube', nx=40, ny=40, nz=40)
    
    # Visualize orbital (requires scikit-image)
    print("\nVisualizing MO 0...")
    try:
        viz.orbital(0, isovalue=0.02, backend='matplotlib', show=True)
    except ImportError as e:
        print(f"Could not visualize: {e}")