"""
NormalMode.py - Normal modes calculation and normal coordinate generation
    - NormalModeAnalyzer: Frequency calculation and normal mode analysis
    - NormalCoordGenerator: Generate Cartesian geometries along normal coordinates

Electronic Structure Calculations: 
    - pyscf (https://pyscf.org/)
    method supported: HF, DFT, CASCI, CASSCF
Quantum Dynamics:
    - pyqed 
    method supported: ab intio LDR

@author: Ruoxi Liu
"""

import numpy as np
import os
import sys
from contextlib import contextmanager
from pyscf import gto
from pyscf.data import nist
from pyscf.data.elements import MASSES as ELEMENT_MASSES


class NormalModeAnalyzer:
    """
    Normal mode analysis from optimized geometry.
    
    Attributes after analyze():
        frequencies_au: Vibrational frequencies in atomic units
        frequencies_cm: Vibrational frequencies in cm^-1
        modes: Mass-weighted normal mode eigenvectors (num_modes, 3N)
        hessian: Cartesian Hessian matrix (3N x 3N)
    """
    
    def __init__(self, atoms, basis="def2-SVP", charge=0, spin=0):
        """
        Initialize analyzer.
        
        Args:
            atoms: List of [symbol, (x, y, z)] in Angstrom
                   Example: [['C', (0.0, 0.0, 0.0)], ['O', (0.0, 0.0, 1.2)]]
            basis: Basis set
            charge: Molecular charge
            spin: 2S (0 for singlet, 1 for doublet, etc.)
        """
        self.mol = gto.M(
            atom=atoms,
            basis=basis,
            symmetry=False,
            charge=charge,
            spin=spin,
            unit="Angstrom"
        )
        self.basis = basis
        self.charge = charge
        self.spin = spin
        
        self.frequencies_au = None
        self.frequencies_cm = None
        self.modes = None
        self.hessian = None
        self.coords0 = None  # (natm, 3) in Angstrom
        self.atom_masses_au = None  # (natm,) in a.u.
        self.symbols = None  # list of atom symbols
        self._is_linear = False
        self.log_file = None
    
    @classmethod
    def from_xyz_file(cls, xyz_file, basis="def2-SVP", charge=0, spin=0):
        """
        Create analyzer from XYZ file.
        
        XYZ format:
            natom
            comment (can be empty)
            Symbol  x  y  z
            ...
        """
        atoms = cls._parse_xyz_file(xyz_file)
        return cls(atoms, basis, charge, spin)
    
    @staticmethod
    def _parse_xyz_file(xyz_file):
        """Parse XYZ file and return atoms list."""
        atoms = []
        with open(xyz_file, "r") as f:
            lines = f.readlines()
        for line in lines[2:]:
            parts = line.split()
            if len(parts) >= 4:
                atoms.append([parts[0], (float(parts[1]), float(parts[2]), float(parts[3]))])
        return atoms
    
    @contextmanager
    def _redirect_output(self, log_file):
        """Redirect stdout and stderr to both terminal and log file."""
        if log_file is None:
            yield
            return
        
        class Tee:
            def __init__(self, *streams):
                self.streams = streams
            def write(self, data):
                for s in self.streams:
                    s.write(data)
                    s.flush()
            def flush(self):
                for s in self.streams:
                    s.flush()
        
        with open(log_file, 'w') as f:
            old_stdout, old_stderr = sys.stdout, sys.stderr
            sys.stdout = Tee(old_stdout, f)
            sys.stderr = Tee(old_stderr, f)
            try:
                yield
            finally:
                sys.stdout, sys.stderr = old_stdout, old_stderr
    
    def run(self, method="dft", xc="b3lyp", is_linear=False, verbose=4,
            ncas=None, nelecas=None, state_average=None, nroots=1, target_state=0,
            log_file=None):
        """
        Run frequency calculation on the current (assumed optimized) geometry.
        
        Args:
            method: "dft", "hf", "casscf", "casci"
            xc: DFT functional (only for method="dft")
            is_linear: True for linear molecules (3N-5 modes instead of 3N-6)
            verbose: PySCF verbosity level
            log_file: Path to save terminal output (None to disable)
            
            CASSCF/CASCI specific:
                ncas: Number of active orbitals
                nelecas: Number of active electrons (int or tuple (nalpha, nbeta))
                state_average: List of weights for state-averaging, e.g., [0.5, 0.5]
                nroots: Number of roots for SA-CASSCF or CASCI
                target_state: Which state to use for gradients (0-indexed)
            
        Returns:
            self
        """
        with self._redirect_output(log_file):
            self._is_linear = is_linear
            
            method_lower = method.lower()
            if method_lower == "dft":
                self._run_dft(xc, verbose)
            elif method_lower == "hf":
                self._run_hf(verbose)
            elif method_lower in ("casscf", "casci"):
                if ncas is None or nelecas is None:
                    raise ValueError("ncas and nelecas must be specified for CASSCF/CASCI")
                self._run_casscf(method_lower, ncas, nelecas, state_average, 
                               nroots, target_state, verbose)
            else:
                raise NotImplementedError(f"Method '{method}' not implemented. "
                                          f"Available: 'dft', 'hf', 'casscf', 'casci'")
            
            self._diagonalize_hessian()
        return self
    
    def optimize_and_run(self, method="dft", xc="b3lyp", is_linear=False, 
                         verbose=4, conv_params=None,
                         ncas=None, nelecas=None, state_average=None, 
                         nroots=1, target_state=0,
                         log_file=None):
        """
        Optimize geometry first, then run frequency calculation.
        
        Args:
            method: "dft", "hf", "casscf", "casci"
            xc: DFT functional (only for method="dft")
            is_linear: True for linear molecules (3N-5 modes instead of 3N-6)
            verbose: PySCF verbosity level
            conv_params: dict of convergence parameters for optimizer
            log_file: Path to save terminal output (None to disable)
            
            CASSCF/CASCI specific:
                ncas: Number of active orbitals
                nelecas: Number of active electrons
                state_average: List of weights for state-averaging
                nroots: Number of roots
                target_state: Which state for gradients (0-indexed)
            
        Returns:
            self
        """
        with self._redirect_output(log_file):
            self._is_linear = is_linear
            
            method_lower = method.lower()
            if method_lower == "dft":
                self._optimize_and_run_dft(xc, verbose, conv_params)
            elif method_lower == "hf":
                self._optimize_and_run_hf(verbose, conv_params)
            elif method_lower in ("casscf", "casci"):
                if ncas is None or nelecas is None:
                    raise ValueError("ncas and nelecas must be specified for CASSCF/CASCI")
                self._optimize_and_run_casscf(method_lower, ncas, nelecas, state_average,
                                              nroots, target_state, verbose, conv_params)
            else:
                raise NotImplementedError(f"Method '{method}' not implemented.")
            
            self._diagonalize_hessian()
        return self
    
    def _run_dft(self, xc, verbose):
        mf = self.mol.KS(xc=xc)
        mf.verbose = verbose
        mf.kernel()
        h4 = mf.Hessian().kernel()
        natm = self.mol.natm
        self.hessian = h4.transpose(0, 2, 1, 3).reshape(natm * 3, natm * 3)
        self.coords0 = self.mol.atom_coords(unit="Angstrom")
        self.symbols = [self.mol.atom_symbol(i) for i in range(natm)]
        atom_masses_amu = self.mol.atom_mass_list(isotope_avg=True)
        self.atom_masses_au = atom_masses_amu * nist.AMU2AU
    
    def _run_hf(self, verbose):
        mf = self.mol.HF()
        mf.verbose = verbose
        mf.kernel()
        h4 = mf.Hessian().kernel()
        natm = self.mol.natm
        self.hessian = h4.transpose(0, 2, 1, 3).reshape(natm * 3, natm * 3)
        self.coords0 = self.mol.atom_coords(unit="Angstrom")
        self.symbols = [self.mol.atom_symbol(i) for i in range(natm)]
        atom_masses_amu = self.mol.atom_mass_list(isotope_avg=True)
        self.atom_masses_au = atom_masses_amu * nist.AMU2AU
    
    def _run_casscf(self, method, ncas, nelecas, state_average, nroots, target_state, verbose):
        """Run CASSCF/CASCI frequency calculation using finite difference Hessian."""
        from pyscf.tools import finite_diff
        from pyscf.mcscf import CASSCF, CASCI
        
        # Build CASSCF/CASCI object
        mf = self.mol.HF()
        mf.verbose = verbose
        mf.kernel()
        
        if method == "casscf":
            mc = CASSCF(mf, ncas, nelecas)
        else:  # casci
            mc = CASCI(mf, ncas, nelecas)
        
        mc.verbose = verbose
        
        # State-averaging
        if state_average is not None:
            mc = mc.state_average_(state_average)
            mc.nroots = len(state_average)
        elif nroots > 1:
            weights = [1.0/nroots] * nroots
            mc = mc.state_average_(weights)
        
        mc.kernel()
        
        # Get gradients object for target state
        if hasattr(mc, 'nroots') and mc.nroots > 1:
            mc_grad = mc.Gradients().as_scanner(state=target_state)
        else:
            mc_grad = mc.Gradients()
        
        # Compute Hessian via finite difference of gradients
        print(f"Computing Hessian via finite difference for {method.upper()}...")
        h4 = finite_diff.kernel(mc_grad)
        
        natm = self.mol.natm
        self.hessian = h4.transpose(0, 2, 1, 3).reshape(natm * 3, natm * 3)
        self.coords0 = self.mol.atom_coords(unit="Angstrom")
        self.symbols = [self.mol.atom_symbol(i) for i in range(natm)]
        atom_masses_amu = self.mol.atom_mass_list(isotope_avg=True)
        self.atom_masses_au = atom_masses_amu * nist.AMU2AU
    
    def _optimize_and_run_dft(self, xc, verbose, conv_params):
        """Run geometry optimization followed by frequency calculation."""
        from pyscf.geomopt.geometric_solver import optimize
        
        # SCF calculation
        mf = self.mol.KS(xc=xc)
        mf.verbose = verbose
        
        # Geometry optimization
        if conv_params is None:
            conv_params = {}
        
        mol_opt = optimize(mf, **conv_params)
        
        # Update molecule with optimized geometry
        self.mol = mol_opt
        
        # Run frequency calculation on optimized geometry
        mf_opt = mol_opt.KS(xc=xc)
        mf_opt.verbose = verbose
        mf_opt.kernel()
        
        h4 = mf_opt.Hessian().kernel()
        natm = mol_opt.natm
        self.hessian = h4.transpose(0, 2, 1, 3).reshape(natm * 3, natm * 3)
        self.coords0 = mol_opt.atom_coords(unit="Angstrom")
        self.symbols = [mol_opt.atom_symbol(i) for i in range(natm)]
        atom_masses_amu = mol_opt.atom_mass_list(isotope_avg=True)
        self.atom_masses_au = atom_masses_amu * nist.AMU2AU
    
    def _optimize_and_run_hf(self, verbose, conv_params):
        """Run HF geometry optimization followed by frequency calculation."""
        from pyscf.geomopt.geometric_solver import optimize
        
        mf = self.mol.HF()
        mf.verbose = verbose
        
        if conv_params is None:
            conv_params = {}
        
        mol_opt = optimize(mf, **conv_params)
        self.mol = mol_opt
        
        mf_opt = mol_opt.HF()
        mf_opt.verbose = verbose
        mf_opt.kernel()
        
        h4 = mf_opt.Hessian().kernel()
        natm = mol_opt.natm
        self.hessian = h4.transpose(0, 2, 1, 3).reshape(natm * 3, natm * 3)
        self.coords0 = mol_opt.atom_coords(unit="Angstrom")
        self.symbols = [mol_opt.atom_symbol(i) for i in range(natm)]
        atom_masses_amu = mol_opt.atom_mass_list(isotope_avg=True)
        self.atom_masses_au = atom_masses_amu * nist.AMU2AU
    
    def _optimize_and_run_casscf(self, method, ncas, nelecas, state_average,
                                  nroots, target_state, verbose, conv_params):
        """Run CASSCF/CASCI geometry optimization followed by frequency calculation."""
        from pyscf.geomopt.geometric_solver import optimize
        from pyscf.tools import finite_diff
        from pyscf.mcscf import CASSCF, CASCI
        
        # Build CASSCF/CASCI object
        mf = self.mol.HF()
        mf.verbose = verbose
        mf.kernel()
        
        if method == "casscf":
            mc = CASSCF(mf, ncas, nelecas)
        else:
            mc = CASCI(mf, ncas, nelecas)
        
        mc.verbose = verbose
        
        # State-averaging
        if state_average is not None:
            mc = mc.state_average_(state_average)
            mc.nroots = len(state_average)
        elif nroots > 1:
            weights = [1.0/nroots] * nroots
            mc = mc.state_average_(weights)
        
        mc.kernel()
        
        # Get scanner for optimization
        if hasattr(mc, 'nroots') and mc.nroots > 1:
            mc_scanner = mc.Gradients().as_scanner(state=target_state)
        else:
            mc_scanner = mc.Gradients().as_scanner()
        
        # Geometry optimization
        if conv_params is None:
            conv_params = {}
        
        mol_opt = optimize(mc_scanner, **conv_params)
        self.mol = mol_opt
        
        # Rebuild MC on optimized geometry for Hessian
        mf_opt = mol_opt.HF()
        mf_opt.verbose = verbose
        mf_opt.kernel()
        
        if method == "casscf":
            mc_opt = CASSCF(mf_opt, ncas, nelecas)
        else:
            mc_opt = CASCI(mf_opt, ncas, nelecas)
        
        mc_opt.verbose = verbose
        
        if state_average is not None:
            mc_opt = mc_opt.state_average_(state_average)
        elif nroots > 1:
            weights = [1.0/nroots] * nroots
            mc_opt = mc_opt.state_average_(weights)
        
        mc_opt.kernel()
        
        # Get gradients for Hessian calculation
        if hasattr(mc_opt, 'nroots') and mc_opt.nroots > 1:
            mc_grad = mc_opt.Gradients().as_scanner(state=target_state)
        else:
            mc_grad = mc_opt.Gradients()
        
        # Finite difference Hessian
        print(f"Computing Hessian via finite difference for {method.upper()}...")
        h4 = finite_diff.kernel(mc_grad)
        
        natm = mol_opt.natm
        self.hessian = h4.transpose(0, 2, 1, 3).reshape(natm * 3, natm * 3)
        self.coords0 = mol_opt.atom_coords(unit="Angstrom")
        self.symbols = [mol_opt.atom_symbol(i) for i in range(natm)]
        atom_masses_amu = mol_opt.atom_mass_list(isotope_avg=True)
        self.atom_masses_au = atom_masses_amu * nist.AMU2AU
    
    def _diagonalize_hessian(self):
        natm = self.mol.natm
        n_remove = 5 if self._is_linear else 6
        mvec = np.repeat(self.atom_masses_au, 3)
        Mhalf = 1.0 / np.sqrt(np.outer(mvec, mvec))
        weighted_hessian = self.hessian * Mhalf
        
        eigvals, eigvecs = np.linalg.eigh(weighted_hessian)
        
        freq_all_au = np.sqrt(np.abs(eigvals))
        freq_all_cm = freq_all_au * nist.HARTREE2WAVENUMBER
        
        num_modes = 3 * natm - n_remove
        self.frequencies_au = freq_all_au[n_remove:n_remove + num_modes]
        self.frequencies_cm = freq_all_cm[n_remove:n_remove + num_modes]
        self.modes = eigvecs.T[n_remove:n_remove + num_modes]
    
    def get_optimized_geometry(self):
        """
        Return the current (optimized) geometry as atoms list.
        
        Returns:
            atoms: List of [symbol, (x, y, z)] in Angstrom
        """
        if self.coords0 is None:
            raise RuntimeError("No geometry available. Run analyze() or optimize_and_analyze() first.")
        return [[s, tuple(c)] for s, c in zip(self.symbols, self.coords0)]
    
    def write_optimized_xyz(self, filename, comment=""):
        """
        Write optimized geometry to XYZ file.
        
        Args:
            filename: Output filename
            comment: Comment line (default: empty)
        """
        if self.coords0 is None:
            raise RuntimeError("No geometry available. Run analyze() or optimize_and_analyze() first.")
        
        with open(filename, "w") as f:
            f.write(f"{len(self.symbols)}\n")
            f.write(f"{comment}\n")
            for s, c in zip(self.symbols, self.coords0):
                f.write(f"{s}  {c[0]:.10f}  {c[1]:.10f}  {c[2]:.10f}\n")
        print(f"Written optimized geometry to {filename}")
    
    def print_frequencies(self):
        print("=" * 45)
        print(" Vibrational Frequencies (cm^-1)")
        print("=" * 45)
        for i, f in enumerate(self.frequencies_cm):
            print(f" Mode {i+1:3d}:  {f:12.4f}  cm^-1")
        print("=" * 45)
    
    def save(self, filename, fmt="npy"):
        """
        Save frequencies, modes, equilibrium geometry, masses, and symbols.
        
        Args:
            filename: Output filename (without extension)
            fmt: "npy" or "txt"
        
        Output files:
            {filename}_freq: (num_modes,) frequencies in a.u.
            {filename}_modes: (num_modes, 3N) mass-weighted eigenvectors
            {filename}_coords0: (natm, 3) equilibrium geometry in Angstrom
            {filename}_masses: (natm,) atomic masses in a.u.
            {filename}_symbols: list of atom symbols (only for npy, as .npy object)
        """
        if fmt == "npy":
            np.save(f"{filename}_freq.npy", self.frequencies_au)
            np.save(f"{filename}_modes.npy", self.modes)
            np.save(f"{filename}_coords0.npy", self.coords0)
            np.save(f"{filename}_masses.npy", self.atom_masses_au)
            np.save(f"{filename}_symbols.npy", np.array(self.symbols))
            print(f"Saved: {filename}_freq.npy, {filename}_modes.npy, "
                  f"{filename}_coords0.npy, {filename}_masses.npy, {filename}_symbols.npy")
        elif fmt == "txt":
            np.savetxt(f"{filename}_freq.txt", self.frequencies_au, fmt="%.12e")
            np.savetxt(f"{filename}_modes.txt", self.modes, delimiter=",", fmt="%.12e")
            np.savetxt(f"{filename}_coords0.txt", self.coords0, fmt="%.10f")
            np.savetxt(f"{filename}_masses.txt", self.atom_masses_au, fmt="%.12e")
            with open(f"{filename}_symbols.txt", "w") as f:
                f.write("\n".join(self.symbols))
            print(f"Saved: {filename}_freq.txt, {filename}_modes.txt, "
                  f"{filename}_coords0.txt, {filename}_masses.txt, {filename}_symbols.txt")
        else:
            raise ValueError(f"Unknown format: {fmt}. Use 'npy' or 'txt'.")


class NormalCoordGenerator:
    """
    Generate Cartesian geometries along normal coordinates.
    """
    
    def __init__(self, filename):
        """
        Initialize from saved normal mode data (npy format, full set of files).
        
        Args:
            filename: Base filename (without _freq.npy etc.)
                      Expected files: {filename}_freq.npy, {filename}_modes.npy,
                                     {filename}_coords0.npy, {filename}_masses.npy,
                                     {filename}_symbols.npy
        """
        self.frequencies_au = np.load(f"{filename}_freq.npy")
        self.modes = np.load(f"{filename}_modes.npy")
        self.coords0 = np.load(f"{filename}_coords0.npy")
        self.atom_masses_au = np.load(f"{filename}_masses.npy")
        self.symbols = list(np.load(f"{filename}_symbols.npy", allow_pickle=True))
        
        self._init_derived()
    
    @classmethod
    def from_files(cls, freq_file, modes_file, xyz_file, fmt="auto"):
        """
        Create generator from separate freq, modes, and xyz files.
        
        The xyz file provides coords0, symbols, and masses (via element lookup).
        
        Args:
            freq_file: Path to frequencies file (.npy or .txt)
            modes_file: Path to modes file (.npy or .txt)
            xyz_file: Path to XYZ file for geometry, symbols, masses
            fmt: "npy", "txt", or "auto" (detect from freq_file extension)
            
        Returns:
            NormalCoordGenerator instance
        """
        instance = cls.__new__(cls)
        
        # Detect format
        if fmt == "auto":
            fmt = "npy" if freq_file.endswith(".npy") else "txt"
        
        # Load frequencies and modes
        if fmt == "npy":
            instance.frequencies_au = np.load(freq_file)
            instance.modes = np.load(modes_file)
        else:
            instance.frequencies_au = np.loadtxt(freq_file)
            instance.modes = np.loadtxt(modes_file, delimiter=",")
            # Handle 1D case (single mode)
            if instance.modes.ndim == 1:
                instance.modes = instance.modes.reshape(1, -1)
        
        # Parse xyz file for coords0, symbols
        instance.symbols, instance.coords0 = cls._parse_xyz(xyz_file)
        
        # Get masses from element symbols
        instance.atom_masses_au = cls._get_masses_from_symbols(instance.symbols)
        
        instance._init_derived()
        return instance
    
    @classmethod
    def from_arrays(cls, frequencies_au, modes, coords0, symbols):
        """
        Create generator directly from numpy arrays.
        
        Args:
            frequencies_au: (num_modes,) frequencies in a.u.
            modes: (num_modes, 3N) mass-weighted eigenvectors
            coords0: (natm, 3) equilibrium geometry in Angstrom
            symbols: list of atom symbols
            
        Returns:
            NormalCoordGenerator instance
        """
        instance = cls.__new__(cls)
        instance.frequencies_au = np.asarray(frequencies_au)
        instance.modes = np.asarray(modes)
        instance.coords0 = np.asarray(coords0)
        instance.symbols = list(symbols)
        instance.atom_masses_au = cls._get_masses_from_symbols(instance.symbols)
        instance._init_derived()
        return instance
    
    def _init_derived(self):
        """Initialize derived quantities."""
        self.natm = len(self.symbols)
        self.num_modes = len(self.frequencies_au)
        self._sqrt_mass = np.sqrt(np.repeat(self.atom_masses_au, 3))
    
    @staticmethod
    def _parse_xyz(xyz_file):
        """
        Parse XYZ file and return symbols and coordinates.
        
        Returns:
            symbols: list of atom symbols
            coords: (natm, 3) array of coordinates in Angstrom
        """
        symbols = []
        coords = []
        with open(xyz_file, "r") as f:
            lines = f.readlines()
        for line in lines[2:]:
            parts = line.split()
            if len(parts) >= 4:
                symbols.append(parts[0])
                coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
        return symbols, np.array(coords)
    
    @staticmethod
    def _get_masses_from_symbols(symbols):
        """
        Get atomic masses in a.u. from element symbols.
        
        Args:
            symbols: list of element symbols
            
        Returns:
            masses_au: (natm,) array of masses in atomic units
        """
        masses_amu = []
        for sym in symbols:
            # ELEMENT_MASSES is indexed by atomic number, need to convert symbol
            from pyscf.data.elements import charge as get_charge
            z = get_charge(sym)
            masses_amu.append(ELEMENT_MASSES[z])
        return np.array(masses_amu) * nist.AMU2AU
    
    def get_geometry(self, mode_indices, q_values):
        """
        Get Cartesian geometry at given dimensionless normal coordinates.
        
        Args:
            mode_indices: int or list of mode indices (0-indexed)
            q_values: float or list of dimensionless coordinate values
                      (must match length of mode_indices)
            
        Returns:
            geometry: (natm, 3) Cartesian coordinates in Angstrom
        """
        if isinstance(mode_indices, int):
            mode_indices = [mode_indices]
            q_values = [q_values]
        
        if len(mode_indices) != len(q_values):
            raise ValueError("mode_indices and q_values must have same length")
        
        total_dx_bohr = np.zeros(3 * self.natm)
        
        for mode_idx, q in zip(mode_indices, q_values):
            omega = self.frequencies_au[mode_idx]
            L = self.modes[mode_idx]
            Q = q / np.sqrt(omega)
            dX = Q * L
            total_dx_bohr += dX / self._sqrt_mass
        
        dx_angstrom = total_dx_bohr.reshape(self.natm, 3) * nist.BOHR
        return self.coords0 + dx_angstrom
    
    def generate_geometries(self, mode_indices, q_ranges, n_points):
        """
        Generate geometries on a grid in normal coordinate space.
        
        Works for 1D, 2D...nD scans.
        
        Args:
            mode_indices: list of mode indices, e.g., [0] for 1D, [0, 2] for 2D
            q_ranges: list of (q_min, q_max) for each mode
            n_points: list of number of points for each mode
            
        Returns:
            dict with:
                'geometries': ndarray of shape (*n_points, natm, 3) in Angstrom
                'q_grids': list of 1D arrays for each mode
                'mode_indices': list of mode indices
                'frequencies_au': frequencies of selected modes (a.u.)
                'frequencies_cm': frequencies of selected modes (cm^-1)
                
        Examples:
            # 1D scan along mode 0
            result = gen.generate_geometries([0], [(-4, 4)], [9])
            # geometries shape: (9, natm, 3)
            
            # 2D scan along modes 0 and 2
            result = gen.generate_geometries([0, 2], [(-4, 4), (-3, 3)], [9, 7])
            # geometries shape: (9, 7, natm, 3)
        """
        ndim = len(mode_indices)
        
        if len(q_ranges) != ndim or len(n_points) != ndim:
            raise ValueError("mode_indices, q_ranges, n_points must have same length")
        
        q_grids = [np.linspace(q_ranges[i][0], q_ranges[i][1], n_points[i]) 
                   for i in range(ndim)]
        
        shape = tuple(n_points) + (self.natm, 3)
        geometries = np.zeros(shape)
        
        for idx in np.ndindex(tuple(n_points)):
            q_values = [q_grids[d][idx[d]] for d in range(ndim)]
            geometries[idx] = self.get_geometry(mode_indices, q_values)
        
        freqs_au = self.frequencies_au[mode_indices]
        
        return {
            'geometries': geometries,
            'q_grids': q_grids,
            'mode_indices': mode_indices,
            'frequencies_au': freqs_au,
            'frequencies_cm': freqs_au * nist.HARTREE2WAVENUMBER
        }
    
    def _build_comment_line(self, mode_indices, q_values):
        """Build standard comment line for XYZ file."""
        ndim = len(mode_indices)
        q_str = ", ".join([f"q{mode_indices[d]}={q_values[d]:.6f}" for d in range(ndim)])
        return f"{ndim}D scan: {q_str}"
    
    def _write_single_xyz(self, filename, geometry, comment):
        """Write a single XYZ file."""
        with open(filename, "w") as f:
            f.write(f"{self.natm}\n")
            f.write(f"{comment}\n")
            for a in range(self.natm):
                x, y, z = geometry[a]
                f.write(f"{self.symbols[a]}  {x:.10f}  {y:.10f}  {z:.10f}\n")
    
    def write_xyz(self, result, filename, separate=False, output_dir=None):
        """
        Write geometries to XYZ file(s).
        
        Args:
            result: dict returned by generate_geometries()
            filename: output filename (base name for separate files)
            separate: If False (default), write multi-frame XYZ file.
                      If True, write separate XYZ files with zero-padded numbering.
            output_dir: Directory for output files (only used when separate=True).
                        If None, uses current directory.
        
        Output format:
            Standard XYZ format:
            - Line 1: number of atoms
            - Line 2: comment (e.g., "2D scan: q0=1.000000, q1=-2.000000")
            - Lines 3+: atom coordinates
            
        Separate file naming:
            For n_total structures, files are named with zero-padded indices:
            - {filename}_001.xyz, {filename}_002.xyz, ... (for n_total >= 100)
            - {filename}_01.xyz, {filename}_02.xyz, ... (for n_total >= 10)
            - {filename}_1.xyz, {filename}_2.xyz, ... (for n_total < 10)
        """
        geometries = result['geometries']
        q_grids = result['q_grids']
        mode_indices = result['mode_indices']
        ndim = len(mode_indices)
        
        # Flatten geometry array for iteration
        original_shape = geometries.shape[:-2]  # e.g., (9,) or (9, 7)
        n_total = int(np.prod(original_shape))
        geom_flat = geometries.reshape(n_total, self.natm, 3)
        
        if not separate:
            # Multi-frame XYZ
            with open(filename, "w") as f:
                for i in range(n_total):
                    idx = np.unravel_index(i, original_shape)
                    q_values = [q_grids[d][idx[d]] for d in range(ndim)]
                    comment = self._build_comment_line(mode_indices, q_values)
                    
                    f.write(f"{self.natm}\n")
                    f.write(f"{comment}\n")
                    for a in range(self.natm):
                        x, y, z = geom_flat[i, a]
                        f.write(f"{self.symbols[a]}  {x:.10f}  {y:.10f}  {z:.10f}\n")
            
            print(f"Written {n_total} structures to {filename}")
        
        else:
            # Separate XYZ files with zero-padded numbering
            if output_dir is not None:
                os.makedirs(output_dir, exist_ok=True)
            
            # Determine zero-padding width
            num_digits = len(str(n_total))
            
            # Get base name without extension
            if filename.endswith(".xyz"):
                base_name = filename[:-4]
            else:
                base_name = filename
            
            for i in range(n_total):
                idx = np.unravel_index(i, original_shape)
                q_values = [q_grids[d][idx[d]] for d in range(ndim)]
                comment = self._build_comment_line(mode_indices, q_values)
                
                # Zero-padded index (1-based)
                padded_idx = str(i + 1).zfill(num_digits)
                
                if output_dir is not None:
                    out_file = os.path.join(output_dir, f"{base_name}_{padded_idx}.xyz")
                else:
                    out_file = f"{base_name}_{padded_idx}.xyz"
                
                self._write_single_xyz(out_file, geom_flat[i], comment)
            
            if output_dir is not None:
                print(f"Written {n_total} structures to {output_dir}/{base_name}_*.xyz")
            else:
                print(f"Written {n_total} structures: {base_name}_01.xyz ... {base_name}_{str(n_total).zfill(num_digits)}.xyz")
    
    def print_modes(self):
        """Print available modes and their frequencies."""
        print("=" * 50)
        print(" Available Normal Modes")
        print("=" * 50)
        for i in range(self.num_modes):
            freq_cm = self.frequencies_au[i] * nist.HARTREE2WAVENUMBER
            print(f" Mode {i:3d}:  {freq_cm:12.4f}  cm^-1")
        print("=" * 50)


# ============================================================
# Example usage
# ============================================================

if __name__ == "__main__":
    
    print("=" * 60)
    print(" Example 1: Direct frequency calculation (optimized geometry)")
    print("=" * 60)
    
    h2o_opt = [
        ['O', (0.000000,  0.000000,  0.117369)],
        ['H', (0.000000,  0.756950, -0.469476)],
        ['H', (0.000000, -0.756950, -0.469476)]
    ]
    
    analyzer = NormalModeAnalyzer(h2o_opt, basis="def2-SVP")
    analyzer.run(method="dft", xc="b3lyp", is_linear=False, verbose=3)
    analyzer.print_frequencies()
    analyzer.save("h2o_opt", fmt="npy")
    analyzer.save("h2o_opt", fmt="txt")
    analyzer.write_optimized_xyz("h2o_optimized.xyz", comment="Optimized H2O")
    
    print("\n" + "=" * 60)
    print(" Example 2: Optimization + frequency (arbitrary geometry)")
    print("=" * 60)
    
    # Slightly distorted water
    h2o_dist = [
        ['O', (0.000000,  0.000000,  0.100000)],
        ['H', (0.000000,  0.800000, -0.500000)],
        ['H', (0.000000, -0.800000, -0.500000)]
    ]
    
    analyzer2 = NormalModeAnalyzer(h2o_dist, basis="def2-SVP")
    analyzer2.optimize_and_run(method="dft", xc="b3lyp", is_linear=False, verbose=3)
    analyzer2.print_frequencies()
    analyzer2.write_optimized_xyz("h2o_from_distorted.xyz", comment="Optimized from distorted")
    
    print("\n" + "=" * 60)
    print(" Example 3: Create generator from npy files")
    print("=" * 60)
    
    generator = NormalCoordGenerator("h2o_opt")
    generator.print_modes()
    
    print("\n" + "=" * 60)
    print(" Example 4: Create generator from txt files + xyz")
    print("=" * 60)
    
    generator_txt = NormalCoordGenerator.from_files(
        freq_file="h2o_opt_freq.txt",
        modes_file="h2o_opt_modes.txt",
        xyz_file="h2o_optimized.xyz"
    )
    generator_txt.print_modes()
    
    print("\n" + "=" * 60)
    print(" Example 5: Create generator from arrays")
    print("=" * 60)
    
    # Load data manually and create generator
    freq = np.load("h2o_opt_freq.npy")
    modes = np.load("h2o_opt_modes.npy")
    coords = np.array([[0.0, 0.0, 0.117], [0.0, 0.757, -0.469], [0.0, -0.757, -0.469]])
    symbols = ['O', 'H', 'H']
    
    generator_arr = NormalCoordGenerator.from_arrays(freq, modes, coords, symbols)
    generator_arr.print_modes()
    
    print("\n" + "=" * 60)
    print(" Example 6: Generate geometries - multi-frame XYZ")
    print("=" * 60)
    
    result_1d = generator.generate_geometries(
        mode_indices=[0],
        q_ranges=[(-4.0, 4.0)],
        n_points=[9]
    )
    print(f"1D scan: shape = {result_1d['geometries'].shape}")
    generator.write_xyz(result_1d, "h2o_mode0_1d.xyz", separate=False)
    
    print("\n" + "=" * 60)
    print(" Example 7: Generate geometries - separate XYZ files")
    print("=" * 60)
    
    result_2d = generator.generate_geometries(
        mode_indices=[0, 1],
        q_ranges=[(-2.0, 2.0), (-2.0, 2.0)],
        n_points=[5, 5]
    )
    print(f"2D scan: shape = {result_2d['geometries'].shape}, total = 25 structures")
    generator.write_xyz(result_2d, "h2o_2d", separate=True, output_dir="h2o_2d_xyz")
    
    print("\n" + "=" * 60)
    print(" Example 8: CASSCF frequency (H2O molecule)")
    print("=" * 60)
    
    analyzer_cas = NormalModeAnalyzer(h2o_opt, basis="def2-SVP")
    analyzer_cas.run(
        method="casscf",
        ncas=4,       
        nelecas=4,    
        is_linear=False,
        verbose=3,
        log_file="h2o_casscf_freq.log"
    )
    analyzer_cas.print_frequencies()
    analyzer_cas.save("h2o_casscf", fmt="npy")