#!/usr/bin/env python
"""Tools for fitting 3rd order Birch-Murnaghan EOS"""

import re
import numpy as np
from scipy.optimize import curve_fit, brentq
import matplotlib.pyplot as plt
from matplotlib import ticker, cm
from typing import List, Tuple, Optional, Callable
from dataclasses import dataclass, field
import logging
import argparse
from joblib import Parallel, delayed
from matplotlib.colors import Normalize



# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Set matplotlib parameters
plt.rcParams['font.family'] = 'Arial'


# Precompile regular expressions
@dataclass
class RegexPatterns:
    a_re: re.Pattern = field(default_factory=lambda: re.compile(r'\ba\s*=\s*([-\d.]+)'))
    b_re: re.Pattern = field(default_factory=lambda: re.compile(r'\bb\s*=\s*([-\d.]+)'))
    c_re: re.Pattern = field(default_factory=lambda: re.compile(r'\bc\s*=\s*([-\d.]+)'))
    vol_re: re.Pattern = field(default_factory=lambda: re.compile(r'Current cell volume =\s+(\d+\.\d+)\s+A\*\*3'))
    zpe_re: re.Pattern = field(default_factory=lambda: re.compile(r'Zero-point energy =\s+(\d+\.\d+)\s+eV'))
    tmo_re: re.Pattern = field(default_factory=lambda: re.compile(
        r'(\d+\.\d+)\s+(\d+\.\d+)\s+([+-]?\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)'))
    ufb_re: re.Pattern = field(default_factory=lambda: re.compile(
        r'Total energy corrected for finite basis set =\s+([+-]?\d+\.\d+)\s+eV'))
    enth_re: re.Pattern = field(default_factory=lambda: re.compile(
        r'LBFGS: finished iteration\s+\d+ with enthalpy=\s+([+-]?\d+\.\d+E[+-]?\d+)\seV'))
    p_re: re.Pattern = field(default_factory=lambda: re.compile(r'\s+Pressure:\s+([+-]?\d+\.\d+)\s+\*'))


# Instantiate regex patterns
regex = RegexPatterns()


@dataclass
class BM3EOSParameters:
    V0: float
    E0: float
    K0: float
    Kp0: float
    V0_err: float
    E0_err: float
    K0_err: float
    Kp0_err: float


def bm3_eos_energy(V: np.ndarray, V0: float, E0: float, K0: float, Kp0: float) -> np.ndarray:
    """Calculate the energy from a 3rd order BM EOS"""
    eta = (V0 / V) ** (2.0 / 3.0) - 1.0
    E = E0 + (9.0 * V0 * K0 / 16.0) * (
            (eta ** 3.0) * Kp0 + (eta ** 2.0) * (6.0 - 4.0 * (V0 / V) ** (2.0 / 3.0))
    )
    return E


def bm3_eos_pressure(V: np.ndarray, V0: float, K0: float, Kp0: float) -> np.ndarray:
    """Calculate the pressure from a 3rd order BM EOS"""
    eta = (V0 / V) ** (2.0 / 3.0) - 1.0
    term1 = (V0 / V) ** (7.0 / 3.0) - (V0 / V) ** (5.0 / 3.0)
    term2 = 1.0 + (3.0 / 4.0) * (Kp0 - 4.0) * eta
    P = (3.0 * K0 / 2.0) * term1 * term2
    return P


def fit_eos(
        V: np.ndarray,
        F: np.ndarray,
        initial_guesses: Optional[List[float]] = None,
        verbose: bool = False
) -> BM3EOSParameters:
    """Fit parameters of a 3rd order BM EOS using energy data"""
    if initial_guesses is None:
        initial_guesses = [
            np.mean(V),  # V0_guess
            np.mean(F),  # E0_guess
            240.0 / 160.218,  # K0_guess in eV.A**3
            4.0  # Kp0_guess
        ]

    try:
        popt, pcov = curve_fit(
            bm3_eos_energy,
            V,
            F,
            p0=initial_guesses,
            maxfev=10000
        )
    except RuntimeError as e:
        logging.error(f"Curve fitting failed: {e}")
        raise

    perr = np.sqrt(np.diag(pcov))
    parameters = BM3EOSParameters(
        V0=popt[0],
        E0=popt[1],
        K0=popt[2],
        Kp0=popt[3],
        V0_err=perr[0],
        E0_err=perr[1],
        K0_err=perr[2],
        Kp0_err=perr[3]
    )

    if verbose:
        logging.info("Fitted 3rd order Birch-Murnaghan EOS parameters:")
        logging.info(f" E0  = {parameters.E0:7g} eV ± {parameters.E0_err:7g} eV")
        logging.info(f" V0  = {parameters.V0:7g} Å³ ± {parameters.V0_err:7g} Å³")
        logging.info(f" K0  = {parameters.K0 * 160.218:7g} GPa ± {parameters.K0_err * 160.218:7g} GPa")
        logging.info(f" K0' = {parameters.Kp0:7g} ± {parameters.Kp0_err:7g}")

    return parameters


def fit_pressure_eos(
        P: np.ndarray,
        V: np.ndarray,
        initial_guesses: Optional[List[float]] = None,
        verbose: bool = False
) -> BM3EOSParameters:
    """Fit parameters of a 3rd order BM EOS using pressure data"""
    if initial_guesses is None:
        initial_guesses = [
            np.mean(V),  # V0_guess
            240.0 / 160.218,  # K0_guess in eV.A**3
            4.0  # Kp0_guess
        ]

    try:
        popt, pcov = curve_fit(
            bm3_eos_pressure,
            V,
            P / 160.218,  # Convert GPa to eV.A**3
            p0=initial_guesses,
            maxfev=10000
        )
    except RuntimeError as e:
        logging.error(f"Pressure EOS fitting failed: {e}")
        raise

    perr = np.sqrt(np.diag(pcov))
    parameters = BM3EOSParameters(
        V0=popt[0],
        E0=0.0,  # Not used in pressure fitting
        K0=popt[1],
        Kp0=popt[2],
        V0_err=perr[0],
        E0_err=0.0,
        K0_err=perr[1],
        Kp0_err=perr[2]
    )

    if verbose:
        logging.info("Fitted 3rd order Birch-Murnaghan EOS parameters (Pressure Fit):")
        logging.info(f" V0  = {parameters.V0:7g} Å³ ± {parameters.V0_err:7g} Å³")
        logging.info(f" K0  = {parameters.K0 * 160.218:7g} GPa ± {parameters.K0_err * 160.218:7g} GPa")
        logging.info(f" K0' = {parameters.Kp0:7g} ± {parameters.Kp0_err:7g}")

    return parameters


def quintic_function(x: np.ndarray, a: float, b: float, c: float, d: float, e: float, f: float) -> np.ndarray:
    """Quintic polynomial function"""
    return a * x ** 5 + b * x ** 4 + c * x ** 3 + d * x ** 2 + e * x + f


def format_latex(value: float, prec: int = 2, noplus: bool = False) -> str:
    """Format a float into LaTeX scientific notation"""
    fmt = f'{{:{"+" if not noplus else ""}.{prec}e}}'
    base, exponent = fmt.format(value).split('e')
    exponent = exponent.lstrip('+').lstrip('0').replace('-0', '-', 1)
    return f"{base}\\times 10^{{{exponent}}}"


def fit_parameters_quintic(
        Ts: np.ndarray,
        V0s: np.ndarray,
        E0s: np.ndarray,
        K0s: np.ndarray,
        Kp0s: np.ndarray,
        plot: bool = False,
        filename: Optional[str] = None,
        table: Optional[str] = None
) -> Tuple[Callable[[float], float], Callable[[float], float],
Callable[[float], float], Callable[[float], float]]:
    """Fit quintic polynomials to BM3 EOS parameters as functions of temperature"""
    params = {}
    fitted_funcs = {}

    for param_name, data in zip(['V0', 'E0', 'K0', 'Kp0'], [V0s, E0s, K0s, Kp0s]):
        popt, _ = curve_fit(quintic_function, Ts, data, maxfev=10000)
        fitted_funcs[param_name] = lambda t, popt=popt: quintic_function(t, *popt)

    if table:
        with open(table, 'w') as fh:
            for param in ['E0', 'V0', 'K0', 'Kp0']:
                popt = None
                if param == 'E0':
                    popt = curve_fit(quintic_function, Ts, E0s)[0]
                elif param == 'V0':
                    popt = curve_fit(quintic_function, Ts, V0s)[0]
                elif param == 'K0':
                    popt = curve_fit(quintic_function, Ts, K0s)[0]
                elif param == 'Kp0':
                    popt = curve_fit(quintic_function, Ts, Kp0s)[0]

                expr = ' + '.join([f"{format_latex(coef, noplus=True)}T^{5 - i}" for i, coef in enumerate(popt[:5])])
                expr += f" + {format_latex(popt[5], noplus=True)}"
                fh.write(f"${param}(T) = {expr}$\n")

    if plot:
        plt.figure(figsize=(14, 10))
        fTs = np.linspace(np.min(Ts), np.max(Ts), 300)

        plt.subplot(2, 2, 1)
        plt.scatter(Ts, V0s, color='black', label='Data')
        plt.plot(fTs, fitted_funcs['V0'](fTs), 'k-', label='Fit')
        plt.xlabel('Temperature (K)')
        plt.ylabel('V$_0$ (Å³)')
        # plt.legend()

        plt.subplot(2, 2, 2)
        plt.scatter(Ts, E0s, color='black', label='Data')
        plt.plot(fTs, fitted_funcs['E0'](fTs), 'k-', label='Fit')
        plt.xlabel('Temperature (K)')
        plt.ylabel('E$_0$ (eV)')
        # plt.legend()

        plt.subplot(2, 2, 3)
        plt.scatter(Ts, K0s * 160.218, color='black', label='Data')  # Convert to GPa
        plt.plot(fTs, fitted_funcs['K0'](fTs) * 160.218, 'k-', label='Fit')
        plt.xlabel('Temperature (K)')
        plt.ylabel('K$_0$ (GPa)')
        # plt.legend()

        plt.subplot(2, 2, 4)
        plt.scatter(Ts, Kp0s, color='black', label='Data')
        plt.plot(fTs, fitted_funcs['Kp0'](fTs), 'k-', label='Fit')
        plt.xlabel('Temperature (K)')
        plt.ylabel(r"K$^{\prime}_0$")
        # plt.legend()

        plt.tight_layout()
        if filename:
            plt.savefig(filename, dpi=300)
            plt.close()
        else:
            plt.show()

    return fitted_funcs['V0'], fitted_funcs['E0'], fitted_funcs['K0'], fitted_funcs['Kp0']


def parse_castep_file(
        filename: str,
        verbose: bool = False
) -> Tuple[List[Tuple], List[float], Optional[float], Optional[float], Optional[float]]:
    """
    Parse a Castep output file and extract thermodynamic data.

    Returns:
        data: List of tuples containing (V, U, zpe, T, E, F, S, Cv, P, a, b, c)
        ts: List of temperatures extracted
        a, b, c: Lattice parameters from the last parsed data block
    """
    data = []
    ts = []
    a, b, c = None, None, None
    current_volume = None
    U, H, zpe, P = None, None, None, None
    in_thermo = False
    skip_lines = 0

    with open(filename, 'r') as fh:
        for line in fh:
            if skip_lines > 0:
                skip_lines -= 1
                continue

            # Volume
            vol_match = regex.vol_re.search(line)
            if vol_match:
                current_volume = float(vol_match.group(1))
                if verbose:
                    logging.debug(f"Volume: {current_volume} Å³")
                continue

            # Lattice parameters
            for param, regex_re in zip(['a', 'b', 'c'], [regex.a_re, regex.b_re, regex.c_re]):
                match = regex_re.search(line)
                if match:
                    value = float(match.group(1))
                    if param == 'a':
                        a = value
                    elif param == 'b':
                        b = value
                    elif param == 'c':
                        c = value
                    if verbose:
                        logging.debug(f"{param}: {value} Å")
                    break
            else:
                # Internal energy
                match = regex.ufb_re.search(line)
                if match:
                    U = float(match.group(1))
                    if verbose:
                        logging.debug(f"Internal Energy (U): {U} eV")
                    continue

                # Enthalpy
                match = regex.enth_re.search(line)
                if match:
                    H = float(match.group(1))
                    if verbose:
                        logging.debug(f"Enthalpy (H): {H} eV")
                    continue

                # Pressure
                match = regex.p_re.search(line)
                if match:
                    P = float(match.group(1))
                    if verbose:
                        logging.debug(f"Pressure (P): {P} GPa")
                    continue

                # Zero-point energy
                match = regex.zpe_re.search(line)
                if match:
                    zpe = float(match.group(1))
                    if verbose:
                        logging.debug(f"Zero-point Energy (ZPE): {zpe} eV")
                    in_thermo = True
                    skip_lines = 3  # Skip next three lines
                    continue

                # Thermodynamic data
                if in_thermo:
                    tmo_match = regex.tmo_re.search(line)
                    if tmo_match:
                        T = float(tmo_match.group(1))
                        E = float(tmo_match.group(2))
                        F = float(tmo_match.group(3))
                        S = float(tmo_match.group(4))
                        Cv = float(tmo_match.group(5))

                        # Calculate U from H and P
                        if H is not None and P is not None and current_volume is not None:
                            U_calc = H - (current_volume * P / 160.21766208)
                        else:
                            U_calc = U  # Fallback

                        if verbose:
                            logging.debug(f"T: {T} K, U: {U_calc} eV, F: {F} eV")

                        # Apply temperature filter
                        if T < 2500.0 and T != 0.0:
                            data.append((current_volume, U_calc, zpe, T, E, F, S, Cv, P, a, b, c))
                            ts.append(T)
                        continue
                    else:
                        # End of thermo block
                        in_thermo = False
                        zpe = None
                        U = None
                        current_volume = None
                        a, b, c = None, None, None

    return data, ts, a, b, c


def get_VF(
        data_table: List[Tuple],
        T: float
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Extract Volume and Free Energy for a given temperature from the data table.

    Args:
        data_table: Parsed data from Castep files.
        T: Temperature in K or 'static'.

    Returns:
        V: Array of volumes (Å³)
        F: Array of Helmholtz free energies (eV)
        P: Array of pressures (GPa) if T == 'static', else None
    """
    if T == 'static':
        mode = 'static'
    elif T == 0.0:
        mode = 'zpe'
    else:
        mode = 'f'

    V, F, P = [], [], []
    for entry in data_table:
        entry_T = entry[3]
        if (entry_T == T) or (mode != 'f' and entry_T == data_table[0][3]):
            V.append(entry[0])
            if mode == 'static':
                F.append(entry[1])  # U(V)
                P.append(entry[8])
            elif mode == 'zpe':
                F.append(entry[1] + entry[2])  # U(V) + ZPE(V)
            else:
                F.append(entry[5] + entry[1])  # F = U + F_vib
    V = np.array(V)
    F = np.array(F)
    if mode == 'static':
        P = np.array(P)
        return V, F, P
    return V, F, None


def get_volume(
        P: float,
        T: float,
        fV0: Callable[[float], float],
        fK0: Callable[[float], float],
        fKp0: Callable[[float], float],
        max_expansion: int = 10
) -> float:
    """
    Calculate the volume for a given pressure and temperature using BM3 EOS.

    Args:
        P: Pressure in GPa.
        T: Temperature in K.
        fV0, fK0, fKp0: Functions returning V0, K0, Kp0 as functions of T.
        max_expansion: Maximum number of expansion attempts.

    Returns:
        Volume in Å³.
    """
    P_eVA3 = P / 160.218  # Convert GPa to eV/Å³
    V0 = fV0(T)
    K0 = fK0(T)
    Kp0 = fKp0(T)

    def pressure_diff(V):
        return bm3_eos_pressure(V, V0, K0, Kp0) - P_eVA3

    a, b = 0.8 * V0, 1.2 * V0
    attempts = max_expansion

    while attempts > 0:
        try:
            return brentq(pressure_diff, a, b)
        except ValueError:
            a *= 0.9
            b *= 1.1
            attempts -= 1
            logging.warning(f"Adjusting search interval for volume. Attempts left: {attempts}")

    raise ValueError("Could not find root after expanding the interval")


def plot_eos_energy(
        V: np.ndarray,
        F: np.ndarray,
        parameters: BM3EOSParameters,
        T: float,
        ax: Optional[plt.Axes] = None
):
    """Plot energy vs volume with BM3 EOS fit"""
    if ax is None:
        fig, ax = plt.subplots()

    ax.scatter(V, F, color='k', marker='o', label=f'Data {T} K')
    V_fine = np.linspace(np.min(V), np.max(V), 100)
    F_fit = bm3_eos_energy(V_fine, parameters.V0, parameters.E0, parameters.K0, parameters.Kp0)
    ax.plot(V_fine, F_fit, 'k--') #, label=f'BM3 Fit {T} K')
    ax.set_xlabel('Volume (Å³)')
    ax.set_ylabel('Helmholtz Free Energy (eV)')
    ax.legend(ncol=3, bbox_to_anchor=(0., 0.96, 1., .102), loc=3,
              mode="expand", borderaxespad=0., numpoints=1)


def plot_eos_pressure(
        V_fine: np.ndarray,
        P_fit: np.ndarray,
        parameters: BM3EOSParameters,
        T: float,
        ax: Optional[plt.Axes] = None
):
    """Plot pressure vs volume with BM3 EOS fit"""
    if ax is None:
        fig, ax = plt.subplots()

    ax.plot(P_fit, V_fine, 'k-') #, label=f'BM3 Fit {T} K')
    ax.set_xlabel('Pressure (GPa)')
    ax.set_ylabel('Volume (Å³)')
    ax.legend()


def plot_twoplots(
        min_V: float,
        max_V: float,
        vs: List[np.ndarray],
        fs: List[np.ndarray],
        eos_params: List[BM3EOSParameters],
        Ts: List[float],
        filename: Optional[str] = None
):
    """Create stacked energy and pressure plots with consistent legends"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 12))

    # Create a colormap
    cmap = cm.inferno # other options: 'plasma', 'inferno', 'magma', 'cividis', 'jet', 'viridis',
    norm = Normalize(vmin=min(Ts), vmax=max(Ts))

    # Energy Plot
    for V, F, params, T in zip(vs, fs, eos_params, Ts):
        color = cmap(norm(T))
        V_fine = np.linspace(np.min(V), np.max(V), 100)
        F_fit = bm3_eos_energy(V_fine, params.V0, params.E0, params.K0, params.Kp0)
        ax1.plot(V_fine, F_fit, color=color, linestyle='--') #, label=f'BM3 Fit {T} K')
        ax1.scatter(V, F, color=color, marker='o', label=f'{T} K')
    ax1.set_xlabel('Volume (Å³)')
    ax1.set_ylabel('Helmholtz Free Energy (eV)')
    ax1.legend(ncol=3, bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
              mode="expand", borderaxespad=1.02, numpoints=1)
    ax1.set_title('Helmholtz Free Energy vs Volume')

    # Pressure Plot
    for params, T in zip(eos_params, Ts):
        if T == 'static' or T == 0.0:
            continue  # Skip static and 0K for pressure plot
        color = cmap(norm(T))
        V_fine = np.linspace(min_V, max_V, 100)
        P_fit = bm3_eos_pressure(V_fine, params.V0, params.K0, params.Kp0) * 160.218  # GPa
        ax2.plot(P_fit, V_fine, color=color, linestyle='-', label=f'{T} K')
    ax2.set_xlabel('Pressure (GPa)')
    ax2.set_ylabel('Volume (Å³)')
    ax2.legend(loc='best', fontsize=8)
    ax2.set_title('Pressure vs Volume')

    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=900)
        plt.close()
    else:
        plt.show()


def main():
    """Main function to execute the EOS fitting and plotting"""
    parser = argparse.ArgumentParser(
        description="Fit a set of EVT data to isothermal 3rd order BM EOS",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('datafiles', nargs='+', help='One file per volume to fit')
    parser.add_argument('--plot_pv', default=None, help='Filename to save PV plot')
    parser.add_argument('--plot_ev', default=None, help='Filename to save EV plot (Not Implemented)')
    parser.add_argument('--plot_both', default=None, help='Filename to save stacked PV and EV plots')
    parser.add_argument('--polyplot', default=None, help='Filename to save polynomial fits plot')
    parser.add_argument('--latex_table', default=None, help='Filename to save LaTeX table of fitting parameters')
    parser.add_argument('--max_t', default=2500.0, type=float, help='Maximum temperature to evaluate results (K)')
    parser.add_argument('--min_t', default=300.0, type=float, help='Minimum temperature to evaluate results (K)')
    parser.add_argument('--step_t', default=105.0, type=float, help='Temperature step to evaluate results (K)')
    parser.add_argument('--max_p', default=101.0, type=float, help='Maximum pressure to evaluate results (GPa)')
    parser.add_argument('--min_p', default=0.0, type=float, help='Minimum pressure to evaluate results (GPa)')
    parser.add_argument('--step_p', default=1.0, type=float, help='Pressure step to evaluate results (GPa)')
    parser.add_argument('--plot_temps', nargs='+', type=float, help='List of temperatures to plot')
    parser.add_argument('--do_pv', action='store_true',
                        help='Fit the volume-pressure data and report (static) EOS parameters')
    parser.add_argument('--use_pv', action='store_true',
                        help='Use volume-pressure EOS parameters as initial guess for FV fits')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')

    args = parser.parse_args()

    # Parse data files
    data = []
    a_all, b_all, c_all = [], [], []
    for file in args.datafiles:
        logging.info(f"Reading data from: {file}")
        parsed_data, ts, a, b, c = parse_castep_file(file, verbose=args.verbose)
        data.extend(parsed_data)
        a_all.append(a)
        b_all.append(b)
        c_all.append(c)

    if not data:
        logging.error("No data parsed from the provided files.")
        return

    # Fit EOS parameters for static case
    logging.info("Processing static case")
    V_static, F_static, P_static = get_VF(data, 'static')
    if args.do_pv or args.use_pv:
        logging.info("Fitting to static PV data")
        eos_pv = fit_pressure_eos(P_static, V_static, verbose=args.verbose)
        v0_guess, k0_guess, kp0_guess = eos_pv.V0, eos_pv.K0, eos_pv.Kp0
    else:
        v0_guess = k0_guess = kp0_guess = None

    # Fit BM3 EOS for static case
    eos_static = fit_eos(
        V_static,
        F_static,
        initial_guesses=[v0_guess, F_static.mean(), k0_guess, kp0_guess] if args.use_pv else None,
        verbose=args.verbose
    )

    # Fit EOS parameters for all temperatures (including 0K)
    unique_ts = sorted(set([entry[3] for entry in data if entry[3] != 0.0]))
    unique_ts = [0.0] + unique_ts  # Include 0K
    eos_params = []
    vs, fs = [], []

    # Parallel fitting for efficiency
    def fit_at_temperature(T):
        if T == 'static':
            V, F, P = V_static, F_static, P_static
        elif T == 0.0:
            V, F, _ = get_VF(data, 0.0)
        else:
            V, F, _ = get_VF(data, T)

        if args.use_pv and T == 'static':
            params = eos_static
        else:
            params = fit_eos(V, F, verbose=args.verbose)
        return (params, V, F)

    results = Parallel(n_jobs=-1)(
        delayed(fit_at_temperature)(T) for T in unique_ts
    )

    for params, V, F in results:
        eos_params.append(params)
        vs.append(V)
        fs.append(F)

    # Determine volume range for plotting
    all_volumes = np.concatenate(vs)
    min_V, max_V = np.min(all_volumes), np.max(all_volumes)

    # Plotting
    if args.plot_both:
        logging.info(f"Creating stacked PV and EV plots: {args.plot_both}")
        plot_twoplots(min_V, max_V, vs, fs, eos_params, unique_ts, filename=args.plot_both)

    if args.plot_pv:
        logging.info(f"Creating PV plot: {args.plot_pv}")
        plt.figure(figsize=(8, 6))
        for params, T in zip(eos_params, unique_ts):
            if T == 'static' or T == 0.0:
                continue  # Skip static and 0K for pressure plot
            V_fine = np.linspace(min_V, max_V, 100)
            P_fit = bm3_eos_pressure(V_fine, params.V0, params.K0, params.Kp0) * 160.218  # GPa
            plt.plot(P_fit, V_fine, '-', label=f'{T} K')
        plt.xlabel('Pressure (GPa)')
        plt.ylabel('Volume (Å³)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(args.plot_pv, dpi=900)
        plt.close()

    if args.polyplot or args.latex_table:
        # Extract parameter arrays
        Ts = np.array(unique_ts)
        V0s = np.array([p.V0 for p in eos_params])
        E0s = np.array([p.E0 for p in eos_params])
        K0s = np.array([p.K0 for p in eos_params])
        Kp0s = np.array([p.Kp0 for p in eos_params])

        # Fit quintic polynomials
        fv0, fe0, fk0, fkp0 = fit_parameters_quintic(
            Ts,
            V0s,
            E0s,
            K0s,
            Kp0s,
            plot=bool(args.polyplot),
            filename=args.polyplot,
            table=args.latex_table
        )

    # Evaluate and print results over pressure and temperature ranges
    if args.polyplot or args.latex_table:
        logging.info("Evaluating volumes over P-T grid")
        for P in np.arange(args.min_p, args.max_p + args.step_p, args.step_p):
            for T in np.arange(args.min_t, args.max_t + args.step_t, args.step_t):
                try:
                    V = get_volume(P, T, fv0, fk0, fkp0)
                    print(f"{P:.2f} GPa, {T:.2f} K, {V:.4f} Å³")
                except ValueError as e:
                    logging.warning(f"Could not determine volume for P={P} GPa, T={T} K: {e}")


if __name__ == "__main__":
    main()
