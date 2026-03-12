"""
Physics-based feature extraction for perovskite solar cells.

This module provides feature extraction based on semiconductor physics,
including band structure, carrier transport, and interface properties.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from scipy.optimize import curve_fit
from scipy.constants import e, k, h, c


@dataclass
class BandGapFeatures:
    """Features related to band gap properties."""
    direct_gap: float  # Direct band gap (eV)
    indirect_gap: Optional[float]  # Indirect band gap (eV)
    effective_mass_electron: float  # Electron effective mass (m_e)
    effective_mass_hole: float  # Hole effective mass (m_h)
    band_gap_temperature_coefficient: float  # dEg/dT (eV/K)


@dataclass
class CarrierTransportFeatures:
    """Features related to carrier transport."""
    electron_mobility: float  # cm²/V·s
    hole_mobility: float  # cm²/V·s
    diffusion_length_electron: float  # μm
    diffusion_length_hole: float  # μm
    lifetime_electron: float  # ns
    lifetime_hole: float  # ns
    diffusion_coefficient_electron: float  # cm²/s
    diffusion_coefficient_hole: float  # cm²/s


@dataclass
class InterfaceFeatures:
    """Features related to interface properties."""
    conduction_band_offset: float  # eV
    valence_band_offset: float  # eV
    defect_density_interface: float  # cm⁻²
    recombination_velocity: float  # cm/s
    interface_dipole: float  # Debye


@dataclass
class AgingFeatures:
    """Features from time-series aging analysis."""
    initial_efficiency: float  # %
    degradation_rate: float  # %/hour
    half_life: Optional[float]  # hours
    degradation_activation_energy: Optional[float]  # eV
    stability_index: float  # dimensionless


class PhysicsFeatureExtractor:
    """
    Extract physics-based features for perovskite solar cells.
    
    This class provides methods to extract semiconductor physics features
    including band structure, carrier transport, interface properties,
    and aging characteristics.
    
    Example:
        >>> extractor = PhysicsFeatureExtractor()
        >>> band_features = extractor.extract_band_gap_features(
        ...     absorption_wavelengths=wavelengths,
        ...     absorption_coefficients=coeffs
        ... )
    """
    
    def __init__(self, temperature: float = 300.0):
        """
        Initialize the physics feature extractor.
        
        Args:
            temperature: Operating temperature in Kelvin (default: 300K)
        """
        self.temperature = temperature
        self.k_B = k / e  # Boltzmann constant in eV/K
        
    def extract_band_gap_features(
        self,
        absorption_wavelengths: np.ndarray,
        absorption_coefficients: np.ndarray,
        effective_mass_electron: Optional[float] = None,
        effective_mass_hole: Optional[float] = None,
        temperature_coefficient: Optional[float] = None
    ) -> BandGapFeatures:
        """
        Extract band gap features from optical absorption data.
        
        Uses Tauc plot method for band gap determination.
        
        Args:
            absorption_wavelengths: Wavelength data in nm
            absorption_coefficients: Absorption coefficient data in cm⁻¹
            effective_mass_electron: Electron effective mass (default: 0.1 m_e)
            effective_mass_hole: Hole effective mass (default: 0.1 m_e)
            temperature_coefficient: dEg/dT (default: -0.0003 eV/K)
            
        Returns:
            BandGapFeatures object with extracted properties
        """
        # Convert wavelength to energy
        energies = 1240.0 / absorption_wavelengths  # eV
        
        # Tauc plot analysis for direct band gap
        # (αhν)² vs hν for direct transitions
        alpha_hv_squared = (absorption_coefficients * energies) ** 2
        
        # Linear fit to extract band gap
        try:
            # Find linear region (usually 0.5-2.0 above band gap)
            mask = (energies > 1.0) & (energies < 3.0) & (alpha_hv_squared > 0)
            coeffs = np.polyfit(energies[mask], alpha_hv_squared[mask], 1)
            direct_gap = -coeffs[1] / coeffs[0]
        except Exception:
            direct_gap = 1.55  # Default for MAPbI3
        
        # Estimate indirect gap (typically 0.1-0.3 eV lower)
        indirect_gap = direct_gap - 0.2 if direct_gap > 1.0 else None
        
        # Default effective masses
        if effective_mass_electron is None:
            effective_mass_electron = 0.1  # Typical for perovskites
        if effective_mass_hole is None:
            effective_mass_hole = 0.1
        
        # Default temperature coefficient
        if temperature_coefficient is None:
            temperature_coefficient = -0.0003  # Typical for perovskites
        
        return BandGapFeatures(
            direct_gap=direct_gap,
            indirect_gap=indirect_gap,
            effective_mass_electron=effective_mass_electron,
            effective_mass_hole=effective_mass_hole,
            band_gap_temperature_coefficient=temperature_coefficient
        )
    
    def extract_carrier_transport_features(
        self,
        mobility_electron: Optional[float] = None,
        mobility_hole: Optional[float] = None,
        lifetime_electron: Optional[float] = None,
        lifetime_hole: Optional[float] = None,
        temperature: Optional[float] = None
    ) -> CarrierTransportFeatures:
        """
        Extract carrier transport features.
        
        Uses Einstein relation to relate mobility and diffusion coefficient.
        
        Args:
            mobility_electron: Electron mobility in cm²/V·s
            mobility_hole: Hole mobility in cm²/V·s
            lifetime_electron: Electron lifetime in ns
            lifetime_hole: Hole lifetime in ns
            temperature: Temperature in K (uses instance default if not provided)
            
        Returns:
            CarrierTransportFeatures object
        """
        temp = temperature if temperature is not None else self.temperature
        
        # Default values for perovskites
        if mobility_electron is None:
            mobility_electron = 100.0  # cm²/V·s
        if mobility_hole is None:
            mobility_hole = 100.0
        if lifetime_electron is None:
            lifetime_electron = 1000.0  # ns
        if lifetime_hole is None:
            lifetime_hole = 1000.0
        
        # Einstein relation: D = μkT/q
        # D in cm²/s, μ in cm²/V·s
        thermal_voltage = self.k_B * temp  # V
        diffusion_coeff_electron = mobility_electron * thermal_voltage
        diffusion_coeff_hole = mobility_hole * thermal_voltage
        
        # Diffusion length: L = sqrt(D*τ)
        # τ in ns -> s, L in μm
        tau_e_s = lifetime_electron * 1e-9
        tau_h_s = lifetime_hole * 1e-9
        
        diffusion_length_electron = np.sqrt(diffusion_coeff_electron * tau_e_s) * 1e4  # cm to μm
        diffusion_length_hole = np.sqrt(diffusion_coeff_hole * tau_h_s) * 1e4
        
        return CarrierTransportFeatures(
            electron_mobility=mobility_electron,
            hole_mobility=mobility_hole,
            diffusion_length_electron=diffusion_length_electron,
            diffusion_length_hole=diffusion_length_hole,
            lifetime_electron=lifetime_electron,
            lifetime_hole=lifetime_hole,
            diffusion_coefficient_electron=diffusion_coeff_electron,
            diffusion_coefficient_hole=diffusion_coeff_hole
        )
    
    def extract_interface_features(
        self,
        perovskite_work_function: float,
        transport_layer_work_function: float,
        perovskite_ea: float,  # Electron affinity
        perovskite_ip: float,  # Ionization potential
        tl_ea: float,
        tl_ip: float,
        defect_density: Optional[float] = None
    ) -> InterfaceFeatures:
        """
        Extract interface features between perovskite and transport layer.
        
        Args:
            perovskite_work_function: Work function of perovskite (eV)
            transport_layer_work_function: Work function of TL (eV)
            perovskite_ea: Electron affinity of perovskite (eV)
            perovskite_ip: Ionization potential of perovskite (eV)
            tl_ea: Electron affinity of transport layer (eV)
            tl_ip: Ionization potential of transport layer (eV)
            defect_density: Interface defect density (cm⁻²)
            
        Returns:
            InterfaceFeatures object
        """
        # Band offsets
        conduction_band_offset = perovskite_ea - tl_ea
        valence_band_offset = perovskite_ip - tl_ip
        
        # Default defect density
        if defect_density is None:
            defect_density = 1e10  # cm⁻²
        
        # Recombination velocity (simplified model)
        # S = σ * v_th * N_def
        sigma = 1e-15  # Capture cross-section (cm²)
        v_th = 1e7  # Thermal velocity (cm/s)
        recombination_velocity = sigma * v_th * defect_density
        
        # Interface dipole
        interface_dipole = abs(perovskite_work_function - transport_layer_work_function) * 10  # Debye
        
        return InterfaceFeatures(
            conduction_band_offset=conduction_band_offset,
            valence_band_offset=valence_band_offset,
            defect_density_interface=defect_density,
            recombination_velocity=recombination_velocity,
            interface_dipole=interface_dipole
        )
    
    def analyze_aging_curves(
        self,
        time_points: np.ndarray,  # hours
        efficiency_values: np.ndarray,  # %
        temperature: Optional[float] = None
    ) -> AgingFeatures:
        """
        Analyze aging/degradation curves to extract stability features.
        
        Args:
            time_points: Time points in hours
            efficiency_values: PCE values at each time point
            temperature: Measurement temperature (uses instance default if not provided)
            
        Returns:
            AgingFeatures object with degradation parameters
        """
        temp = temperature if temperature is not None else self.temperature
        
        # Initial efficiency
        initial_efficiency = efficiency_values[0]
        
        # Fit exponential decay: η(t) = η_0 * exp(-kt)
        def decay_model(t, eta0, k):
            return eta0 * np.exp(-k * t)
        
        try:
            popt, _ = curve_fit(
                decay_model,
                time_points,
                efficiency_values,
                p0=[initial_efficiency, 1e-4],
                maxfev=5000
            )
            degradation_rate = popt[1] * 100  # Convert to %/hour
            
            # Half-life calculation
            if degradation_rate > 0:
                half_life = np.log(2) / (degradation_rate / 100)
            else:
                half_life = None
        except Exception:
            # Linear fit as fallback
            coeffs = np.polyfit(time_points, efficiency_values, 1)
            degradation_rate = -coeffs[0]  # %/hour
            half_life = initial_efficiency / (2 * abs(degradation_rate)) if degradation_rate > 0 else None
        
        # Stability index (ratio of final to initial efficiency)
        stability_index = efficiency_values[-1] / initial_efficiency if initial_efficiency > 0 else 0.0
        
        # Activation energy (placeholder - requires multi-temperature data)
        degradation_activation_energy = None
        
        return AgingFeatures(
            initial_efficiency=initial_efficiency,
            degradation_rate=degradation_rate,
            half_life=half_life,
            degradation_activation_energy=degradation_activation_energy,
            stability_index=stability_index
        )
    
    def calculate_all_features(
        self,
        absorption_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        mobility_data: Optional[Dict[str, float]] = None,
        interface_data: Optional[Dict[str, float]] = None,
        aging_data: Optional[Tuple[np.ndarray, np.ndarray]] = None
    ) -> Dict[str, Union[BandGapFeatures, CarrierTransportFeatures, InterfaceFeatures, AgingFeatures]]:
        """
        Calculate all physics features from provided data.
        
        Args:
            absorption_data: Tuple of (wavelengths, absorption_coefficients)
            mobility_data: Dict with mobility and lifetime data
            interface_data: Dict with interface parameters
            aging_data: Tuple of (time_points, efficiency_values)
            
        Returns:
            Dictionary of all extracted features
        """
        features = {}
        
        if absorption_data is not None:
            features['band_gap'] = self.extract_band_gap_features(*absorption_data)
        
        if mobility_data is not None:
            features['carrier_transport'] = self.extract_carrier_transport_features(**mobility_data)
        
        if interface_data is not None:
            features['interface'] = self.extract_interface_features(**interface_data)
        
        if aging_data is not None:
            features['aging'] = self.analyze_aging_curves(*aging_data)
        
        return features
