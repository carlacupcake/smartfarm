# stochastic_weather.py
"""
Stochastic weather generation from historical data.

Generates synthetic weather years with tunable extremity levels,
from normal variability to extreme drought/heat events.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class WeatherScenario:
    """Configuration for a single weather scenario."""
    name: str
    precip_scale: float = 1.0       # <1 = drought, >1 = wet
    temp_offset: float = 0.0        # +/- degrees C shift
    radiation_scale: float = 1.0    # radiation multiplier
    noise_std: float = 0.05         # relative noise level (fraction of std)
    drought_periods: Optional[List[Tuple[int, int, float]]] = None  # [(start_hour, duration, intensity)]
    heatwave_periods: Optional[List[Tuple[int, int, float]]] = None  # [(start_hour, duration, temp_add)]
    cold_snap_periods: Optional[List[Tuple[int, int, float]]] = None  # [(start_hour, duration, temp_drop)]

    def extremity_index(self) -> float:
        """Compute a scalar metric quantifying deviation from normal conditions."""
        # Base extremity from global shifts
        precip_ext = abs(1.0 - self.precip_scale) * 2
        temp_ext = abs(self.temp_offset) / 5.0

        # Event extremity
        event_ext = 0.0
        if self.drought_periods:
            for _, duration, intensity in self.drought_periods:
                event_ext += (duration / 500) * intensity
        if self.heatwave_periods:
            for _, duration, temp_add in self.heatwave_periods:
                event_ext += (duration / 200) * (temp_add / 5)
        if self.cold_snap_periods:
            for _, duration, temp_drop in self.cold_snap_periods:
                event_ext += (duration / 200) * (temp_drop / 5)

        return precip_ext + temp_ext + event_ext


class StochasticWeatherGenerator:
    """
    Generate synthetic weather scenarios from historical data.

    Supports:
    - Global scaling (drought/wet years)
    - Temperature offsets (warm/cool years)
    - Temporally-correlated noise
    - Discrete extreme events (droughts, heat waves, cold snaps)
    """

    def __init__(
        self,
        historical_df: pd.DataFrame,
        precip_col: str = 'Hourly Precipitation (in)',
        temp_col: str = 'Temperature (C)',
        radiation_col: str = 'Hourly Radiation (W/m2)',
        seed: Optional[int] = None
    ):
        """
        Initialize generator with historical data.

        Args:
            historical_df: DataFrame with hourly weather data
            precip_col: Column name for precipitation
            temp_col: Column name for temperature
            radiation_col: Column name for solar radiation
            seed: Random seed for reproducibility
        """
        self.historical = historical_df.copy()
        self.precip_col = precip_col
        self.temp_col = temp_col
        self.radiation_col = radiation_col
        self.rng = np.random.default_rng(seed)

        # Store statistics for noise generation
        self.temp_std = self.historical[temp_col].std()
        self.radiation_std = self.historical[radiation_col].std()
        self.precip_mean = self.historical[precip_col].mean()

    def generate(self, scenario: WeatherScenario) -> pd.DataFrame:
        """
        Generate a synthetic weather year from a scenario configuration.

        Args:
            scenario: WeatherScenario configuration

        Returns:
            DataFrame with synthetic weather data
        """
        df = self.historical.copy()
        n = len(df)

        # 1. Global scaling
        df[self.precip_col] = df[self.precip_col] * scenario.precip_scale
        df[self.temp_col] = df[self.temp_col] + scenario.temp_offset
        df[self.radiation_col] = df[self.radiation_col] * scenario.radiation_scale

        # 2. Add temporally-correlated noise (smoothed over 24 hours)
        if scenario.noise_std > 0:
            # Temperature noise
            temp_noise = self.rng.normal(0, scenario.noise_std * self.temp_std, n)
            temp_noise = np.convolve(temp_noise, np.ones(24)/24, mode='same')
            df[self.temp_col] = df[self.temp_col] + temp_noise

            # Radiation noise (only during daytime - where radiation > 0)
            rad_noise = self.rng.normal(0, scenario.noise_std * self.radiation_std, n)
            rad_noise = np.convolve(rad_noise, np.ones(12)/12, mode='same')
            daytime_mask = self.historical[self.radiation_col] > 10
            df.loc[daytime_mask, self.radiation_col] += rad_noise[daytime_mask]

        # 3. Inject drought periods (reduce precipitation)
        if scenario.drought_periods:
            for start, duration, intensity in scenario.drought_periods:
                end = min(start + duration, n)
                # intensity=1.0 means complete drought, intensity=0.5 means 50% reduction
                df.loc[start:end, self.precip_col] *= (1.0 - intensity)

        # 4. Inject heat waves (increase temperature)
        if scenario.heatwave_periods:
            for start, duration, temp_add in scenario.heatwave_periods:
                end = min(start + duration, n)
                # Smooth the heat wave edges with a taper
                taper = self._create_taper(duration, taper_frac=0.1)
                indices = range(start, min(start + len(taper), n))
                df.loc[list(indices), self.temp_col] += temp_add * taper[:len(indices)]

        # 5. Inject cold snaps (decrease temperature)
        if scenario.cold_snap_periods:
            for start, duration, temp_drop in scenario.cold_snap_periods:
                end = min(start + duration, n)
                taper = self._create_taper(duration, taper_frac=0.1)
                indices = range(start, min(start + len(taper), n))
                df.loc[list(indices), self.temp_col] -= temp_drop * taper[:len(indices)]

        # 6. Clip to physical bounds
        df[self.precip_col] = df[self.precip_col].clip(lower=0)
        df[self.radiation_col] = df[self.radiation_col].clip(lower=0)
        # Temperature can go negative, but clip extreme values
        df[self.temp_col] = df[self.temp_col].clip(lower=-20, upper=50)

        return df

    def _create_taper(self, duration: int, taper_frac: float = 0.1) -> np.ndarray:
        """Create a tapered window (ramps up, holds, ramps down)."""
        taper_len = max(1, int(duration * taper_frac))
        hold_len = duration - 2 * taper_len

        ramp_up = np.linspace(0, 1, taper_len)
        hold = np.ones(max(0, hold_len))
        ramp_down = np.linspace(1, 0, taper_len)

        return np.concatenate([ramp_up, hold, ramp_down])

    def generate_batch(self, scenarios: List[WeatherScenario]) -> Dict[str, pd.DataFrame]:
        """
        Generate multiple weather scenarios.

        Args:
            scenarios: List of WeatherScenario configurations

        Returns:
            Dictionary mapping scenario names to DataFrames
        """
        return {s.name: self.generate(s) for s in scenarios}


def get_default_scenarios() -> List[WeatherScenario]:
    """
    Get the default set of 20 scenarios ranging from normal to extreme.

    Returns:
        List of WeatherScenario configurations
    """
    scenarios = [
        # Normal years (1-5): minimal perturbation
        WeatherScenario(name="normal_1", precip_scale=1.0, temp_offset=0.0, noise_std=0.03),
        WeatherScenario(name="normal_2", precip_scale=0.95, temp_offset=0.5, noise_std=0.03),
        WeatherScenario(name="normal_3", precip_scale=1.05, temp_offset=-0.5, noise_std=0.03),
        WeatherScenario(name="normal_4", precip_scale=0.9, temp_offset=1.0, noise_std=0.05),
        WeatherScenario(name="normal_5", precip_scale=1.1, temp_offset=-1.0, noise_std=0.05),

        # Moderate years (6-10): noticeable but not extreme
        WeatherScenario(name="moderate_dry", precip_scale=0.75, temp_offset=1.5, noise_std=0.05),
        WeatherScenario(name="moderate_wet", precip_scale=1.25, temp_offset=-1.0, noise_std=0.05),
        WeatherScenario(name="moderate_warm", precip_scale=0.9, temp_offset=2.5, noise_std=0.05),
        WeatherScenario(name="moderate_cool", precip_scale=1.1, temp_offset=-2.0, noise_std=0.05),
        WeatherScenario(name="moderate_variable", precip_scale=1.0, temp_offset=0.0, noise_std=0.15),

        # Drought years (11-14): reduced precipitation with possible events
        WeatherScenario(
            name="mild_drought",
            precip_scale=0.6,
            temp_offset=1.0,
            noise_std=0.05,
            drought_periods=[(800, 300, 0.5)]  # mid-season partial drought
        ),
        WeatherScenario(
            name="summer_drought",
            precip_scale=0.7,
            temp_offset=2.0,
            noise_std=0.05,
            drought_periods=[(1200, 500, 0.7)]  # summer drought during critical growth
        ),
        WeatherScenario(
            name="early_drought",
            precip_scale=0.65,
            temp_offset=1.5,
            noise_std=0.05,
            drought_periods=[(200, 400, 0.8)]  # early season drought
        ),
        WeatherScenario(
            name="late_drought",
            precip_scale=0.7,
            temp_offset=1.0,
            noise_std=0.05,
            drought_periods=[(1800, 600, 0.6)]  # late season drought during grain fill
        ),

        # Wet/cool years (15-16)
        WeatherScenario(name="wet_year", precip_scale=1.4, temp_offset=-1.5, noise_std=0.05),
        WeatherScenario(name="cool_wet", precip_scale=1.3, temp_offset=-2.5, noise_std=0.08),

        # Heat stress years (17-18)
        WeatherScenario(
            name="heat_stress",
            precip_scale=0.8,
            temp_offset=2.0,
            noise_std=0.05,
            heatwave_periods=[(1400, 150, 6)]  # heat wave during flowering
        ),
        WeatherScenario(
            name="multiple_heatwaves",
            precip_scale=0.85,
            temp_offset=1.5,
            noise_std=0.05,
            heatwave_periods=[(800, 100, 5), (1500, 120, 7), (2200, 80, 4)]
        ),

        # Extreme years (19-20)
        WeatherScenario(
            name="extreme_drought_heat",
            precip_scale=0.4,
            temp_offset=3.0,
            noise_std=0.08,
            drought_periods=[(500, 400, 0.9), (1500, 500, 0.85)],
            heatwave_periods=[(1200, 200, 8)]
        ),
        WeatherScenario(
            name="worst_case",
            precip_scale=0.35,
            temp_offset=4.0,
            noise_std=0.1,
            drought_periods=[(300, 600, 0.95), (1600, 800, 0.9)],
            heatwave_periods=[(1000, 250, 10), (2000, 150, 8)]
        ),
    ]

    return scenarios


def scenarios_to_dataframe(scenarios: List[WeatherScenario]) -> pd.DataFrame:
    """Convert scenario list to a summary DataFrame."""
    data = []
    for s in scenarios:
        data.append({
            'name': s.name,
            'precip_scale': s.precip_scale,
            'temp_offset': s.temp_offset,
            'noise_std': s.noise_std,
            'n_drought_events': len(s.drought_periods) if s.drought_periods else 0,
            'n_heatwave_events': len(s.heatwave_periods) if s.heatwave_periods else 0,
            'extremity_index': s.extremity_index()
        })
    return pd.DataFrame(data)
