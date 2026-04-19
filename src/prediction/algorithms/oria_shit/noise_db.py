"""
Comprehensive Noise Database for Smartphone IMU Sensors.

This module provides noise parameters for various smartphone models, essential for
accurate sensor fusion in both Factor Graph (GTSAM) and Google EKF implementations.

Data sources:
- Sensor chip datasheets (Bosch, STMicroelectronics, TDK Invensense)
- Academic papers on smartphone IMU characterization
- Allan Variance analysis from literature

Units:
- Gyro noise: rad/s (converted from dps/√Hz at 100Hz sampling)
- Accel noise: m/s² (converted from µg/√Hz at 100Hz sampling)
- Bias instability: rad/s for gyro, m/s² for accel
- Random walk: rad/s/√s for gyro, m/s²/√s for accel

Conversion notes:
- Noise density (ND) to sigma at sampling rate f: sigma = ND * sqrt(f)
- dps to rad/s: multiply by π/180
- µg to m/s²: multiply by 9.81e-6
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
import math

# =============================================================================
# Dataclasses for Noise Parameters
# =============================================================================

@dataclass(frozen=True)
class NoiseParams:
    """
    Stores noise parameters for a specific sensor configuration.
    
    All values are standard deviations in SI units unless otherwise noted.
    """
    # White noise (measurement noise) - used in EKF R matrix
    accel_noise_sigma: float  # m/s² - accelerometer white noise std dev
    gyro_noise_sigma: float   # rad/s - gyroscope white noise std dev
    mag_noise_sigma: float    # µT - magnetometer white noise std dev
    
    # Bias instability (low-frequency drift) - used in EKF Q matrix for bias states
    accel_bias_instability: float = 0.0  # m/s²
    gyro_bias_instability: float = 0.0   # rad/s
    
    # Bias prior (initial uncertainty on bias estimate)
    accel_bias_sigma: float = 0.1  # m/s² - prior std dev for accel bias
    gyro_bias_sigma: float = 0.01  # rad/s - prior std dev for gyro bias
    
    # Random walk (for factor graph continuous-time models)
    accel_random_walk: float = 0.0  # m/s²/√s (VRW)
    gyro_random_walk: float = 0.0   # rad/s/√s (ARW)
    
    # Sensor metadata
    sensor_chip: str = "unknown"
    notes: str = ""


@dataclass(frozen=True)
class SensorChipSpec:
    """
    Raw specifications from sensor chip datasheets.
    These are noise densities that need to be converted based on sampling rate.
    """
    name: str
    manufacturer: str
    
    # Noise densities (from datasheet)
    gyro_noise_density_dps_sqrt_hz: float  # dps/√Hz
    accel_noise_density_ug_sqrt_hz: float  # µg/√Hz
    
    # Bias instability (from Allan Variance analysis in datasheets)
    gyro_bias_instability_dph: float = 0.0  # deg/hr
    accel_bias_instability_ug: float = 0.0  # µg
    
    # Typical bias offset range
    gyro_offset_dps: float = 0.0  # ±deg/s
    accel_offset_mg: float = 0.0  # ±mg
    
    def to_noise_params(self, sampling_rate_hz: float = 100.0, 
                       indoor_mag_sigma: float = 50.0,
                       outdoor_mag_sigma: float = 5.0,
                       is_indoor: bool = True) -> NoiseParams:
        """
        Convert datasheet specs to NoiseParams at a given sampling rate.
        
        Args:
            sampling_rate_hz: IMU sampling rate (Hz)
            indoor_mag_sigma: Magnetometer noise for indoor environments (µT)
            outdoor_mag_sigma: Magnetometer noise for outdoor environments (µT)
            is_indoor: Whether this is for indoor or outdoor use
        """
        # Convert noise density to sigma at sampling rate
        # sigma = noise_density * sqrt(sampling_rate)
        sqrt_rate = math.sqrt(sampling_rate_hz)
        
        # Gyro: dps/√Hz -> rad/s
        gyro_sigma = self.gyro_noise_density_dps_sqrt_hz * sqrt_rate * (math.pi / 180.0)
        
        # Accel: µg/√Hz -> m/s²
        accel_sigma = self.accel_noise_density_ug_sqrt_hz * sqrt_rate * 9.81e-6
        
        # Bias instability conversions
        # Gyro: deg/hr -> rad/s
        gyro_bias_inst = self.gyro_bias_instability_dph * (math.pi / 180.0) / 3600.0
        
        # Accel: µg -> m/s²
        accel_bias_inst = self.accel_bias_instability_ug * 9.81e-6
        
        # Bias priors (based on typical offset ranges from datasheet)
        gyro_bias_prior = self.gyro_offset_dps * (math.pi / 180.0) if self.gyro_offset_dps > 0 else 0.01
        accel_bias_prior = self.accel_offset_mg * 9.81e-3 if self.accel_offset_mg > 0 else 0.1
        
        # Random walk (approximate from noise density)
        # ARW ≈ noise_density in rad/s/√Hz
        gyro_rw = self.gyro_noise_density_dps_sqrt_hz * (math.pi / 180.0)
        # VRW ≈ noise_density in m/s²/√Hz
        accel_rw = self.accel_noise_density_ug_sqrt_hz * 9.81e-6
        
        mag_sigma = indoor_mag_sigma if is_indoor else outdoor_mag_sigma
        
        return NoiseParams(
            accel_noise_sigma=accel_sigma,
            gyro_noise_sigma=gyro_sigma,
            mag_noise_sigma=mag_sigma,
            accel_bias_instability=accel_bias_inst,
            gyro_bias_instability=gyro_bias_inst,
            accel_bias_sigma=accel_bias_prior,
            gyro_bias_sigma=gyro_bias_prior,
            accel_random_walk=accel_rw,
            gyro_random_walk=gyro_rw,
            sensor_chip=self.name,
            notes=f"Converted from {self.manufacturer} {self.name} datasheet at {sampling_rate_hz}Hz"
        )


# =============================================================================
# Sensor Chip Database (from official datasheets)
# =============================================================================

SENSOR_CHIPS: Dict[str, SensorChipSpec] = {
    # --- Bosch Sensortec ---
    "bmi270": SensorChipSpec(
        name="BMI270",
        manufacturer="Bosch",
        gyro_noise_density_dps_sqrt_hz=0.007,  # Performance mode
        accel_noise_density_ug_sqrt_hz=160.0,
        gyro_bias_instability_dph=2.0,
        accel_bias_instability_ug=40.0,
        gyro_offset_dps=1.0,
        accel_offset_mg=20.0,
    ),
    "bmi160": SensorChipSpec(
        name="BMI160",
        manufacturer="Bosch",
        gyro_noise_density_dps_sqrt_hz=0.007,
        accel_noise_density_ug_sqrt_hz=180.0,
        gyro_bias_instability_dph=3.0,
        accel_bias_instability_ug=50.0,
        gyro_offset_dps=3.0,
        accel_offset_mg=40.0,
    ),
    "bmi323": SensorChipSpec(
        name="BMI323",
        manufacturer="Bosch",
        gyro_noise_density_dps_sqrt_hz=0.006,
        accel_noise_density_ug_sqrt_hz=120.0,
        gyro_bias_instability_dph=1.5,
        accel_bias_instability_ug=30.0,
        gyro_offset_dps=0.5,
        accel_offset_mg=15.0,
    ),
    
    # --- STMicroelectronics ---
    "lsm6dsr": SensorChipSpec(
        name="LSM6DSR",
        manufacturer="STMicroelectronics",
        gyro_noise_density_dps_sqrt_hz=0.005,  # 5 mdps/√Hz
        accel_noise_density_ug_sqrt_hz=60.0,
        gyro_bias_instability_dph=1.0,
        accel_bias_instability_ug=20.0,
        gyro_offset_dps=0.5,
        accel_offset_mg=10.0,
    ),
    "lsm6dso": SensorChipSpec(
        name="LSM6DSO",
        manufacturer="STMicroelectronics",
        gyro_noise_density_dps_sqrt_hz=0.0035,  # ~3.5 mdps/√Hz
        accel_noise_density_ug_sqrt_hz=70.0,
        gyro_bias_instability_dph=1.5,
        accel_bias_instability_ug=25.0,
        gyro_offset_dps=1.0,
        accel_offset_mg=20.0,
    ),
    "lsm6dsox": SensorChipSpec(
        name="LSM6DSOX",
        manufacturer="STMicroelectronics",
        gyro_noise_density_dps_sqrt_hz=0.0035,
        accel_noise_density_ug_sqrt_hz=70.0,
        gyro_bias_instability_dph=1.0,
        accel_bias_instability_ug=20.0,
        gyro_offset_dps=0.5,
        accel_offset_mg=10.0,
    ),
    "ism330dhcx": SensorChipSpec(
        name="ISM330DHCX",
        manufacturer="STMicroelectronics",
        gyro_noise_density_dps_sqrt_hz=0.0028,  # High performance
        accel_noise_density_ug_sqrt_hz=55.0,
        gyro_bias_instability_dph=0.7,
        accel_bias_instability_ug=15.0,
        gyro_offset_dps=0.3,
        accel_offset_mg=8.0,
    ),
    
    # --- TDK InvenSense ---
    "icm42688": SensorChipSpec(
        name="ICM-42688-P",
        manufacturer="TDK InvenSense",
        gyro_noise_density_dps_sqrt_hz=0.0028,  # 2.8 mdps/√Hz
        accel_noise_density_ug_sqrt_hz=70.0,
        gyro_bias_instability_dph=0.8,
        accel_bias_instability_ug=18.0,
        gyro_offset_dps=0.5,
        accel_offset_mg=12.0,
    ),
    "icm45631": SensorChipSpec(
        name="ICM-45631",
        manufacturer="TDK InvenSense",
        gyro_noise_density_dps_sqrt_hz=0.0038,  # 3.8 mdps/√Hz
        accel_noise_density_ug_sqrt_hz=70.0,
        gyro_bias_instability_dph=1.0,
        accel_bias_instability_ug=20.0,
        gyro_offset_dps=0.5,
        accel_offset_mg=15.0,
    ),
    "icm40609d": SensorChipSpec(
        name="ICM-40609-D",
        manufacturer="TDK InvenSense",
        gyro_noise_density_dps_sqrt_hz=0.0035,
        accel_noise_density_ug_sqrt_hz=65.0,
        gyro_bias_instability_dph=0.9,
        accel_bias_instability_ug=18.0,
        gyro_offset_dps=0.5,
        accel_offset_mg=12.0,
    ),
    "mpu6050": SensorChipSpec(
        name="MPU-6050",
        manufacturer="TDK InvenSense",
        gyro_noise_density_dps_sqrt_hz=0.005,
        accel_noise_density_ug_sqrt_hz=400.0,  # Older, noisier
        gyro_bias_instability_dph=5.0,
        accel_bias_instability_ug=100.0,
        gyro_offset_dps=3.0,
        accel_offset_mg=50.0,
    ),
    
    # --- Generic/Fallback ---
    "generic_premium": SensorChipSpec(
        name="Generic Premium",
        manufacturer="Generic",
        gyro_noise_density_dps_sqrt_hz=0.004,
        accel_noise_density_ug_sqrt_hz=80.0,
        gyro_bias_instability_dph=1.5,
        accel_bias_instability_ug=25.0,
        gyro_offset_dps=1.0,
        accel_offset_mg=20.0,
    ),
    "generic_midrange": SensorChipSpec(
        name="Generic Midrange",
        manufacturer="Generic",
        gyro_noise_density_dps_sqrt_hz=0.007,
        accel_noise_density_ug_sqrt_hz=150.0,
        gyro_bias_instability_dph=3.0,
        accel_bias_instability_ug=50.0,
        gyro_offset_dps=2.0,
        accel_offset_mg=40.0,
    ),
    "generic_budget": SensorChipSpec(
        name="Generic Budget",
        manufacturer="Generic",
        gyro_noise_density_dps_sqrt_hz=0.01,
        accel_noise_density_ug_sqrt_hz=250.0,
        gyro_bias_instability_dph=5.0,
        accel_bias_instability_ug=80.0,
        gyro_offset_dps=3.0,
        accel_offset_mg=60.0,
    ),
}


# =============================================================================
# Smartphone to Sensor Chip Mapping
# =============================================================================

# Maps smartphone model patterns to their known or estimated sensor chips
# Format: "pattern": ("sensor_chip_key", "notes")
SMARTPHONE_SENSOR_MAP: Dict[str, Tuple[str, str]] = {
    # --- Google Pixel ---
    "pixel_10": ("icm45631", "Google Pixel 10 - TDK ICM-456xx series"),
    "pixel_9": ("icm45631", "Google Pixel 9 series - TDK ICM-456xx"),
    "pixel_9_pro": ("icm45631", "Google Pixel 9 Pro - TDK ICM-456xx"),
    "pixel_9_pro_xl": ("icm45631", "Google Pixel 9 Pro XL - TDK ICM-45631 confirmed via app"),
    "pixel_9_pro_fold": ("icm45631", "Google Pixel 9 Pro Fold"),
    "pixel_8": ("icm42688", "Google Pixel 8 series - TDK ICM-42xxx"),
    "pixel_8_pro": ("icm42688", "Google Pixel 8 Pro"),
    "pixel_8a": ("icm42688", "Google Pixel 8a"),
    "pixel_7": ("icm42688", "Google Pixel 7 - Low noise per studies"),
    "pixel_7_pro": ("icm42688", "Google Pixel 7 Pro"),
    "pixel_7a": ("icm42688", "Google Pixel 7a"),
    "pixel_6": ("lsm6dso", "Google Pixel 6 series"),
    "pixel_6_pro": ("lsm6dso", "Google Pixel 6 Pro"),
    "pixel_6a": ("bmi270", "Google Pixel 6a - Budget variant"),
    "pixel_5": ("bmi270", "Google Pixel 5"),
    "pixel_5a": ("bmi270", "Google Pixel 5a"),
    "pixel_4": ("bmi160", "Google Pixel 4 series"),
    "pixel_4_xl": ("bmi160", "Google Pixel 4 XL"),
    "pixel_4a": ("bmi160", "Google Pixel 4a"),
    "pixel_3": ("bmi160", "Google Pixel 3 series"),
    "pixel_fold": ("icm42688", "Google Pixel Fold"),
    
    # --- Apple iPhone ---
    "iphone_16": ("generic_premium", "iPhone 16 series - High-G accelerometer"),
    "iphone_16_pro": ("generic_premium", "iPhone 16 Pro/Pro Max"),
    "iphone_15": ("generic_premium", "iPhone 15 series - High-G accelerometer"),
    "iphone_15_pro": ("generic_premium", "iPhone 15 Pro/Pro Max"),
    "iphone_15_plus": ("generic_premium", "iPhone 15 Plus"),
    "iphone_14": ("generic_premium", "iPhone 14 series - 256G accelerometer"),
    "iphone_14_pro": ("generic_premium", "iPhone 14 Pro/Pro Max"),
    "iphone_14_plus": ("generic_premium", "iPhone 14 Plus"),
    "iphone_13": ("generic_premium", "iPhone 13 series"),
    "iphone_13_pro": ("generic_premium", "iPhone 13 Pro/Pro Max"),
    "iphone_13_mini": ("generic_premium", "iPhone 13 mini"),
    "iphone_12": ("generic_premium", "iPhone 12 series"),
    "iphone_12_pro": ("generic_premium", "iPhone 12 Pro/Pro Max"),
    "iphone_12_mini": ("generic_premium", "iPhone 12 mini"),
    "iphone_11": ("generic_premium", "iPhone 11 series"),
    "iphone_se_3": ("generic_premium", "iPhone SE (3rd gen)"),
    "iphone_se_2": ("generic_midrange", "iPhone SE (2nd gen)"),
    
    # --- Samsung Galaxy S Series ---
    "galaxy_s24": ("lsm6dsr", "Samsung Galaxy S24 series"),
    "galaxy_s24_plus": ("lsm6dsr", "Samsung Galaxy S24+"),
    "galaxy_s24_ultra": ("lsm6dsr", "Samsung Galaxy S24 Ultra"),
    "galaxy_s23": ("lsm6dsr", "Samsung Galaxy S23 series"),
    "galaxy_s23_plus": ("lsm6dsr", "Samsung Galaxy S23+"),
    "galaxy_s23_ultra": ("lsm6dsr", "Samsung Galaxy S23 Ultra"),
    "galaxy_s23_fe": ("lsm6dso", "Samsung Galaxy S23 FE"),
    "galaxy_s22": ("lsm6dsr", "Samsung Galaxy S22 series"),
    "galaxy_s22_plus": ("lsm6dsr", "Samsung Galaxy S22+"),
    "galaxy_s22_ultra": ("lsm6dsr", "Samsung Galaxy S22 Ultra"),
    "galaxy_s21": ("lsm6dso", "Samsung Galaxy S21 series"),
    "galaxy_s21_plus": ("lsm6dso", "Samsung Galaxy S21+"),
    "galaxy_s21_ultra": ("lsm6dso", "Samsung Galaxy S21 Ultra"),
    "galaxy_s21_fe": ("lsm6dso", "Samsung Galaxy S21 FE"),
    "galaxy_s20": ("lsm6dso", "Samsung Galaxy S20 series"),
    "galaxy_s20_plus": ("lsm6dso", "Samsung Galaxy S20+"),
    "galaxy_s20_ultra": ("lsm6dso", "Samsung Galaxy S20 Ultra"),
    "galaxy_s20_fe": ("bmi270", "Samsung Galaxy S20 FE"),
    
    # --- Samsung Galaxy A Series ---
    "galaxy_a55": ("bmi270", "Samsung Galaxy A55"),
    "galaxy_a54": ("bmi270", "Samsung Galaxy A54"),
    "galaxy_a53": ("bmi270", "Samsung Galaxy A53"),
    "galaxy_a52": ("bmi270", "Samsung Galaxy A52"),
    "galaxy_a35": ("bmi270", "Samsung Galaxy A35"),
    "galaxy_a34": ("bmi270", "Samsung Galaxy A34"),
    "galaxy_a33": ("bmi270", "Samsung Galaxy A33"),
    "galaxy_a25": ("generic_midrange", "Samsung Galaxy A25"),
    "galaxy_a24": ("generic_midrange", "Samsung Galaxy A24"),
    "galaxy_a15": ("generic_budget", "Samsung Galaxy A15"),
    "galaxy_a14": ("generic_budget", "Samsung Galaxy A14"),
    
    # --- Samsung Galaxy Z (Foldables) ---
    "galaxy_z_fold_6": ("lsm6dsr", "Samsung Galaxy Z Fold 6"),
    "galaxy_z_fold_5": ("lsm6dsr", "Samsung Galaxy Z Fold 5"),
    "galaxy_z_fold_4": ("lsm6dsr", "Samsung Galaxy Z Fold 4"),
    "galaxy_z_flip_6": ("lsm6dsr", "Samsung Galaxy Z Flip 6"),
    "galaxy_z_flip_5": ("lsm6dsr", "Samsung Galaxy Z Flip 5"),
    "galaxy_z_flip_4": ("lsm6dsr", "Samsung Galaxy Z Flip 4"),
    
    # --- Samsung Galaxy Note ---
    "galaxy_note_20": ("lsm6dso", "Samsung Galaxy Note 20 series"),
    "galaxy_note_20_ultra": ("lsm6dso", "Samsung Galaxy Note 20 Ultra"),
    "galaxy_note_10": ("lsm6dsr", "Samsung Galaxy Note 10 - LSM6DSR confirmed"),
    
    # --- OnePlus ---
    "oneplus_12": ("icm42688", "OnePlus 12"),
    "oneplus_12r": ("icm42688", "OnePlus 12R"),
    "oneplus_11": ("icm42688", "OnePlus 11"),
    "oneplus_11r": ("bmi270", "OnePlus 11R"),
    "oneplus_10": ("bmi270", "OnePlus 10 series"),
    "oneplus_10_pro": ("bmi270", "OnePlus 10 Pro"),
    "oneplus_9": ("bmi270", "OnePlus 9 series"),
    "oneplus_9_pro": ("bmi270", "OnePlus 9 Pro"),
    "oneplus_8": ("bmi160", "OnePlus 8 series"),
    "oneplus_8_pro": ("bmi160", "OnePlus 8 Pro"),
    "oneplus_7": ("bmi160", "OnePlus 7 series - Low noise confirmed"),
    "oneplus_7_pro": ("bmi160", "OnePlus 7 Pro"),
    "oneplus_nord": ("generic_midrange", "OnePlus Nord series"),
    "oneplus_nord_ce": ("generic_midrange", "OnePlus Nord CE series"),
    
    # --- Xiaomi ---
    "xiaomi_14": ("icm42688", "Xiaomi 14 series"),
    "xiaomi_14_pro": ("icm42688", "Xiaomi 14 Pro"),
    "xiaomi_14_ultra": ("icm42688", "Xiaomi 14 Ultra"),
    "xiaomi_13": ("bmi270", "Xiaomi 13 series"),
    "xiaomi_13_pro": ("bmi270", "Xiaomi 13 Pro"),
    "xiaomi_13_ultra": ("bmi270", "Xiaomi 13 Ultra"),
    "xiaomi_12": ("bmi270", "Xiaomi 12 series"),
    "xiaomi_12_pro": ("bmi270", "Xiaomi 12 Pro"),
    "poco_f6": ("bmi270", "Poco F6 series"),
    "poco_x6": ("generic_midrange", "Poco X6 series"),
    "redmi_note_13": ("generic_midrange", "Redmi Note 13 series"),
    "redmi_note_12": ("generic_midrange", "Redmi Note 12 series"),
    "redmi_13": ("generic_budget", "Redmi 13 series"),
    
    # --- OPPO ---
    "oppo_find_x7": ("icm42688", "OPPO Find X7 series"),
    "oppo_find_x6": ("bmi270", "OPPO Find X6 series"),
    "oppo_find_n3": ("icm42688", "OPPO Find N3 Fold"),
    "oppo_reno_11": ("bmi270", "OPPO Reno 11 series"),
    "oppo_reno_10": ("bmi270", "OPPO Reno 10 series"),
    "oppo_a79": ("generic_midrange", "OPPO A79"),
    "oppo_a58": ("generic_budget", "OPPO A58"),
    
    # --- Vivo ---
    "vivo_x100": ("icm42688", "Vivo X100 series"),
    "vivo_x90": ("bmi270", "Vivo X90 series"),
    "vivo_v30": ("bmi270", "Vivo V30 series"),
    "vivo_y200": ("generic_midrange", "Vivo Y200 series"),
    
    # --- Honor ---
    "honor_magic_6": ("icm42688", "Honor Magic 6 series"),
    "honor_magic_5": ("bmi270", "Honor Magic 5 series"),
    "honor_90": ("bmi270", "Honor 90 series"),
    "honor_x50": ("generic_midrange", "Honor X50 series"),
    
    # --- Huawei ---
    "huawei_mate_60": ("bmi323", "Huawei Mate 60 series"),
    "huawei_mate_50": ("bmi270", "Huawei Mate 50 series"),
    "huawei_p60": ("bmi323", "Huawei P60 series"),
    "huawei_p50": ("bmi270", "Huawei P50 series"),
    "huawei_nova_12": ("bmi270", "Huawei Nova 12 series"),
    
    # --- Sony ---
    "sony_xperia_1_vi": ("lsm6dsox", "Sony Xperia 1 VI"),
    "sony_xperia_1_v": ("lsm6dsox", "Sony Xperia 1 V"),
    "sony_xperia_5_v": ("lsm6dso", "Sony Xperia 5 V"),
    "sony_xperia_10_vi": ("bmi270", "Sony Xperia 10 VI"),
    
    # --- ASUS ---
    "asus_rog_phone_8": ("icm42688", "ASUS ROG Phone 8"),
    "asus_rog_phone_7": ("icm42688", "ASUS ROG Phone 7"),
    "asus_zenfone_11": ("bmi270", "ASUS Zenfone 11"),
    "asus_zenfone_10": ("bmi270", "ASUS Zenfone 10"),
    
    # --- Motorola ---
    "moto_edge_50": ("bmi270", "Motorola Edge 50 series"),
    "moto_edge_40": ("bmi270", "Motorola Edge 40 series"),
    "moto_g84": ("generic_midrange", "Motorola Moto G84"),
    "moto_g54": ("generic_midrange", "Motorola Moto G54"),
    
    # --- Nothing ---
    "nothing_phone_2a": ("bmi270", "Nothing Phone (2a)"),
    "nothing_phone_2": ("bmi270", "Nothing Phone (2)"),
    "nothing_phone_1": ("bmi270", "Nothing Phone (1)"),
    
    # --- Realme ---
    "realme_gt_5": ("bmi270", "Realme GT 5 series"),
    "realme_gt_neo_5": ("bmi270", "Realme GT Neo 5"),
    "realme_12_pro": ("generic_midrange", "Realme 12 Pro series"),
    "realme_c67": ("generic_budget", "Realme C67"),
}


# =============================================================================
# Noise Database Class
# =============================================================================

class NoiseDatabase:
    """
    Manages noise parameters for different smartphone models and environments.
    
    Usage:
        from core.noise_db import noise_db
        
        # Get parameters for a specific phone
        params = noise_db.get_params("pixel_9_pro", is_indoor=True)
        
        # Access individual values
        print(f"Gyro noise: {params.gyro_noise_sigma} rad/s")
        print(f"Accel bias prior: {params.accel_bias_sigma} m/s²")
    """
    
    def __init__(self, sampling_rate_hz: float = 100.0):
        """
        Initialize the noise database.
        
        Args:
            sampling_rate_hz: Default IMU sampling rate for noise calculations
        """
        self.sampling_rate = sampling_rate_hz
        self._cache: Dict[str, Dict[str, NoiseParams]] = {}
    
    def _normalize_key(self, device_model: str) -> str:
        """Normalize device model string to a lookup key."""
        return device_model.lower().replace(" ", "_").replace("-", "_")
    
    def _find_best_match(self, key: str) -> Tuple[str, str, str]:
        """
        Find the best matching sensor chip for a device key.
        
        Returns:
            (sensor_chip_key, notes, match_type)
        """
        # Exact match
        if key in SMARTPHONE_SENSOR_MAP:
            chip, notes = SMARTPHONE_SENSOR_MAP[key]
            return chip, notes, "exact"
        
        # Partial match (try progressively shorter prefixes)
        best_match = None
        best_match_len = 0
        
        for pattern, (chip, notes) in SMARTPHONE_SENSOR_MAP.items():
            if key.startswith(pattern) or pattern.startswith(key):
                match_len = min(len(key), len(pattern))
                if match_len > best_match_len:
                    best_match = (chip, notes, "partial")
                    best_match_len = match_len
        
        if best_match:
            return best_match
        
        # Brand-based fallback
        brand_defaults = {
            "pixel": ("icm42688", "Google Pixel (generic)", "brand"),
            "iphone": ("generic_premium", "Apple iPhone (generic)", "brand"),
            "galaxy": ("lsm6dso", "Samsung Galaxy (generic)", "brand"),
            "oneplus": ("bmi270", "OnePlus (generic)", "brand"),
            "xiaomi": ("bmi270", "Xiaomi (generic)", "brand"),
            "redmi": ("generic_midrange", "Redmi (generic)", "brand"),
            "poco": ("generic_midrange", "Poco (generic)", "brand"),
            "oppo": ("bmi270", "OPPO (generic)", "brand"),
            "vivo": ("bmi270", "Vivo (generic)", "brand"),
            "honor": ("bmi270", "Honor (generic)", "brand"),
            "huawei": ("bmi270", "Huawei (generic)", "brand"),
            "sony": ("lsm6dso", "Sony Xperia (generic)", "brand"),
            "asus": ("bmi270", "ASUS (generic)", "brand"),
            "moto": ("generic_midrange", "Motorola (generic)", "brand"),
            "nothing": ("bmi270", "Nothing (generic)", "brand"),
            "realme": ("generic_midrange", "Realme (generic)", "brand"),
        }
        
        for brand, (chip, notes, match_type) in brand_defaults.items():
            if brand in key:
                return chip, notes, match_type
        
        # Ultimate fallback
        return "generic_midrange", "Unknown device - using generic midrange", "fallback"
    
    def get_params(self, device_model: str, is_indoor: bool = True,
                  sampling_rate_hz: Optional[float] = None) -> NoiseParams:
        """
        Get noise parameters for a specific device.
        
        Args:
            device_model: Device model name (e.g., "pixel_9_pro", "galaxy_s24")
            is_indoor: Whether indoor (high mag noise) or outdoor parameters
            sampling_rate_hz: Override default sampling rate
            
        Returns:
            NoiseParams with all required values for sensor fusion
        """
        key = self._normalize_key(device_model)
        env = "indoor" if is_indoor else "outdoor"
        cache_key = f"{key}_{env}_{sampling_rate_hz or self.sampling_rate}"
        
        # Check cache
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # Find sensor chip
        chip_key, notes, match_type = self._find_best_match(key)
        
        if chip_key not in SENSOR_CHIPS:
            chip_key = "generic_midrange"
        
        chip = SENSOR_CHIPS[chip_key]
        rate = sampling_rate_hz or self.sampling_rate
        
        # Generate parameters
        params = chip.to_noise_params(
            sampling_rate_hz=rate,
            is_indoor=is_indoor
        )
        
        # Cache and return
        self._cache[cache_key] = params
        return params
    
    def get_sensor_chip(self, device_model: str) -> Optional[SensorChipSpec]:
        """Get the sensor chip specification for a device."""
        key = self._normalize_key(device_model)
        chip_key, _, _ = self._find_best_match(key)
        return SENSOR_CHIPS.get(chip_key)
    
    def list_devices(self) -> Dict[str, str]:
        """List all known devices and their sensor chips."""
        return {k: v[0] for k, v in SMARTPHONE_SENSOR_MAP.items()}
    
    def list_sensor_chips(self) -> Dict[str, str]:
        """List all sensor chips in the database."""
        return {k: f"{v.manufacturer} {v.name}" for k, v in SENSOR_CHIPS.items()}


# =============================================================================
# Global Instance
# =============================================================================

noise_db = NoiseDatabase(sampling_rate_hz=100.0)


# =============================================================================
# Convenience Function for Quick Access
# =============================================================================

def get_noise_params(device: str, indoor: bool = True, rate: float = 100.0) -> NoiseParams:
    """
    Quick access to noise parameters.
    
    Example:
        params = get_noise_params("pixel_9_pro", indoor=True)
    """
    return noise_db.get_params(device, is_indoor=indoor, sampling_rate_hz=rate)
