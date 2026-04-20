"""Phone-model → accelerometer white-noise σ lookup.

Port of the sensor-chip table from ``uriya_shit/noise_db.py`` reduced to
only what the prediction algorithms need: the accelerometer noise
density σ (m/s²) at a given sampling rate. The table maps smartphone
model patterns to a sensor-chip spec, then converts the datasheet
noise density to σ via σ = ND · √f.

Kept intentionally small so we don't drag gyro/mag noise specs into
the prediction stack. Chips and phones were adapted from the original
table.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class _ChipSpec:
    """Bare-minimum sensor chip spec for accel noise."""
    name: str
    accel_noise_density_ug_sqrt_hz: float  # µg/√Hz

    def accel_noise_sigma(self, fs_hz: float) -> float:
        """Convert ND (µg/√Hz) to σ (m/s²) at sampling rate ``fs_hz``."""
        return self.accel_noise_density_ug_sqrt_hz * math.sqrt(fs_hz) * 9.81e-6


_CHIPS: dict[str, _ChipSpec] = {
    # Bosch
    "bmi270": _ChipSpec("BMI270", 160.0),
    "bmi160": _ChipSpec("BMI160", 180.0),
    "bmi323": _ChipSpec("BMI323", 120.0),
    # ST
    "lsm6dsr": _ChipSpec("LSM6DSR", 60.0),
    "lsm6dso": _ChipSpec("LSM6DSO", 70.0),
    "lsm6dsox": _ChipSpec("LSM6DSOX", 70.0),
    # TDK
    "icm42688": _ChipSpec("ICM-42688-P", 70.0),
    "icm45631": _ChipSpec("ICM-45631", 70.0),
    "icm40609d": _ChipSpec("ICM-40609-D", 65.0),
    "mpu6050": _ChipSpec("MPU-6050", 400.0),
    # Generic fallbacks
    "generic_premium": _ChipSpec("Generic Premium", 80.0),
    "generic_midrange": _ChipSpec("Generic Midrange", 150.0),
    "generic_budget": _ChipSpec("Generic Budget", 250.0),
}


# Pattern → chip. Patterns are matched as lowercase substrings of the
# normalised phone model string. First match wins; more specific
# patterns should appear first.
_PHONE_PATTERNS: list[tuple[str, str]] = [
    # Google Pixel
    ("pixel10", "icm45631"),
    ("pixel_10", "icm45631"),
    ("pixel 10", "icm45631"),
    ("pixel9", "icm45631"),
    ("pixel_9", "icm45631"),
    ("pixel 9", "icm45631"),
    ("pixel8", "icm42688"),
    ("pixel7", "icm42688"),
    ("pixel6", "lsm6dso"),
    ("pixel5", "bmi270"),
    ("pixel4", "bmi160"),
    # Samsung Galaxy S
    ("sm-s9", "lsm6dsr"),   # S22/S23 (the ones in our dataset)
    ("galaxys24", "lsm6dsr"),
    ("galaxy_s24", "lsm6dsr"),
    ("galaxy s24", "lsm6dsr"),
    ("galaxys23", "lsm6dsr"),
    ("galaxy_s23", "lsm6dsr"),
    ("galaxy s23", "lsm6dsr"),
    ("galaxys22", "lsm6dsr"),
    ("galaxy s22", "lsm6dsr"),
    # Samsung Galaxy A
    ("sm-a2", "bmi270"),    # A23/A25
    ("sm-a3", "bmi270"),    # A33/A34/A35 (incl. SM-A235F, SM-A356)
    ("galaxy_a5", "bmi270"),
    ("galaxy a5", "bmi270"),
    # Samsung Galaxy Z (foldables)
    ("zflip6", "lsm6dsr"),
    ("z_flip_6", "lsm6dsr"),
    ("z flip 6", "lsm6dsr"),
    ("zflip", "lsm6dsr"),
    ("zfold", "lsm6dsr"),
    # Xiaomi model numbers
    ("22101320", "bmi270"),  # Xiaomi 12T-class
    ("xiaomi13", "bmi270"),
    ("xiaomi_13", "bmi270"),
    ("xiaomi14", "icm42688"),
    ("xiaomi_14", "icm42688"),
    # iPhone (generic premium — datasheet undisclosed)
    ("iphone", "generic_premium"),
]

# Brand → chip, ultimate fallback if no pattern matches.
_BRAND_FALLBACK: list[tuple[str, str]] = [
    ("pixel", "icm42688"),
    ("galaxy", "lsm6dso"),
    ("samsung", "lsm6dso"),
    ("xiaomi", "bmi270"),
    ("redmi", "generic_midrange"),
    ("poco", "generic_midrange"),
    ("oneplus", "bmi270"),
    ("oppo", "bmi270"),
    ("vivo", "bmi270"),
    ("huawei", "bmi270"),
    ("honor", "bmi270"),
    ("motorola", "generic_midrange"),
    ("moto", "generic_midrange"),
    ("nothing", "bmi270"),
    ("realme", "generic_midrange"),
    ("sony", "lsm6dso"),
    ("asus", "bmi270"),
    ("iphone", "generic_premium"),
    ("apple", "generic_premium"),
]


def _normalise(phone: str) -> str:
    return phone.lower().replace("-", "").replace("_", "").replace(" ", "")


def resolve_phone_to_chip(phone_model: str) -> str:
    """Return the chip key best matching ``phone_model``.

    The match is a substring search against ``_PHONE_PATTERNS`` first,
    then brand fallback. Unknown phones fall back to ``generic_midrange``.
    """
    if not phone_model:
        return "generic_midrange"
    key = _normalise(phone_model)

    for pattern, chip in _PHONE_PATTERNS:
        if _normalise(pattern) in key:
            return chip

    for brand, chip in _BRAND_FALLBACK:
        if brand in key:
            return chip

    return "generic_midrange"


def get_phone_accel_noise_sigma(phone_model: str, fs_hz: float = 50.0) -> float:
    """Return accelerometer white-noise σ (m/s²) for ``phone_model`` at
    sampling rate ``fs_hz``. Falls back to a generic midrange chip if
    the phone is unknown.
    """
    chip_key = resolve_phone_to_chip(phone_model)
    chip = _CHIPS.get(chip_key) or _CHIPS["generic_midrange"]
    return float(chip.accel_noise_sigma(fs_hz))
