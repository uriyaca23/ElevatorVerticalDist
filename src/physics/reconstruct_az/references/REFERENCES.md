# Source papers — `reconstruct_az`

Primary references for the five orientation-fusion algorithms used to
reconstruct world-frame (gravity-removed) acceleration from accelerometer +
gyroscope. Each algorithm estimates device orientation `q(t)`, rotates the
measured specific force into the world frame, and subtracts gravity; `a_z`
in the world frame is then the *real vertical acceleration*.

| # | Algorithm (module) | Primary reference | PDF here? |
|---|---|---|---|
| 1 | Complementary (`complementary.py`) | Euston, Coote, Mahony, Kim, Hamel (2008), *A complementary filter for attitude estimation of a fixed-wing UAV*, IROS. (Foundational idea: Higgins (1975), *A comparison of complementary and Kalman filtering*, IEEE T-AES.) | textbook — see links below |
| 2 | Mahony (`mahony.py`) | Mahony, Hamel, Pflimlin (2008), *Nonlinear Complementary Filters on the Special Orthogonal Group*, IEEE TAC 53(5):1203–1218. doi:10.1109/TAC.2008.923738 | bot-walled — see links below |
| 3 | Madgwick (`madgwick.py`) | Madgwick (2010), *An efficient orientation filter for inertial and inertial/magnetic sensor arrays*, x-io technical report. (Published: Madgwick, Harrison, Vaidyanathan (2011), IEEE ICORR.) | ✅ `madgwick2010_report.pdf` |
| 4 | Valenti / AQUA (`valenti.py`) | Valenti, Dryanovski, Xiao (2015), *Keeping a Good Attitude: A Quaternion-Based Orientation Filter for IMUs and MARGs*, Sensors 15(8):19302–19330. doi:10.3390/s150819302 | ✅ `valenti2015_aqua.pdf` |
| 5 | Error-State Kalman Filter (`eskf.py`) | Solà (2017), *Quaternion kinematics for the error-state Kalman filter*, arXiv:1711.02508. | ✅ `sola2017_eskf.pdf` |

## Manual-download links (WAF/bot-blocked from this environment)

- **Mahony 2008** (open-access copy): <https://hal.science/hal-00488376/document>
  · IEEE: <https://doi.org/10.1109/TAC.2008.923738>
- **Euston 2008** (complementary, attitude): <https://doi.org/10.1109/IROS.2008.4650766>
- **Higgins 1975** (complementary vs Kalman): <https://doi.org/10.1109/TAES.1975.308081>

> The complementary filter has no single canonical paper — it is standard
> textbook sensor fusion. The implementation in `complementary.py` follows
> the tilt-from-accelerometer + gyro-integration form described in Euston
> (2008) and summarised in the `ahrs` library docs
> (<https://ahrs.readthedocs.io/en/latest/filters/complementary.html>).
