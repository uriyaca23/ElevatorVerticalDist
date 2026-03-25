# Algorithm 1: State Machine via Low-Pass Filtering
## Theoretical Background
In pedestrian dead reckoning and indoor localization literature, detecting an elevator segment purely from a smartphone accelerometer is a well-studied problem. Elevators produce a very distinct physical footprint:
1. **Low Frequency Acceleration/Deceleration:** To maintain passenger comfort, elevators accelerate smoothly, hold a constant velocity (zero acceleration), and then decelerate. This produces a pair of opposite acceleration pulses.
2. **Absence of High-Frequency Noise:** Unlike walking or running, riding an elevator provides a physically stable platform. High-frequency variance (step impacts) drops to near zero.

## Proposed Strategy
1. **Gravity Isolation:** Calculate the magnitude of the 3-axis accelerometer and subtract standard gravity (9.8m/s²) to retrieve linear acceleration `a_{lin}` in the vertical frame.
2. **Low-Pass Filter:** Apply a strong Butterworth low-pass filter (cutoff frequency ~0.5 Hz) to the magnitude to eliminate the minor jitter from holding the phone and hand movements.
3. **Variance Thresholding:** Apply a rolling standard deviation (window ~2s). If variance > threshold (e.g., > 1.0 m/s²), tag the state as "WALKING" and reject any elevator hypothesis.
4. **State Machine Logic:** 
    - When variance is low (STANDING), search for a positive or negative acceleration pulse exceeding `thr_acc` (e.g., 0.3 m/s²) lasting at least `t_min` seconds.
    - If a pulse is found, enter `ELEVATOR_MOVING` state.
    - Integrate the acceleration to track virtual velocity. 
    - The segment ends when an opposite pulse brings the virtual velocity back to near zero, followed by the user exiting the elevator (variance returning high -> WALKING).
    
## Advantages and Disadvantages
- **Advantage:** Highly interpretable. Easily adjustable thresholds based on physical principles.
- **Disadvantage:** Vulnerable to false positives if the user stands still on an escalator or moving walkway, or if they shake the phone in a very specific low-frequency manner.
