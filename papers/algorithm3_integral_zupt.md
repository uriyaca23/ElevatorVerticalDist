# Algorithm 3: Velocity-Integral Bounding with ZUPT
## Theoretical Background
While State Machines (Algorithm 1) rely on crossing specific acceleration thresholds, and DTW (Algorithm 2) relies on shape, the most fundamental physical law of an elevator is conservation of momentum / integral constraint: the start velocity is zero, and the end velocity is exactly zero.

Zero-Velocity Update (ZUPT) is standard in inertial navigation. By identifying "Standstill" events from variance, we know boundaries where velocity = 0.

## Proposed Strategy
1. **Standstill Detection:** Use a sliding variance window to classify every timestamp as `WALKING` or `STANDING`.
2. **Velocity Integration:** Between any two `WALKING` events, we have a continuous `STANDING` block. Within this block, integrate `a_{linear}` to calculate velocity `v(t)`.
3. **Event Constraint Checking:** 
   - Rule A: The maximum absolute velocity `max(|v|)` within the block must exceed an elevator minimum speed (e.g., `> 0.8 m/s`).
   - Rule B: The integral of velocity (Distance) must exceed a minimum floor height (e.g., `> 2.0 m`).
   - Rule C: The final velocity at the end of the `STANDING` block must naturally return near zero (e.g., `< 0.3 m/s` error) BEFORE any ZUPT correction is forcefully applied.
4. **Classification:** Any `STANDING` block that satisfies A, B, and C is automatically classified as an elevator ride segment. The exact start/end of the internal movement dictates the segment bounds.

## Advantages and Disadvantages
- **Advantage:** Extremely clean, physics-based constraint approach. Leverages the very definitions of a complete elevator trip.
- **Disadvantage:** If the user shifts their weight heavily while waiting in the elevator, it might prematurely trigger a `WALKING` state, artificially breaking the `STANDING` block into two halves, neither of which return perfectly to zero velocity! Careful tuning of the standstill threshold is critical.
