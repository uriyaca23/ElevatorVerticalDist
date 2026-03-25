# Algorithm 2: Dynamic Time Warping (DTW) Pattern Matching
## Theoretical Background
Dynamic Time Warping (DTW) is a classic algorithmic technique used in speech recognition, gesture recognition, and time-series analysis to measure similarity between two temporal sequences which may vary in speed.

An elevator ride from a static start to a static end always has a structural template:
1. An acceleration pulse (area = +V).
2. A constant velocity period of arbitrary length (Z-accel = 0).
3. A deceleration pulse (area = -V).

## Proposed Strategy
1. **Template Generation:** We synthesize a continuous "ideal" elevator acceleration template. For example, a 1-second pulse of +0.5 m/s², a stretch of 0 m/s², and a 1-second pulse of -0.5 m/s².
2. **Signal Preprocessing:** Extract moving variance to filter out walking segments. Only operate DTW on contiguous spans of "low variance" (standing still).
3. **Sliding DTW:** Since DTW can be computationally expensive over long sequences, we apply a sliding window or use a derivative form of Subsequence DTW (sDTW). We warp the template against any arbitrary length of quiet standing data.
4. **Thresholding:** If the DTW cost (distance) between the optimal warped template and the measured linear vertical acceleration is below a specific threshold, we classify the entire subsequence as an `ELEVATOR_RIDE`.

## Advantages and Disadvantages
- **Advantage:** Extremely robust against varied lengths of elevator rides (short 1-floor hops or long 20-floor rides) because DTW gracefully expands the zero-duration segment.
- **Disadvantage:** Computationally intensive `O(N*M)`. Requires careful tuning of the penalty for stretching the non-zero acceleration segments so that DTW doesn't blindly match noise.
