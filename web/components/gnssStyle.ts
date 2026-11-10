// Colours for the GPS map, so a class reads the same in the map, its legend
// and its segment list. Slots from the reference data-viz palette, validated as a
// pair (stay↔movement); "unknown" is a neutral that asserts nothing and is
// always also drawn dashed / hatched.
import type { StayClass } from "@/lib/types";

export const STAY_COLOR: Record<StayClass, string> = {
  stay: "#d55181",
  movement: "#2a78d6",
  unknown: "#9a9993",
};

export const STAY_LABEL: Record<StayClass, string> = {
  stay: "stay",
  movement: "movement",
  unknown: "unknown",
};

export const INK = "#0b0b0b";
export const INK_2 = "#52514e";
export const INK_MUTED = "#9a9993";
