"""Constants and the parameter ranges the patch generator samples from.

Ranges are a first guess for a Space Apps prototype. Tighten them once real
HiRISE DTM statistics for the chosen landing site are available.
"""

G_MARS = 3.711  # m/s^2
G_EARTH = 9.81  # m/s^2

# MyoSuite environment the walker runs in. It has a "terrain" heightfield geom
# that we overwrite with our own patch on every reset.
ENV_ID = "myoLegRoughTerrainWalk-v0"

# Heightfield layout (the hfield grid in the MyoSuite model is 100 x 100).
HF_HALF_WIDTH_M = 6.0    # lateral half-width (x): 12 m wide, 0.12 m per column
HF_HALF_LENGTH_M = 8.0   # forward half-length: 16 m long, ~0.16 m per row
BACK_MARGIN_M = 2.0      # ground behind the start point
FLAT_START_M = 1.0       # flat run-in before the patch begins
BLEND_M = 1.0            # distance over which slope/roughness fade in

# Simulation limits per patch
MAX_SIM_TIME_S = 10.0    # the registered env truncates at 1000 steps x 0.01 s
END_MARGIN_M = 1.5       # stop when the walker is this close to the far edge

# Metabolic proxy (Margaria efficiencies), see runner.py
EFF_POSITIVE = 0.25
EFF_NEGATIVE = 1.20
BASAL_W_PER_KG = 1.2

# Post spacing of the terrain model the planner reads (HiRISE DTMs are posted at
# 1 m). The dtm_* features are measured after averaging the patch onto this
# spacing, so the surrogate is trained on what the planner can really compute.
# Finer detail is kept in the fine_* columns for analysis only.
DTM_POST_M = 1.0

# Seeded noise (rad, std) added to the walker's joint angles at reset. Without
# it the MyoLeg start state is identical on every patch, so repeated runs on
# smooth ground are copies of each other. Set to 0 to start from the exact keyframe.
INIT_JOINT_NOISE_RAD = 0.005

# A walked patch needs at least this much distance inside the measured region
# for its cost / speed / stability labels to be used (about two strides).
MIN_LABEL_DISTANCE_M = 3.0

# Sampling ranges: (low, high), sampled uniformly unless stated otherwise
RANGES = {
    "slope_deg": (-25.0, 25.0),
    "cross_slope_deg": (-10.0, 10.0),
    "rough_amp_m": (0.0, 0.06),          # std of surface noise
    "rough_wavelength_m": (0.3, 2.0),    # correlation length of that noise
    "rock_density_per_m2": (0.0, 0.4),
    "rock_height_m": (0.03, 0.25),
    "friction": (0.4, 1.0),              # regolith-like Coulomb friction
    "suit_mass_kg": (0.0, 60.0),         # suit + PLSS + tools carried on torso
    "pack_up_m": (-0.15, 0.25),          # pack COM above (+) / below (-) torso COM
    "pack_back_m": (0.05, 0.25),         # pack COM behind torso COM
}

# Fraction of patches simulated at Earth gravity with no suit, so the dataset
# carries its own Earth sanity check against the Minetti / Pandolf curves.
EARTH_FRACTION = 0.1

# Patches per "map". Patches in one map share surface (friction, roughness
# character, rock field statistics) and differ in slope, so they are split
# together (train/val/test) to avoid leakage.
PATCHES_PER_MAP = 8
