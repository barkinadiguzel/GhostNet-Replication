# GhostNet architecture configuration based on the original paper (Table 1)
# Each entry:
# (kernel_size, exp_size, out_channels, use_se, stride)

GHOSTNET_CONFIG = [
    # stage 1
    (3, 16, 16, False, 1),

    # stage 2
    (3, 48, 24, False, 2),
    (3, 72, 24, False, 1),

    # stage 3
    (5, 72, 40, True, 2),
    (5, 120, 40, True, 1),

    # stage 4
    (3, 240, 80, False, 2),
    (3, 200, 80, False, 1),
    (3, 184, 80, False, 1),
    (3, 184, 80, False, 1),

    # stage 5
    (5, 480, 112, True, 1),
    (5, 672, 112, True, 1),

    # stage 6
    (5, 672, 160, True, 2),
    (5, 960, 160, False, 1),
    (5, 960, 160, True, 1),
    (5, 960, 160, False, 1),
    (5, 960, 160, True, 1),
]

# final layers
FINAL_EXP = 960
FINAL_OUT = 1280
NUM_CLASSES = 1000
