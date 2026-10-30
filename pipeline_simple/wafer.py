import numpy as np

# The 8 defects, in the order the label uses
DEFECTS = ["Center", "Donut", "Edge-Loc", "Edge-Ring", "Loc", "Near-Full", "Scratch", "Random"]

# For every square on the 52 x 52 map: row, column, how far it is from the middle
ROWS, COLS = np.indices((52, 52))
DISTANCE = np.sqrt((ROWS - 26) ** 2 + (COLS - 26) ** 2)

# Ring 0 (middle) to 4 (edge). No die is further than 26.5 from the middle
RING = np.minimum((DISTANCE / 26.5 * 5).astype(int), 4)

# Slice 0 to 7 around the middle, like a pizza cut in 8
ANGLE = np.degrees(np.arctan2(ROWS - 26, COLS - 26)) + 180
SLICE = (ANGLE // 45).astype(int) % 8

FEATURE_NAMES = ["all",
                 "ring_0", "ring_1", "ring_2", "ring_3", "ring_4",
                 "slice_0", "slice_1", "slice_2", "slice_3", "slice_4", "slice_5", "slice_6", "slice_7"]


def label_name(label):
    # [1 0 0 0 0 0 1 0] -> "Center + Scratch". All zeros = Normal.
    names = []
    for i in range(8):
        if label[i] == 1:
            names.append(DEFECTS[i])
    if len(names) == 0:
        return "Normal"
    return " + ".join(names)


def share_failed(wafer, area):
    # of the dies inside the area, the share that failed. 0 = no die, 1 = good, 2 = failed.
    dies = (wafer > 0) & area
    if dies.sum() == 0:
        return 0.0   # an empty area, so no division by zero
    return (wafer[dies] == 2).mean()


def features(wafer):
    # One wafer map -> 14 numbers: the wafer, 5 rings and 8 slices.
    numbers = [share_failed(wafer, wafer >= 0)]
    for k in range(5):
        numbers.append(share_failed(wafer, RING == k))
    for k in range(8):
        numbers.append(share_failed(wafer, SLICE == k))
    return numbers
