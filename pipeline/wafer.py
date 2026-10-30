import os
import numpy as np

# Shared functions for the pipeline notebooks.

# The 8 basic defects, in the order the label uses. The order comes from the dataset description .
DEFECTS = ["Center", "Donut", "Edge-Loc", "Edge-Ring", "Loc", "Near-Full", "Scratch", "Random"]

# Row and column number of every square on the 52 x 52 map
ROWS, COLS = np.indices((52, 52))

# Distance of every square to the middle. 26 is where the dies are centred: no die is further than 26 from it
DISTANCE = np.sqrt((ROWS - 52/2) ** 2 + (COLS - 52/2) ** 2)

# Ring 0 (middle) to 4 (edge). The corners end up in ring 4 too, but they have no dies so it doesn't matter
# 26.5 is just over the wafer's radius, so the outer dies still land in ring 4 and not past it
RING = np.minimum((DISTANCE / 26.5 * 5).astype(int), 4)

# Slice 0 to 7 around the middle, like a pizza. arctan2 gives -180 to 180, +180 makes it 0 to 360
ANGLE = np.degrees(np.arctan2(ROWS - 52/2, COLS - 52/2))
SECTOR = ((ANGLE + 180) // 45).astype(int) % 8

# The whole map as one area, for the first feature
EVERYWHERE = np.ones((52, 52), dtype=bool)

# Names of the 14 features, in the same order features() gives them back
FEATURE_NAMES = ["share_failed",
                 "ring_0", "ring_1", "ring_2", "ring_3", "ring_4",
                 "slice_0", "slice_1", "slice_2", "slice_3", "slice_4", "slice_5", "slice_6", "slice_7"]


def label_name(label):
    """[0 1 0 0 0 0 1 0] -> "Donut + Scratch", and all zeros -> "Normal" (FR-02)."""
    defect_present = []
    # Loop through each defect position (0-7)
    for i in range(len(DEFECTS)):
        # If this position is 1, the defect is present
        if label[i] == 1:
            defect_present.append(DEFECTS[i])

    if len(defect_present) == 0:
        return "Normal"
    return " + ".join(defect_present)


def make_split(labels, seed):
    """Split 70/15/15 per type with a fixed seed. The same loop as in notebook 02, kept here so 08 can rerun it."""
    rng = np.random.default_rng(seed)
    train = []
    val = []
    test = []

    # every different label combination is one type
    for combo in np.unique(labels, axis=0):
        rows = np.where((labels == combo).all(axis=1))[0]
        rng.shuffle(rows)

        n_train = int(len(rows) * 0.70)
        n_val = int(len(rows) * 0.15)

        train.extend(rows[:n_train])
        val.extend(rows[n_train:n_train + n_val])
        test.extend(rows[n_train + n_val:])   # the rest, so no wafer gets lost to rounding

    return np.sort(np.array(train)), np.sort(np.array(val)), np.sort(np.array(test))


def share_failed_in(wafer, area):
    """Of the dies inside `area`, the share that failed (0 to 1)."""
    # 0 means no die, so those squares don't count
    dies = (wafer > 0) & area

    # no dies in this area: 0 instead of nan, because a nan breaks training
    if dies.sum() == 0:
        return 0.0

    failed = wafer[dies] == 2
    return float(failed.mean())


def features(wafer):
    """One wafer map -> the 14 numbers, in the order of FEATURE_NAMES."""
    numbers = [share_failed_in(wafer, EVERYWHERE)]

    for k in range(5):
        numbers.append(share_failed_in(wafer, RING == k))

    for k in range(8):
        numbers.append(share_failed_in(wafer, SECTOR == k))

    return numbers


def feature_table(maps):
    """features() for many wafers, stacked into one table: one row per wafer, 14 columns."""
    rows = []
    for w in maps:
        rows.append(features(w))
    return np.array(rows)


def scores_per_defect(y_true, y_pred):
    """Recall and precision for each of the 8 defects."""
    recall = []
    precision = []

    for i in range(len(DEFECTS)):
        really_there = y_true[:, i] == 1
        model_said_yes = y_pred[:, i] == 1

        found = (really_there & model_said_yes).sum()          # said yes, and it was there
        missed = (really_there & ~model_said_yes).sum()        # said no, but it was there
        false_alarm = (~really_there & model_said_yes).sum()   # said yes, but it wasn't there

        # a defect that never shows up would divide by zero, so that gives 0
        if found + missed > 0:
            recall.append(found / (found + missed))
        else:
            recall.append(0.0)

        if found + false_alarm > 0:
            precision.append(found / (found + false_alarm))
        else:
            precision.append(0.0)

    return np.array(recall), np.array(precision)


def f1_per_defect(recall, precision):
    """F1 for each defect: only high when recall and precision both are, so saying yes to everything can't win."""
    f1 = []
    for r, p in zip(recall, precision):
        # never right once gives 0 + 0, so that gives 0 instead of dividing by zero
        if r + p > 0:
            f1.append(2 * p * r / (p + r))
        else:
            f1.append(0.0)
    return np.array(f1)


# The saved files, in the order the notebooks make them (02, 03, 04, 05, 06, 07)
MADE_IN_THIS_ORDER = ["output/split.npz", "output/features.npz", "output/simple_model.joblib",
                      "output/baseline_validation.csv", "output/forest.joblib", "output/timings.csv",
                      "output/test_predictions.npz", "output/generated.npz"]

def check_one_run():
    """Stop when the saved files don't come from one run of run_all.py, used by 08 and 09.

    One run makes them in pipeline order. If one notebook was run again by hand, its file is newer than the files
    made after it, and the numbers in 08 and 09 would mix two runs.
    """
    for earlier, later in zip(MADE_IN_THIS_ORDER, MADE_IN_THIS_ORDER[1:]):
        assert os.path.getmtime(earlier) <= os.path.getmtime(later), \
            f"{earlier} is newer than {later}: these files come from different runs. Run python run_all.py"


def share_all_eight_right(y_true, y_pred):
    """Share of wafers where all 8 positions are right (QC-02)."""
    # True when every position of that wafer matches
    correct = (y_true == y_pred).all(axis=1)
    return float(correct.mean())
