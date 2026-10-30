"""Fill visualiser.html with the real wafers of the last pipeline run.

An animated page to understand what the pipeline does to the data, in 12 scenes from the raw
file to a prediction (the raw file, one wafer, the label, 38 types, value 3, flattening, copies, the split, the
features, training, one prediction, scoring). Under each scene: what happens and the notebook line that does it.

Run the pipeline first (python run_all.py in pipeline/), then from inside this folder:  python build.py
It only reads: nothing in data/ or pipeline/output/ changes. It rewrites the one line in visualiser.html that holds
the data; open that page in a browser, it needs no internet.
"""
import json
import re
import runpy

import numpy as np
import joblib


def as_text(w):
    """A 52 x 52 map as one string of 2704 digits."""
    return "".join(str(int(v)) for v in w.ravel())


def story_data():
    wafer = runpy.run_path("../pipeline/wafer.py")
    DEFECTS = wafer["DEFECTS"]
    label_name = wafer["label_name"]
    features = wafer["features"]
    rng = np.random.default_rng(42)

    raw = np.load("../data/Wafer_Map_Datasets.npz")
    raw_maps = raw["arr_0"]
    raw_labels = raw["arr_1"]
    split = np.load("../pipeline/output/split.npz")
    maps, labels = split["maps"], split["labels"]
    train_idx, val_idx, test_idx = split["train_idx"], split["val_idx"], split["test_idx"]
    f = np.load("../pipeline/output/features.npz", allow_pickle=True)
    answers = np.load("../pipeline/output/test_predictions.npz")
    forest = joblib.load("../pipeline/output/forest.joblib")
    forest.set_params(n_jobs=1)

    has_3 = (raw_maps == 3).any(axis=(1, 2))
    flat = raw_maps.reshape(len(raw_maps), -1)
    _, inverse, counts = np.unique(flat, axis=0, return_inverse=True, return_counts=True)
    inverse = inverse.ravel()
    copies_per_wafer = counts[inverse]
    data = {"defects": DEFECTS, "feature_names": list(wafer["FEATURE_NAMES"])}

    # The raw file: how big, and a deck of real wafers with their labels
    deck = rng.choice(np.where(~has_3)[0], 12, replace=False)
    data["raw"] = {"wafers": len(raw_maps), "mb": round(raw_maps.nbytes / 1e6), "dtype": str(raw_maps.dtype),
                   "deck": [{"map": as_text(raw_maps[i]), "label": raw_labels[i].tolist()} for i in deck]}

    # The wafer that the story follows: a test wafer with two defects that the forest gets right
    y_test, y_pred = answers["y_test"], answers["y_pred"]
    right_two = np.where(((y_pred == y_test).all(axis=1)) & (y_test.sum(axis=1) == 2) & (y_test[:, 0] == 1))[0]
    k = right_two[0] if len(right_two) else np.where((y_pred == y_test).all(axis=1))[0][0]
    hero = maps[test_idx[k]]
    hero_label = labels[test_idx[k]]
    x = np.array([features(hero)])

    # Value 3: 45 normal wafers and 3 that have a 3, in a random order
    with_3 = rng.choice(np.where(has_3)[0], 3, replace=False)
    plain = rng.choice(np.where(~has_3 & (copies_per_wafer == 1))[0], 45, replace=False)
    pile = rng.permutation(np.concatenate([plain, with_3]))
    data["value3"] = {"before": len(raw_maps), "after": int((~has_3).sum()),
                      "pile": [{"map": as_text(raw_maps[i]), "has3": bool(has_3[i])} for i in pile]}

    # Copies: 12 wafers that are there once, and 2 that are there twice, in file order
    groups = np.where(counts == 2)[0]
    pairs = []
    for g in rng.permutation(groups):
        members = np.where(inverse == g)[0]
        if not has_3[members].any():
            pairs.extend(members.tolist())
        if len(pairs) == 4:
            break
    once = rng.choice(np.where(~has_3 & (copies_per_wafer == 1))[0], 12, replace=False)
    rows = np.sort(np.concatenate([once, pairs]))
    data["copies"] = {"before": int((~has_3).sum()), "after": len(maps),
                      "rows": [{"index": int(i), "map": as_text(raw_maps[i])} for i in rows]}

    # The 38 types
    types = np.unique(labels, axis=0)
    gallery = []
    for combo in types:
        members = np.where((labels == combo).all(axis=1))[0]
        gallery.append({"name": label_name(combo), "label": combo.tolist(), "count": len(members),
                        "map": as_text(maps[members[0]])})
    data["types"] = gallery

    # The split of the hero's type
    is_type = (labels == hero_label).all(axis=1)
    n = int(is_type.sum())
    data["split"] = {"type": label_name(hero_label), "n": n, "n_train": int(n * 0.70), "n_val": int(n * 0.15),
                     "train": len(train_idx), "val": len(val_idx), "test": len(test_idx),
                     "order": rng.permutation(100).tolist()}

    # Features: for every one of the 14 areas, how many dies and how many failed
    areas = [wafer["EVERYWHERE"]] + [wafer["RING"] == r for r in range(5)] + [wafer["SECTOR"] == s for s in range(8)]
    steps = []
    for area in areas:
        dies = (hero > 0) & area
        steps.append({"dies": int(dies.sum()), "failed": int((hero[dies] == 2).sum()), "squares": int(area.sum())})
    data["features"] = {"ring": as_text(wafer["RING"]), "slice": as_text(wafer["SECTOR"]), "steps": steps,
                        "values": [round(v, 3) for v in x[0]],
                        "X_train": np.round(f["X_train"][:10], 2).tolist(), "y_train": f["y_train"][:10].tolist(),
                        "rows": len(f["X_train"])}

    # Training: the top of the first tree
    first = forest.estimators_[0].tree_

    def node(n, depth):
        if first.children_left[n] == -1 or depth == 3:
            return {"samples": int(first.n_node_samples[n])}
        return {"q": f"{data['feature_names'][first.feature[n]]} <= {first.threshold[n]:.3f}",
                "samples": int(first.n_node_samples[n]),
                "yes": node(first.children_left[n], depth + 1), "no": node(first.children_right[n], depth + 1)}

    data["tree"] = {"trees": len(forest.estimators_), "top": node(0, 0)}

    # One prediction: the path through the first tree, and what each of the 300 trees says
    path = forest.estimators_[0].decision_path(x).indices
    steps = []
    for n in path[:-1]:
        feature = first.feature[n]
        steps.append({"feature": data["feature_names"][feature], "value": round(float(x[0][feature]), 3),
                      "threshold": round(float(first.threshold[n]), 3),
                      "yes": bool(x[0][feature] <= first.threshold[n])})
    per_tree = np.array([tree.predict(x)[0] for tree in forest.estimators_]).astype(int)
    data["predict"] = {"map": as_text(hero), "truth": hero_label.tolist(), "said": forest.predict(x)[0].tolist(),
                       "name": label_name(hero_label), "path": steps, "depth": len(path) - 1,
                       "votes": ["".join(str(v) for v in per_tree[:, i]) for i in range(8)]}
    data["hero"] = {"map": as_text(hero), "label": hero_label.tolist(), "name": label_name(hero_label)}

    # Scores: the four boxes for every defect, on the whole test part
    scores = []
    for i in range(8):
        there, said = y_test[:, i] == 1, y_pred[:, i] == 1
        scores.append({"found": int((there & said).sum()), "missed": int((there & ~said).sum()),
                       "false_alarm": int((~there & said).sum()), "rest": int((~there & ~said).sum())})
    data["scores"] = {"per_defect": scores, "n": len(y_test)}
    return data


if __name__ == "__main__":
    print("Collecting the wafers for the animation")
    data = json.dumps(story_data(), separators=(",", ":"))

    # The page keeps its data on one line, const DATA = ...; only that line is replaced
    with open("visualiser.html", encoding="utf-8") as file:
        html = file.read()
    html = re.sub(r"^const DATA = .*;$", lambda m: f"const DATA = {data};", html, count=1, flags=re.M)
    with open("visualiser.html", "w", encoding="utf-8") as file:
        file.write(html)
    print(f"Updated visualiser.html ({len(html) / 1e6:.1f} MB). Open it in a browser.")
