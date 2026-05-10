#!/usr/bin/env python3
"""
Convert Firebase raw JSONL dump (from surveyAnswers0727 collection)
into image_selection.csv matching the format used in the paper:

    PId, MobilityAid, ImageSet, ImageID, ImageType, Selection

The survey stores yes/no/unsure votes implicitly via subgroups:
  imageSelections → groupN → {groupNA: [...], groupNB: [...]}

  groupNA = images where user answered YES or UNSURE
  groupNB = images where user answered NO  or UNSURE

  Image in A only  → YES
  Image in B only  → NO
  Image in both    → UNSURE

Usage:
    python src/data/firebase.py \
        --input  data/raw/firebase_raw.jsonl \
        --output data/processed/image_selection_firebase.csv \
        --merge  data/processed/image_selection.csv
"""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

# Map Firebase mobilityAid strings → paper column names
AID_MAP = {
    "manual wheelchair":    "Manual wheelchair",
    "manual_wheelchair":    "Manual wheelchair",
    "motorized wheelchair": "Motorized wheelchair",
    "motorized_wheelchair": "Motorized wheelchair",
    "mobility scooter":     "Mobility scooter",
    "mobility_scooter":     "Mobility scooter",
    "walker":               "Walker",
    "walking cane":         "Walking cane",
    "walking_cane":         "Walking cane",
    "cane":                 "Walking cane",
}

# group0..group8 → ImageSet integer
GROUP_TO_SET = {f"group{i}": i for i in range(9)}

# LabelTypeID → ImageType string (from Project Sidewalk schema)
LABEL_TYPE_MAP = {
    1: "CurbRamp",
    2: "NoCurbRamp",
    3: "Obstacle",
    4: "SurfaceProblem",
    5: "NoSidewalk",
    6: "Crosswalk",
    7: "Signal",
}


def extract_value(field: dict) -> object:
    """Unwrap a Firestore field value dict to a Python value."""
    if "stringValue"  in field: return field["stringValue"]
    if "integerValue" in field: return int(field["integerValue"])
    if "booleanValue" in field: return field["booleanValue"]
    if "doubleValue"  in field: return field["doubleValue"]
    if "arrayValue"   in field:
        return [extract_value(v) for v in field["arrayValue"].get("values", [])]
    if "mapValue"     in field:
        return {k: extract_value(v) for k, v in field["mapValue"].get("fields", {}).items()}
    if "nullValue"    in field: return None
    return None


def parse_doc(doc: dict) -> list[dict]:
    """
    Parse one Firestore document into image_selection rows.

    The survey encodes yes/no/unsure implicitly:
      groupNA = images answered YES or UNSURE (passed to pairwise comparison group A)
      groupNB = images answered NO  or UNSURE (passed to pairwise comparison group B)

    To recover the original answer:
      LabelID in A only  → YES
      LabelID in B only  → NO
      LabelID in A and B → UNSURE
    """
    fields = {k: extract_value(v) for k, v in doc.get("fields", {}).items()}

    # Skip incomplete/temp sessions — only process sessions that answered at least one group
    # (logType "final" is complete; "CompletedOneMobilityAid" also has usable data)
    log_type = str(fields.get("logType", "")).lower()

    # Participant identifier
    doc_id  = doc.get("name", "").split("/")[-1]
    user_id = fields.get("userId") or fields.get("sessionId") or doc_id

    # Mobility aid
    raw_aid = str(fields.get("mobilityAid", "")).strip().lower()
    aid     = AID_MAP.get(raw_aid)
    if not aid:
        return []

    image_selections = fields.get("imageSelections")
    if not isinstance(image_selections, dict):
        return []

    rows: list[dict] = []

    for group_key, group_val in image_selections.items():
        image_set = GROUP_TO_SET.get(group_key, -1)
        if not isinstance(group_val, dict):
            continue

        # Collect LabelIDs in subgroup A and subgroup B
        key_a = f"{group_key}A"
        key_b = f"{group_key}B"
        entries_a = group_val.get(key_a) or []
        entries_b = group_val.get(key_b) or []

        if not isinstance(entries_a, list):
            entries_a = []
        if not isinstance(entries_b, list):
            entries_b = []

        # Skip groups where BOTH subgroups are empty (user didn't reach this group)
        if not entries_a and not entries_b:
            continue

        def label_ids(entries):
            ids = {}
            for e in entries:
                if not isinstance(e, dict):
                    continue
                lid = e.get("LabelID") or e.get("labelId") or e.get("imageId")
                if lid is not None:
                    ids[int(lid)] = e
            return ids

        ids_a = label_ids(entries_a)
        ids_b = label_ids(entries_b)
        all_ids = set(ids_a) | set(ids_b)

        for lid in all_ids:
            in_a = lid in ids_a
            in_b = lid in ids_b

            if in_a and in_b:
                selection = "Unsure"
            elif in_a:
                selection = "Yes"
            else:
                selection = "No"

            # ImageType from LabelTypeID
            entry = ids_a.get(lid) or ids_b.get(lid) or {}
            label_type_id = entry.get("LabelTypeID") or entry.get("labelTypeId")
            if label_type_id is not None:
                image_type = LABEL_TYPE_MAP.get(int(label_type_id), "Unknown")
            else:
                image_type = "Unknown"

            rows.append({
                "PId":        user_id,
                "MobilityAid": aid,
                "ImageSet":   image_set,
                "ImageID":    lid,
                "ImageType":  image_type,
                "Selection":  selection,
            })

    return rows


def load_jsonl(path: Path) -> list[dict]:
    docs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    docs.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return docs


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Firebase JSONL to image_selection.csv")
    parser.add_argument("--input",  required=True, help="Path to firebase_raw.jsonl")
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument("--merge",  default=None,
                        help="Existing image_selection.csv to merge with (deduplicates)")
    args = parser.parse_args()

    print(f"Loading {args.input} …")
    docs = load_jsonl(Path(args.input))
    print(f"  {len(docs):,} documents loaded")

    all_rows: list[dict] = []
    skipped = 0
    for doc in docs:
        rows = parse_doc(doc)
        if rows:
            all_rows.extend(rows)
        else:
            skipped += 1

    print(f"  Parsed: {len(all_rows):,} vote rows  |  skipped: {skipped:,} docs (no usable data)")

    # Deduplicate on (PId, MobilityAid, ImageID) — keep first occurrence
    seen:    set[tuple] = set()
    deduped: list[dict] = []
    for row in all_rows:
        key = (str(row["PId"]), row["MobilityAid"], row["ImageID"])
        if key not in seen:
            seen.add(key)
            deduped.append(row)
    print(f"  After dedup: {len(deduped):,} rows")

    # Merge with existing CSV
    existing: list[dict] = []
    if args.merge and Path(args.merge).exists():
        with open(args.merge) as f:
            existing = list(csv.DictReader(f))
        existing_keys = {(str(r["PId"]), r["MobilityAid"], int(r["ImageID"])) for r in existing}
        print(f"  Existing CSV: {len(existing):,} rows")

        new_only = [r for r in deduped
                    if (str(r["PId"]), r["MobilityAid"], r["ImageID"]) not in existing_keys]
        combined = existing + new_only
        print(f"  New rows added from Firebase: {len(new_only):,}")
    else:
        combined = deduped

    # Stats
    unique_images = len({int(r["ImageID"]) for r in combined})
    unique_pids   = len({str(r["PId"])     for r in combined})
    aid_counts    = Counter(r["MobilityAid"] for r in combined)
    sel_counts    = Counter(r["Selection"]   for r in combined)

    print(f"\n── Final dataset ──────────────────────────────")
    print(f"  Total rows     : {len(combined):,}")
    print(f"  Unique images  : {unique_images}")
    print(f"  Unique raters  : {unique_pids}")
    print(f"  By mobility aid: {dict(aid_counts)}")
    print(f"  By selection   : {dict(sel_counts)}")
    print(f"───────────────────────────────────────────────")

    # Write output
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["PId", "MobilityAid", "ImageSet", "ImageID", "ImageType", "Selection"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(combined)

    print(f"\n✓ Saved → {out_path}")
    print(f"\nNext step:")
    print(f"  python src/data/preprocess.py --votes_csv {out_path} \\")
    print(f"      --output data/processed/tallies_firebase.json")


if __name__ == "__main__":
    main()
