#!/usr/bin/env python3
"""
Preprocess script for sidewalk accessibility data.
Loads votes, computes tallies with uncertainty policies.
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import json

# Constants
CLASS3 = ['no', 'unsure', 'yes']
AIDS = ["Walking cane", "Walker", "Mobility scooter", "Manual wheelchair", "Motorized wheelchair"]

def load_and_normalize_votes(votes_path):
    df = pd.read_csv(votes_path)
    df['Selection'] = (
        df['Selection']
        .astype(str).str.strip().str.lower()
        .replace({'maybe': 'unsure', 'cannot': 'no', 'cannot_go': 'no', 'n': 'no', 'y': 'yes'})
    )
    need_cols = ['ImageID', 'MobilityAid', 'Selection']
    df = df.dropna(subset=need_cols)[need_cols].copy()
    return df

def compute_tallies(df_votes, ent_tau=1.00, margin_tau=0.10, min_votes=3, target_mode="hard_relabel", weight_mode="inv_entropy"):
    tallies = []
    for aid in AIDS:
        df_aid = df_votes[df_votes['MobilityAid'] == aid].copy()
        if df_aid.empty:
            continue
        grouped = df_aid.groupby('ImageID')['Selection'].value_counts().unstack(fill_value=0)
        grouped = grouped.reindex(columns=CLASS3, fill_value=0)
        grouped['total_votes'] = grouped.sum(axis=1)
        grouped['p_no'] = grouped['no'] / grouped['total_votes']
        grouped['p_unsure'] = grouped['unsure'] / grouped['total_votes']
        grouped['p_yes'] = grouped['yes'] / grouped['total_votes']
        grouped['entropy'] = - (grouped[['p_no', 'p_unsure', 'p_yes']] * np.log(grouped[['p_no', 'p_unsure', 'p_yes']] + 1e-12)).sum(axis=1)
        probs = grouped[['p_no', 'p_unsure', 'p_yes']].values
        top2 = np.sort(probs, axis=1)[:, -2:]
        grouped['margin'] = top2[:, 1] - top2[:, 0]
        grouped['is_uncertain'] = (grouped['entropy'] > ent_tau) | (grouped['margin'] < margin_tau) | (grouped['total_votes'] < min_votes)
        grouped['argmax_label'] = grouped[['p_no', 'p_unsure', 'p_yes']].idxmax(axis=1)
        grouped['MobilityAid'] = aid
        grouped['ImageID'] = grouped.index
        tallies.append(grouped.reset_index(drop=True))
    tallies_df = pd.concat(tallies, ignore_index=True)
    # Apply target mode
    if target_mode == "hard_relabel":
        tallies_df['label_name'] = np.where(tallies_df['is_uncertain'], 'unsure', tallies_df['argmax_label'])
    elif target_mode == "hard_drop":
        tallies_df = tallies_df[~tallies_df['is_uncertain']].copy()
        tallies_df['label_name'] = tallies_df['argmax_label']
    else:
        tallies_df['label_name'] = None
    if 'label_name' in tallies_df.columns and tallies_df['label_name'].notna().any():
        tallies_df['label'] = tallies_df['label_name'].map({c: i for i, c in enumerate(CLASS3)})
    # Weights
    if weight_mode == "inv_entropy":
        max_ent = np.log(3.0)
        norm_ent = np.clip(tallies_df['entropy'] / max_ent, 0.0, 1.0)
        tallies_df['sample_weight'] = 1.0 - norm_ent
    else:
        tallies_df['sample_weight'] = 1.0
    return tallies_df

def main():
    parser = argparse.ArgumentParser(description="Preprocess sidewalk accessibility data.")
    parser.add_argument("--votes_csv", type=str, required=True, help="Path to image_selection.csv")
    parser.add_argument("--output", type=str, required=True, help="Output JSON file for tallies")
    parser.add_argument("--ent_tau", type=float, default=1.00)
    parser.add_argument("--margin_tau", type=float, default=0.10)
    parser.add_argument("--min_votes", type=int, default=3)
    parser.add_argument("--target_mode", choices=["hard_relabel", "hard_drop", "soft"], default="hard_relabel")
    parser.add_argument("--weight_mode", choices=["none", "inv_entropy", "votes"], default="inv_entropy")
    args = parser.parse_args()

    df_votes = load_and_normalize_votes(args.votes_csv)
    tallies = compute_tallies(df_votes, args.ent_tau, args.margin_tau, args.min_votes, args.target_mode, args.weight_mode)
    tallies.to_json(args.output, orient='records')
    print(f"Tallies saved to {args.output}")

if __name__ == "__main__":
    main()