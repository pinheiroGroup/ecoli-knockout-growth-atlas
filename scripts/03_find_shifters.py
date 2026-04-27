#!/usr/bin/env python3
import pandas as pd

lb  = pd.read_csv("results/clusters_lb.csv") \
        .rename(columns={"Cluster": "cluster_lb",  "Well": "gene"})[["gene","cluster_lb"]]
m63 = pd.read_csv("results/clusters_m63.csv") \
        .rename(columns={"Cluster": "cluster_m63", "Well": "gene"})[["gene","cluster_m63"]]

joined   = lb.merge(m63, on="gene")
shifters = joined[joined["cluster_lb"] != joined["cluster_m63"]]

joined.to_csv("results/cluster_assignments_both.csv",   index=False)
shifters.to_csv("results/cluster_shifters.csv",          index=False)
print(f"Total genes: {len(joined)}")
print(f"Shifters:    {len(shifters)}  ({100*len(shifters)/len(joined):.1f}%)")
