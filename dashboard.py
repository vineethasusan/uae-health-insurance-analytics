"""Investigator and bed-manager views."""
import json
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="UAE Health Insurance & Hospital Analytics", layout="wide")
st.title("UAE Health Insurance & Hospital Analytics")
m = json.loads(Path("outputs/metrics.json").read_text())
f, b = m["fwa"], m["beds"]

tab1, tab2 = st.tabs(["Claims FWA review queue", "Bed occupancy forecast"])
with tab1:
    c = st.columns(4)
    c[0].metric("FWA AED caught (5% audit capacity)", f"{f['model']['share_of_fwa_aed']:.0%}", f"{f['model']['share_of_fwa_aed'] - f['rules']['share_of_fwa_aed']:+.0%} vs rules")
    c[1].metric("Queue precision", f"{f['model']['precision']:.0%}")
    c[2].metric("Net saving (4 months)", f"AED {f['model']['net_saving_aed']:,}")
    c[3].metric("Claims reviewed", f"{f['model']['claims_reviewed']:,}")
    q = pd.read_csv("outputs/fwa_review_queue.csv")
    st.dataframe(q, hide_index=True, use_container_width=True)
with tab2:
    c = st.columns(3)
    c[0].metric("7-day forecast error", f"{b['mae_beds']['fc_model']} beds", f"{b['mae_beds']['fc_model'] - b['mae_beds']['fc_persistence']:+.1f} vs same-as-today", delta_color="inverse")
    c[1].metric("High-occupancy alert recall", f"{b['alert_recall']:.0%}")
    c[2].metric("Median length of stay", f"{b['median_los_days']} days")
    st.markdown(Path("outputs/report.md").read_text().split("## 2.")[1])
