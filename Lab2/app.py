"""
app.py — Streamlit dashboard for Lab 2 (CSP gas allocation)
============================================================

Run with:
    cd /home/ubuntu/Papry\\ Edu/1.\\ AI/Lab/Lab2
    /home/ubuntu/Papry\\ Edu/1.\\ AI/Lab/ml_env/bin/python -m streamlit run app.py

The same solvers from csp_main.py are used here — this file is purely UI.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from gas_world import build_world
from csp_model import ALL_CONSTRAINTS, Policy, build_csp
from backtracking import backtracking_search
from min_conflicts import min_conflicts
from visualize import (
    plot_assignment_map,
    plot_comparison_bars,
    plot_convergence,
    plot_station_loads,
)


# ---------------------------------------------------------------------------

BBOX_PRESETS: dict[str, tuple[float, float, float, float] | None] = {
    "dhanmondi":   (23.735, 23.770, 90.360, 90.395),
    "mohakhali":   (23.770, 23.810, 90.395, 90.430),
    "mirpur":      (23.795, 23.835, 90.345, 90.385),
    "old_dhaka":   (23.700, 23.730, 90.395, 90.430),
    "full_dhaka":  None,
}

CONSTRAINT_LABELS: dict[str, str] = {
    "U1_reachability":     "U1 — Reachability (jam + road-size aware)",
    "U2_emergency_only":   "U2 — Emergency-only access",
    "G1_plate_parity":     "G1 — Plate-parity rationing",
    "G3_public_reserve":   "G3' — Public transport reservation (60%)",
    "G4_strategic_reserve":"G4 — Strategic reserve",
    "G6_peak_hour_cng":    "G6' — Peak-hour CNG closure",
    "G8_queue_cap":        "G8 — Queue cap",
    "G9_min_service":      "G9 — Min service quota",
    "B1_household":        "B1 — Same-household → different station (binary)",
}

STREAMLIT_OUT = Path("cache/streamlit")
STREAMLIT_OUT.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# World cache — rebuilding the distance oracle is the expensive step.
# st.cache_resource keys on the scalar args so a slider change in policy
# (which doesn't affect the world) does NOT rebuild.
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Rebuilding distance oracle (only on world-scope change)…")
def get_world(n_users: int, time_slot: str, seed: int, bbox_name: str, max_stations: int):
    bbox = BBOX_PRESETS[bbox_name]
    return build_world(
        n_users=n_users, time_slot=time_slot, seed=seed,
        bbox=bbox, max_stations=max_stations,
        area_name=bbox_name,
    )


# ---------------------------------------------------------------------------
# Page header
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Lab 2 — CSP Gas Allocation",
    page_icon="⛽",
    layout="wide",
)

st.title("⛽ CSP Gas Allocation — Dhaka")
st.caption(
    "Method 1 (Backtracking + MRV + Degree + LCV + FC) vs "
    "Method 2 (Min-Conflicts local search). "
    "Reuses Lab 1's Dhaka OSM graph + distance machinery."
)


# ---------------------------------------------------------------------------
# Sidebar — scenario, constraint toggles, policy knobs
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("🗺️  Scenario (Method 1 / BT scope)")
    n_users = st.slider("Number of users", 2, 200, 15, step=1)
    bbox_name = st.selectbox("Area preset", list(BBOX_PRESETS.keys()), index=0)
    max_stations = st.slider("Max stations after bbox", 2, 40, 8)
    time_slot = st.selectbox(
        "Time slot (affects jam + G6')",
        ["07:00-10:00", "10:00-12:00", "15:00-17:00", "17:00-21:00", "21:00-06:00"],
        index=1,
    )
    seed = st.number_input("Seed", value=42, step=1, min_value=0, max_value=999_999)

    st.header("🎯 Method 2 (Min-Conflicts) scope")
    scope_mode = st.radio(
        "Run Min-Conflicts on…",
        ["Same as BT (fair head-to-head)",
         "Full Dhaka (slide-aligned — MC on the whole map)"],
        index=0,
        help="Teacher's intent is the second option: BT explores a small portion of "
             "the network while Min-Conflicts handles the whole city. The first "
             "option makes both methods comparable on identical conditions."
    )
    mc_use_full = scope_mode.startswith("Full Dhaka")
    if mc_use_full:
        mc_n_users = st.slider("MC: users on full Dhaka", 50, 400, 150, step=10)
        mc_max_stations = st.slider("MC: stations on full Dhaka", 10, 60, 30)
    else:
        mc_n_users = n_users
        mc_max_stations = max_stations

    if st.button("🔄 Force rebuild world(s)"):
        st.cache_resource.clear()
        st.rerun()

    st.header("🚦  Active constraints")
    enabled: dict[str, bool] = {}
    for name in ALL_CONSTRAINTS:
        enabled[name] = st.checkbox(CONSTRAINT_LABELS[name], value=True, key=f"c_{name}")

    st.header("⚖️  Policy")
    parity = st.selectbox("Plate-parity today", ["any", "odd", "even"])
    pub_frac = st.slider("G3' public reserve fraction", 0.0, 0.95, 0.60, 0.05)
    strat_frac = st.slider("G4 strategic reserve fraction", 0.0, 0.5, 0.10, 0.05)
    queue_cap = st.slider("G8 queue cap (max users / station)", 1, 100, 8)
    min_quota = st.slider("G9 min service quota", 0, 20, 0)
    hh_frac = st.slider("B1 household pairing fraction", 0.0, 0.6, 0.20, 0.05)


# ---------------------------------------------------------------------------
# Build world(s) + CSP
#
# When scope_mode = "Same as BT", a single world is shared by both solvers.
# When scope_mode = "Full Dhaka", we build a separate full-Dhaka world for MC.
# ---------------------------------------------------------------------------

with st.spinner(f"Building BT-scope world (n_users={n_users}, bbox={bbox_name})..."):
    bt_world = get_world(int(n_users), time_slot, int(seed), bbox_name, int(max_stations))

policy = Policy(
    today_parity=parity,
    public_reserve_fraction=float(pub_frac),
    strategic_reserve_fraction=float(strat_frac),
    queue_cap=int(queue_cap),
    min_service_quota=int(min_quota),
    household_fraction=float(hh_frac),
)
bt_csp = build_csp(bt_world, policy=policy, enabled=enabled, household_seed=int(seed))

if mc_use_full:
    with st.spinner(f"Building MC-scope world (n_users={mc_n_users}, full Dhaka)..."):
        mc_world = get_world(int(mc_n_users), time_slot, int(seed), "full_dhaka", int(mc_max_stations))
    mc_csp = build_csp(mc_world, policy=policy, enabled=enabled, household_seed=int(seed))
else:
    mc_world = bt_world
    mc_csp = bt_csp

# Back-compat aliases so the rest of the file keeps working
world = bt_world
csp = bt_csp

empty_domains = sum(1 for d in bt_csp.domains.values() if not d)
feasible_users = len(bt_csp.users) - empty_domains

# ---------------------------------------------------------------------------
# Top metrics
# ---------------------------------------------------------------------------

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("BT stations", len(bt_world.stations))
c2.metric("BT users", len(bt_world.users))
c3.metric("BT feasible (AC-3)", f"{feasible_users} / {len(bt_world.users)}")
c4.metric("MC stations", len(mc_world.stations))
c5.metric("MC users", len(mc_world.users))

mode_caption = (
    "🎯 **Slide-aligned scope**: BT solves a small portion of the network, "
    "Min-Conflicts solves the whole Dhaka map."
    if mc_use_full else
    "⚖️  **Same-condition scope**: both methods see the same world for a fair head-to-head."
)
st.caption(mode_caption)


# ---------------------------------------------------------------------------
# Run both solvers
# ---------------------------------------------------------------------------

with st.spinner("Method 1: Backtracking + MRV + Degree + LCV + FC..."):
    bt = backtracking_search(
        bt_csp, use_mrv=True, use_lcv=True, use_fc=True, time_limit_ms=15_000,
    )
with st.spinner(f"Method 2: Min-Conflicts on {'full Dhaka' if mc_use_full else 'BT scope'}..."):
    mc = min_conflicts(
        mc_csp, max_steps=5000, max_restarts=3, seed=int(seed), time_limit_ms=15_000,
    )


# ---------------------------------------------------------------------------
# Top-of-page method comparison metrics
# (How many users got the same vs different station across methods)
# ---------------------------------------------------------------------------

# Same/disagree counts only make sense in "same scope" mode where both methods
# see the same users. In slide-aligned mode the user sets are disjoint by design.
if not mc_use_full:
    m1_set = set(bt.assignment)
    m2_set = set(mc.assignment)
    both_served = m1_set & m2_set
    agree_count = sum(1 for uid in both_served if bt.assignment[uid] == mc.assignment[uid])
    disagree_count = len(both_served) - agree_count
    only_m1 = len(m1_set - m2_set)
    only_m2 = len(m2_set - m1_set)
    unassigned_by_both = len(bt_world.users) - len(m1_set | m2_set)

    st.markdown("### How the two methods compare on this scenario")
    mm1, mm2, mm3, mm4, mm5 = st.columns(5)
    mm1.metric("Both agree (same station)", agree_count,
               help="Users that BOTH methods assigned to the exact same station.")
    mm2.metric("Both serve, disagree", disagree_count,
               help="Users served by both methods but to different stations.")
    mm3.metric("Only Method 1 served", only_m1)
    mm4.metric("Only Method 2 served", only_m2)
    mm5.metric("Served by neither", unassigned_by_both,
               help="Includes users whose domain was empty after AC-3 (infeasible).")
else:
    st.markdown("### Headline numbers per method")
    mm1, mm2, mm3, mm4 = st.columns(4)
    mm1.metric("M1 served (small portion)",
               f"{len(bt.assignment)} / {len(bt_world.users)}")
    mm2.metric("M2 served (full Dhaka)",
               f"{len(mc.assignment)} / {len(mc_world.users)}")
    mm3.metric("M1 cost", f"{bt.cost:.0f}" if bt.found else "—")
    mm4.metric("M2 cost", f"{mc.cost:.0f}" if mc.found else "—")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _assignment_df(w, assignment: dict[int, int]) -> pd.DataFrame:
    rows = []
    for uid in sorted(assignment):
        sid = assignment[uid]
        u = w.users[uid]
        s = w.stations[sid]
        d = w.distance(u, s)
        rows.append({
            "user": uid,
            "vehicle": u.vehicle_class,
            "plate": u.plate_parity,
            "zone": u.zone,
            "station_id": sid,
            "station_name": s.name,
            "meters": int(round(d)),
        })
    return pd.DataFrame(rows)


def _render_method(label: str, result, w, prefix: str, plot_bbox=None, is_mc: bool = False):
    cc1, cc2, cc3, cc4 = st.columns(4)
    cc1.metric("Found", "yes" if result.found else "no")
    cc2.metric("Cost", f"{result.cost:.0f}" if result.found else "—")
    cc3.metric("Served", f"{len(result.assignment)} / {len(w.users)}")
    cc4.metric("Runtime", f"{result.runtime_ms:.1f} ms")

    if is_mc:
        st.caption(f"iterations={result.iterations}  restarts={result.restarts}  "
                   f"final_violations={result.final_violations}")
    else:
        st.caption(f"expanded={result.nodes_expanded}  backtracks={result.backtracks}  "
                   f"unassignable={len(result.unassignable)}")

    map_path = str(STREAMLIT_OUT / f"{prefix}_map.png")
    loads_path = str(STREAMLIT_OUT / f"{prefix}_loads.png")
    plot_assignment_map(
        w, result.assignment, result.unassignable,
        title=f"{label}  |  cost={result.cost:.0f}  |  served={len(result.assignment)}/{len(w.users)}",
        output_path=map_path,
        bbox=plot_bbox,
    )
    plot_station_loads(
        w, result.assignment,
        policy_strategic_frac=float(strat_frac),
        policy_public_frac=float(pub_frac),
        output_path=loads_path,
    )

    img_col, side_col = st.columns([2, 1])
    img_col.image(map_path, caption=f"{label} — assignment map")
    side_col.image(loads_path, caption="Station loads")

    if is_mc and result.convergence:
        conv_path = str(STREAMLIT_OUT / f"{prefix}_convergence.png")
        plot_convergence(result.convergence, output_path=conv_path)
        st.image(conv_path, caption="Min-Conflicts convergence (violations vs iter)")

    with st.expander(f"Per-user assignment ({label})"):
        df = _assignment_df(w, result.assignment)
        st.dataframe(df, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Method 1 — Backtracking + 3 heuristics",
    "Method 2 — Min-Conflicts (local search)",
    "⚖️  Same-condition compare (M1 vs M2)",
    "🔍  Per-user analysis",
    "🧪  Heuristic ablation (naive → improved)",
])

bt_bbox = BBOX_PRESETS[bbox_name]
mc_bbox = None if mc_use_full else BBOX_PRESETS[bbox_name]

with tab1:
    st.subheader("Slide H1 (MRV) + H2 (Degree) + H3 (LCV) + Forward Checking")
    st.caption(f"Scope: {bbox_name}  |  users={len(bt_world.users)}  |  stations={len(bt_world.stations)}")
    _render_method("Method 1 (BT)", bt, bt_world, prefix="m1", plot_bbox=bt_bbox, is_mc=False)

with tab2:
    st.subheader("Slide page 40–42 — Min-Conflicts local search heuristic")
    scope_label = "full Dhaka" if mc_use_full else bbox_name
    st.caption(f"Scope: {scope_label}  |  users={len(mc_world.users)}  |  stations={len(mc_world.stations)}")
    _render_method("Method 2 (Min-Conflicts)", mc, mc_world, prefix="m2", plot_bbox=mc_bbox, is_mc=True)

with tab3:
    if mc_use_full:
        st.subheader("Side-by-side maps (different scopes)")
        st.info(
            "🎯 **Slide-aligned mode is on** — the two methods run on different worlds "
            "(BT on the small bbox you picked, MC on the whole Dhaka map), so a "
            "per-user comparison isn't meaningful here. Switch the sidebar to "
            "*Same as BT* if you want the head-to-head fair comparison."
        )
        col_a, col_b = st.columns(2)
        col_a.image(str(STREAMLIT_OUT / "m1_map.png"),
                    caption=f"Method 1 (BT) — {bbox_name}, {len(bt_world.users)} users")
        col_b.image(str(STREAMLIT_OUT / "m2_map.png"),
                    caption=f"Method 2 (Min-Conflicts) — full Dhaka, {len(mc_world.users)} users")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("M1 cost", f"{bt.cost:.0f}" if bt.found else "—")
        c2.metric("M2 cost", f"{mc.cost:.0f}" if mc.found else "—")
        c3.metric("M1 runtime", f"{bt.runtime_ms:.1f} ms")
        c4.metric("M2 runtime", f"{mc.runtime_ms:.1f} ms")
    else:
        st.subheader("Same scenario, both methods — direct head-to-head")
        st.caption(
            "Both solvers receive the **same world, same users, same constraints, same seed** — "
            "the only thing that differs is the algorithm. This is the fair comparison."
        )

        # side-by-side maps
        col_a, col_b = st.columns(2)
        col_a.image(str(STREAMLIT_OUT / "m1_map.png"),
                    caption="Method 1 (BT + MRV + Degree + LCV + FC)")
        col_b.image(str(STREAMLIT_OUT / "m2_map.png"),
                    caption="Method 2 (Min-Conflicts)")

        # cost / runtime bars
        cost_path = str(STREAMLIT_OUT / "compare_cost.png")
        runtime_path = str(STREAMLIT_OUT / "compare_runtime.png")
        plot_comparison_bars(
            {"Method 1 (BT)": bt.cost, "Method 2 (MC)": mc.cost},
            title="Total cost (lower is better)", ylabel="Cost",
            output_path=cost_path,
        )
        plot_comparison_bars(
            {"Method 1 (BT)": bt.runtime_ms, "Method 2 (MC)": mc.runtime_ms},
            title="Runtime", ylabel="ms",
            output_path=runtime_path,
        )
        cmp_a, cmp_b = st.columns(2)
        cmp_a.image(cost_path)
        cmp_b.image(runtime_path)

        # per-user "where did each method send them?"
        df_m1 = _assignment_df(bt_world, bt.assignment).rename(
            columns={"station_id": "M1_station", "station_name": "M1_name", "meters": "M1_m"}
        )[["user", "vehicle", "plate", "zone", "M1_station", "M1_name", "M1_m"]]
        df_m2 = _assignment_df(mc_world, mc.assignment).rename(
            columns={"station_id": "M2_station", "station_name": "M2_name", "meters": "M2_m"}
        )[["user", "M2_station", "M2_name", "M2_m"]]
        merged = df_m1.merge(df_m2, on="user", how="outer")
        merged["same?"] = merged["M1_station"] == merged["M2_station"]

        st.markdown("**Per-user comparison** — which station each user got from each method:")
        st.dataframe(merged, use_container_width=True, hide_index=True)

        differ = merged[merged["same?"] == False]
        same = merged[merged["same?"] == True]
        st.caption(
            f"📊  {len(same)} users went to the same station in both methods · "
            f"{len(differ)} users got a different station"
        )


# ---------------------------------------------------------------------------
# Tab 4 — single-user drill-down
# ---------------------------------------------------------------------------

with tab4:
    st.subheader("Trace one user")
    if mc_use_full:
        st.info(
            "🎯 Slide-aligned mode: BT and MC see different users entirely. "
            "Pick which world's user you want to inspect; cross-method comparison "
            "only makes sense in *Same as BT* scope mode."
        )
        which_world = st.radio(
            "Inspect user from…",
            [f"Method 1 world ({bbox_name}, {len(bt_world.users)} users)",
             f"Method 2 world (full Dhaka, {len(mc_world.users)} users)"],
            horizontal=True,
        )
        is_m1 = which_world.startswith("Method 1")
        active_world = bt_world if is_m1 else mc_world
        active_csp = bt_csp if is_m1 else mc_csp
        active_result = bt if is_m1 else mc
        active_method_label = "Method 1 — Backtracking" if is_m1 else "Method 2 — Min-Conflicts"
    else:
        st.caption(
            "Pick a user and see: their attributes, their domain after AC-3, "
            "which station each method assigned them, and their household neighbours (B1)."
        )
        active_world = bt_world  # same as mc_world here
        active_csp = bt_csp
        active_result = None  # show both
        active_method_label = None

    if not active_world.users:
        st.warning("No users to analyse.")
    else:
        labels = []
        for u in active_world.users:
            d_size = len(active_csp.domains[u.id])
            tag = " (∅ infeasible)" if d_size == 0 else f" (|D|={d_size})"
            labels.append(
                f"User {u.id} | {u.vehicle_class} | plate={u.plate_parity} | "
                f"zone={u.zone}{tag}"
            )
        idx = st.selectbox(
            "User",
            options=range(len(active_world.users)),
            format_func=lambda i: labels[i],
        )
        u = active_world.users[idx]

        # --- attributes ---
        st.markdown("### 1. User attributes")
        col_a, col_b, col_c, col_d, col_e = st.columns(5)
        col_a.metric("Vehicle class", u.vehicle_class)
        col_b.metric("Plate", u.plate_parity)
        col_c.metric("Zone", u.zone)
        col_d.metric("Fuel left", f"{u.fuel_left_km:.1f} km")
        col_e.metric("Demand", f"{u.demand_liters:.1f} L")

        # --- domain analysis ---
        st.markdown("### 2. Domain analysis (which stations could this user even go to?)")
        reachable_ids = [s.id for s in active_world.stations if active_world.reachable(u, s)]
        post_unary = active_csp.domains[u.id]

        rows = []
        for s in active_world.stations:
            row = {
                "station": f"S{s.id}",
                "name": s.name,
                "zone": s.zone,
                "is_cng": s.is_cng,
                "emerg_only": s.is_emergency_only,
                "distance_m": int(round(active_world.distance(u, s))) if active_world.distance(u, s) != float("inf") else None,
                "U1 reachable?": active_world.reachable(u, s),
                "in domain after AC-3?": s.id in post_unary,
            }
            rows.append(row)
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        st.caption(
            f"Stations within fuel range (U1): {len(reachable_ids)} · "
            f"Stations remaining after all unary + AC-3 pruning: {len(post_unary)}"
        )

        # --- assignment ---
        if mc_use_full:
            # Single method (whichever world the user is from)
            st.markdown(f"### 3. Where {active_method_label} sent this user")
            sid = active_result.assignment.get(idx)
            if sid is None:
                if idx in active_result.unassignable:
                    st.error("Infeasible (empty domain after AC-3)")
                else:
                    st.warning("Not assigned by this solver")
            else:
                s = active_world.stations[sid]
                st.success(f"→ S{sid}: **{s.name}** "
                           f"({active_world.distance(u, s):.0f} m, zone={s.zone})")
        else:
            st.markdown("### 3. Where each method sent this user")
            m1_sid = bt.assignment.get(idx)
            m2_sid = mc.assignment.get(idx)

            out_a, out_b = st.columns(2)
            with out_a:
                st.markdown("**Method 1 — Backtracking**")
                if m1_sid is None:
                    if idx in bt.unassignable:
                        st.error("Infeasible (empty domain after AC-3)")
                    else:
                        st.warning("Not assigned (BT may have failed)")
                else:
                    s = active_world.stations[m1_sid]
                    st.success(f"→ S{m1_sid}: **{s.name}** "
                               f"({active_world.distance(u, s):.0f} m, zone={s.zone})")
            with out_b:
                st.markdown("**Method 2 — Min-Conflicts**")
                if m2_sid is None:
                    if idx in mc.unassignable:
                        st.error("Infeasible (empty domain after AC-3)")
                    else:
                        st.warning("Not assigned")
                else:
                    s = active_world.stations[m2_sid]
                    st.success(f"→ S{m2_sid}: **{s.name}** "
                               f"({active_world.distance(u, s):.0f} m, zone={s.zone})")

            if m1_sid is not None and m2_sid is not None:
                if m1_sid == m2_sid:
                    st.info("Both methods agreed — same station.")
                else:
                    d1 = active_world.distance(u, active_world.stations[m1_sid])
                    d2 = active_world.distance(u, active_world.stations[m2_sid])
                    delta = d2 - d1
                    st.warning(
                        f"Methods disagreed. M2 distance − M1 distance = **{delta:+.0f} m**"
                    )

        # --- household neighbours (B1) ---
        st.markdown("### 4. Household neighbours (B1 binary constraint)")
        nbrs = active_csp._household_neighbors().get(idx, [])
        if not nbrs:
            st.caption("Lives alone — B1 does not constrain this user.")
        else:
            hh_rows = []
            for n in nbrs:
                nu = active_world.users[n]
                if mc_use_full:
                    hh_rows.append({
                        "neighbour user": n,
                        "neighbour class": nu.vehicle_class,
                        f"{active_method_label.split('—')[0].strip()} station": active_result.assignment.get(n),
                        "conflict?": active_result.assignment.get(n) is not None
                                     and active_result.assignment.get(n) == active_result.assignment.get(idx),
                    })
                else:
                    hh_rows.append({
                        "neighbour user": n,
                        "neighbour class": nu.vehicle_class,
                        "M1 station": bt.assignment.get(n),
                        "M2 station": mc.assignment.get(n),
                        "M1 conflict?": bt.assignment.get(n) is not None and bt.assignment.get(n) == m1_sid,
                        "M2 conflict?": mc.assignment.get(n) is not None and mc.assignment.get(n) == m2_sid,
                    })
            st.dataframe(pd.DataFrame(hh_rows), use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Tab 5 — Heuristic ablation: naive BT → +MRV → +LCV → +FC
# This is the teacher's "first do naive, then add heuristics, see the
# improvement" demonstration.
# ---------------------------------------------------------------------------

with tab5:
    st.subheader("Naive BT → each heuristic added one at a time")
    st.caption(
        "Slide page 31 claims MRV alone can be **1000× faster** than naive search. "
        "Run the same problem under four configurations and watch "
        "**expanded nodes / backtracks / runtime** drop as we layer the heuristics on."
    )

    with st.spinner("Running 4 BT configurations (this may take a moment for tight problems)…"):
        ablation_variants = []
        for label, (mrv, lcv, fc) in [
            ("1. Naive BT (no heuristic)",       (False, False, False)),
            ("2. + MRV+Degree (slide H1+H2)",    (True,  False, False)),
            ("3. + LCV (slide H3)",              (True,  True,  False)),
            ("4. + Forward Checking",            (True,  True,  True)),
        ]:
            r = backtracking_search(
                csp, use_mrv=mrv, use_lcv=lcv, use_fc=fc,
                time_limit_ms=10_000,
            )
            ablation_variants.append((label, r))

    ablation_rows = []
    for label, r in ablation_variants:
        ablation_rows.append({
            "variant":   label,
            "found":     r.found,
            "cost":      f"{r.cost:.0f}" if r.found else "—",
            "served":    f"{len(r.assignment)}/{len(world.users)}",
            "expanded":  r.nodes_expanded,
            "backtracks": r.backtracks,
            "runtime (ms)": f"{r.runtime_ms:.2f}",
            "timed_out": r.timed_out,
        })
    st.dataframe(pd.DataFrame(ablation_rows), use_container_width=True, hide_index=True)

    # comparison bars — runtime + expanded
    rt_path = str(STREAMLIT_OUT / "ablation_runtime.png")
    exp_path = str(STREAMLIT_OUT / "ablation_expanded.png")
    plot_comparison_bars(
        {label: r.runtime_ms for label, r in ablation_variants},
        title="Runtime — naive vs progressively-improved",
        ylabel="Runtime (ms)",
        output_path=rt_path,
    )
    plot_comparison_bars(
        {label: r.nodes_expanded for label, r in ablation_variants},
        title="Search-tree nodes expanded",
        ylabel="Expanded nodes",
        output_path=exp_path,
    )
    abc1, abc2 = st.columns(2)
    abc1.image(rt_path, caption="Smaller is better — each heuristic should reduce runtime")
    abc2.image(exp_path, caption="Smaller is better — each heuristic prunes more")

    naive = ablation_variants[0][1]
    full = ablation_variants[-1][1]
    if naive.nodes_expanded > 0 and full.nodes_expanded > 0:
        speedup_nodes = naive.nodes_expanded / max(1, full.nodes_expanded)
        speedup_time = naive.runtime_ms / max(0.01, full.runtime_ms)
        st.success(
            f"📈  Adding all three heuristics + FC reduced expanded nodes by "
            f"**{speedup_nodes:.1f}×** and runtime by **{speedup_time:.1f}×** on this scenario."
        )
    else:
        st.info(
            "Tip: this problem may be too small to show a big speedup. "
            "Increase N users / tighten constraints / lower bbox to stress the search."
        )


# ---------------------------------------------------------------------------
# Footer hint
# ---------------------------------------------------------------------------

with st.expander("ℹ️  How to read this dashboard"):
    st.markdown("""
- **Sidebar** lets you flip individual constraints on/off, tune policy
  knobs, and pick the scenario (users, bbox, time-slot, seed). Anything
  except `n_users / bbox / seed / time_slot / max_stations` is "fast" —
  only re-solving runs, no rebuild of the distance oracle.
- **Method 1** (Backtracking) is *constraint satisfaction* with the
  slide's 3 heuristics (MRV + Degree + LCV) plus Forward Checking.
- **Method 2** (Min-Conflicts) is the *local-search heuristic* from
  slide pages 40–42 — random initial assignment, iteratively move a
  conflicted user to the value that minimises total violations.
- Red **X** on the map = user with empty domain after AC-3 (cannot be
  served under the current constraint set, regardless of method).
- Toggle a constraint and watch the maps redraw — this is the
  constraint-sensitivity demo for the report.
""")
