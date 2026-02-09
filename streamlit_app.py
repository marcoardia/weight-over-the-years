import io
from io import StringIO
from datetime import date

import numpy as np
import pandas as pd
import altair as alt
import streamlit as st

# =========================
# Page setup
# =========================
st.set_page_config(page_title="Year-over-Year Weight Tracker", layout="wide")
st.title("📈 Year-over-Year Weight Tracker")

with st.expander("ℹ️ About this app", expanded=False):
    st.markdown(
        """
This app lets you upload a CSV with **date** and **weight** and will:

- Draw a **multi-year comparison chart** where the x-axis is **Jan 1 → Dec 31**, and each colored line is a **year**.
- Show **per-year statistics** (start, end, change, min/max, mean, median, count, coverage).
- Compare **today (or selected date) in the current year** with the **same calendar day across previous years**.
- **Clean outliers** (IQR / Z-score / Robust Z via MAD), either **remove** or **winsorize**.
- Provide a **minimum y-axis** control (default **75**) to better see differences.
- **Trendline smoothing** options: Rolling mean, EMA, and LOESS — overlay or replace.

**Notes**
- If you have multiple measurements on the same day, the **last entry per day** is used.
- Missing days are filled by **linear interpolation** for visualization/comparison.
- You can choose to **include or drop Feb 29** for 365-day alignment.
        """
    )

# =========================
# Sidebar controls
# =========================
st.sidebar.header("⚙️ Settings")

uploaded = st.sidebar.file_uploader("Upload weight CSV", type=["csv"])

decimal_sep = st.sidebar.selectbox("Decimal separator", options=[",", "."], index=1)
custom_date_format = st.sidebar.text_input(
    "Custom date format (optional, e.g. %d/%m/%Y)",
    value="",
    help="Leave empty to auto-detect. Examples: %Y-%m-%d, %d/%m/%Y",
)

include_feb29 = st.sidebar.checkbox(
    "Include Feb 29 in charts",
    value=False,
    help="If off, Feb 29 is removed to align all years to 365 days."
)

# (Base smoothing before plotting; keep as-is if you like light smoothing on source lines)
smoothing_win = st.sidebar.number_input(
    "Base smoothing window (days, rolling mean)",
    min_value=1, max_value=31, value=1, step=1,
    help="Applies a centered rolling mean to the base lines before plotting. Set to 1 to disable."
)

units = st.sidebar.text_input("Units label", value="kg")

compare_date = st.sidebar.date_input(
    "Compare for date (current year vs history)",
    value=date.today(),
    help="Pick the calendar day to compare across years. Defaults to today."
)

st.sidebar.markdown("---")
st.sidebar.subheader("🧹 Outlier cleaning")

outlier_method = st.sidebar.selectbox(
    "Method",
    options=["None", "IQR (per year)", "Z-score (per year)", "Robust Z (MAD, per year)"],
    index=0,
    help=(
        "IQR: values outside [Q1 - k*IQR, Q3 + k*IQR].\n"
        "Z-score: |(x - mean)/std| > z.\n"
        "Robust Z: |(x - median)/(1.4826*MAD)| > z."
    )
)
iqr_k = st.sidebar.number_input("IQR multiplier k", min_value=0.5, max_value=5.0, value=1.5, step=0.1)
z_thresh = st.sidebar.number_input("Z / Robust Z threshold", min_value=1.0, max_value=6.0, value=3.5, step=0.1)

outlier_action = st.sidebar.selectbox("Action", options=["Remove", "Winsorize (cap to threshold)"], index=0)

apply_to_current_year = st.sidebar.checkbox(
    "Apply outlier cleaning to current year as well",
    value=True,
    help="If unchecked, the most recent year is left untouched."
)

st.sidebar.markdown("---")
st.sidebar.subheader("📉 Trendline smoothing")

trend_method = st.sidebar.selectbox(
    "Trendline method",
    options=["None", "Rolling mean (trend)", "Exponential moving average (EMA)", "LOESS (Altair)"],
    index=0,
    help=(
        "Add a smoothed trend: Rolling mean (centered), EMA, or LOESS.\n"
        "Can be overlaid or used to replace the base lines."
    ),
)
trend_overlay = st.sidebar.checkbox(
    "Overlay trendline (dashed)",
    value=True,
    help="If off, the chart will show only the smoothed trendlines."
)

# Parameters per method
trend_win = st.sidebar.number_input(
    "Rolling window (trend)",
    min_value=2, max_value=61, value=7, step=1,
    help="Window for Rolling mean trend (centered). Used when method = 'Rolling mean (trend)'."
)
ema_span = st.sidebar.number_input(
    "EMA span (days)",
    min_value=2, max_value=90, value=14, step=1,
    help="Span for Exponential Moving Average trend. Used when method = 'EMA'."
)
loess_bw = st.sidebar.slider(
    "LOESS bandwidth",
    min_value=0.05, max_value=1.0, value=0.3, step=0.05,
    help="Smoothing strength for LOESS trend. Used when method = 'LOESS (Altair)'."
)

# =========================
# Helpers
# =========================
@st.cache_data(show_spinner=False)
def _read_csv_flexible(content: bytes, decimal_sep: str) -> pd.DataFrame:
    """
    Read CSV from bytes with a tolerant approach to encodings and decimal separators.
    Returns a DataFrame with all columns as strings first.
    """
    # Try utf-8 then latin-1 (common fallback)
    df = None
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            txt = content.decode(enc)
            df = pd.read_csv(StringIO(txt), dtype=str)
            break
        except Exception:
            df = None
    if df is None:
        # Last resort: let pandas infer from bytes
        df = pd.read_csv(io.BytesIO(content), dtype=str)

    df.columns = [c.strip() for c in df.columns]

    # Normalize decimal separator by replacing commas with dots in object columns
    if decimal_sep == ",":
        df = df.apply(lambda s: s.str.replace(",", ".", regex=False) if s.dtype == "object" else s)

    return df


def _guess_date_cols(cols):
    cands = []
    for c in cols:
        lc = c.lower()
        if any(k in lc for k in ["date", "datum", "day", "time", "weigh", "measure"]):
            cands.append(c)
    return cands or list(cols)


def _guess_weight_cols(cols):
    cands = []
    for c in cols:
        lc = c.lower()
        if any(k in lc for k in ["weight", "kg", "massa", "mass", "weigh"]):
            cands.append(c)
    return cands or list(cols)


def _parse_dates(series: pd.Series, fmt: str | None):
    if fmt and fmt.strip():
        return pd.to_datetime(series, format=fmt, errors="coerce")
    return pd.to_datetime(series, errors="coerce")


def _coerce_numeric(series: pd.Series):
    return pd.to_numeric(series, errors="coerce")


def _dedupe_last_per_day(df: pd.DataFrame, date_col: str):
    return (
        df.sort_values(date_col)
          .drop_duplicates(subset=[date_col], keep="last")
          .reset_index(drop=True)
    )


def _daily_interpolated(df: pd.DataFrame, date_col: str, weight_col: str, include_feb29: bool):
    """
    Returns a daily, within-year, linearly interpolated DataFrame with:
      - date (actual year)
      - year
      - weight (interpolated)
      - ref_date (same month/day on a reference non-leap year 2001; Feb 29 → 2000 if included)
      - month_day label (e.g., 'Jan 03')
    """
    if df.empty:
        return df

    df = df[[date_col, weight_col]].copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[weight_col] = pd.to_numeric(df[weight_col], errors="coerce")
    df = df.dropna(subset=[date_col, weight_col]).sort_values(date_col)

    out = []
    years = np.sort(df[date_col].dt.year.unique())

    for y in years:
        g = df[df[date_col].dt.year == y].copy()
        if g.empty:
            continue

        g = g.set_index(date_col)
        g = g[~g.index.duplicated(keep="last")]

        start = pd.Timestamp(year=y, month=1, day=1)
        end = pd.Timestamp(year=y, month=12, day=31)
        idx = pd.date_range(start=start, end=end, freq="D")

        gg = g.reindex(idx)
        gg[weight_col] = gg[weight_col].astype(float)
        gg[weight_col] = gg[weight_col].interpolate(method="time", limit_direction="both")

        gg["date"] = gg.index
        gg["year"] = y
        gg["month"] = gg.index.month
        gg["day"] = gg.index.day

        if not include_feb29:
            gg = gg[~((gg["month"] == 2) & (gg["day"] == 29))]

        # Build ref_date aligned to non-leap reference (2001). If Feb 29 included, map that day to 2000-02-29.
        ref_dates = []
        for m, d in zip(gg["month"], gg["day"]):
            try:
                ref_dates.append(pd.Timestamp(year=2001, month=m, day=d))
            except ValueError:
                # Only triggered by Feb 29; use a leap reference year for display ordering
                ref_dates.append(pd.Timestamp(year=2000, month=m, day=d))
        gg["ref_date"] = pd.to_datetime(ref_dates)

        # ✅ Use .dt for datetime Series (fixes AttributeError)
        gg["month_day"] = pd.to_datetime(gg["date"]).dt.strftime("%b %d")

        out.append(gg.reset_index(drop=True))

    outdf = pd.concat(out, ignore_index=True)
    outdf = outdf.rename(columns={weight_col: "weight"})
    return outdf[["date", "year", "ref_date", "month_day", "weight"]]


def _rolling(df: pd.DataFrame, window: int):
    if window <= 1:
        return df
    return (
        df.sort_values(["year", "ref_date"])
          .groupby("year", group_keys=False)
          .apply(lambda g: g.assign(weight=g["weight"].rolling(window, min_periods=1, center=True).mean()))
          .reset_index(drop=True)
    )


def _yearly_stats(original_daily: pd.DataFrame, original_points: pd.DataFrame):
    points = original_points.copy()
    points["year"] = points["date"].dt.year
    byyear_pts = points.sort_values("date").groupby("year")

    start = byyear_pts.first()["weight"].rename("start")
    end = byyear_pts.last()["weight"].rename("end")
    change = (end - start).rename("change")
    pct_change = ((change / start) * 100).rename("pct_change")

    count = byyear_pts["weight"].count().rename("measurements")
    coverage = byyear_pts["date"].nunique().rename("unique_days").astype(int)

    daily = original_daily.copy()
    byyear_daily = daily.groupby("year")["weight"]
    stats = pd.concat(
        [
            start,
            end,
            change,
            pct_change,
            byyear_daily.min().rename("min"),
            byyear_daily.max().rename("max"),
            byyear_daily.mean().rename("mean"),
            byyear_daily.median().rename("median"),
            byyear_daily.std(ddof=0).rename("std"),
            count,
            coverage,
        ],
        axis=1,
    ).reset_index().rename(columns={"year": "Year"})

    stats = stats[
        ["Year", "start", "end", "change", "pct_change", "min", "max", "mean", "median", "std", "measurements", "unique_days"]
    ]
    return stats


def _value_on_ref_date(daily_df: pd.DataFrame, year: int, ref_date: pd.Timestamp) -> float | None:
    g = daily_df[daily_df["year"] == year]
    if g.empty:
        return None
    g2 = g[g["ref_date"].dt.month.eq(ref_date.month) & g["ref_date"].dt.day.eq(ref_date.day)]
    if g2.empty:
        return None
    return float(g2["weight"].iloc[0])


def _ytd_change(daily_df: pd.DataFrame, year: int, ref_date: pd.Timestamp) -> float | None:
    g = daily_df[daily_df["year"] == year]
    if g.empty:
        return None
    jan1 = pd.Timestamp(year=2001, month=1, day=1)
    v0 = _value_on_ref_date(daily_df, year, jan1)
    vt = _value_on_ref_date(daily_df, year, ref_date)
    if v0 is None or vt is None:
        return None
    return float(vt - v0)


# ---------- Outlier utilities ----------
def _iqr_bounds(x: pd.Series, k: float):
    q1 = x.quantile(0.25)
    q3 = x.quantile(0.75)
    iqr = q3 - q1
    lo = q1 - k * iqr
    hi = q3 + k * iqr
    return lo, hi


def _z_bounds(x: pd.Series, z: float):
    mu = x.mean()
    sd = x.std(ddof=0)
    if sd == 0 or np.isnan(sd):
        return -np.inf, np.inf
    lo = mu - z * sd
    hi = mu + z * sd
    return lo, hi


def _robust_z_bounds(x: pd.Series, z: float):
    med = x.median()
    mad = (x - med).abs().median()
    sigma = 1.4826 * mad  # consistency constant
    if sigma == 0 or np.isnan(sigma):
        return -np.inf, np.inf
    lo = med - z * sigma
    hi = med + z * sigma
    return lo, hi


def _apply_outlier_cleaning(points_df: pd.DataFrame,
                            method: str,
                            iqr_k: float,
                            z_thresh: float,
                            action: str,
                            include_current_year: bool):
    """
    points_df: deduped last-per-day points with columns ['date','weight']
    Returns cleaned points_df, mask of outliers (bool), and capping bounds per year if winsorized
    """
    if method == "None" or points_df.empty:
        points_df = points_df.copy()
        points_df["is_outlier"] = False
        return points_df, pd.Series(False, index=points_df.index), {}

    pts = points_df.copy()
    pts["year"] = pts["date"].dt.year

    years = sorted(pts["year"].unique())
    if not include_current_year and len(years) > 0:
        years_to_clean = years[:-1]  # leave max year as-is
    else:
        years_to_clean = years

    is_out = pd.Series(False, index=pts.index)
    bounds_by_year = {}

    for y in years_to_clean:
        g_idx = pts.index[pts["year"] == y]
        x = pts.loc[g_idx, "weight"].astype(float)

        if method.startswith("IQR"):
            lo, hi = _iqr_bounds(x, iqr_k)
        elif method.startswith("Z-score"):
            lo, hi = _z_bounds(x, z_thresh)
        elif method.startswith("Robust Z"):
            lo, hi = _robust_z_bounds(x, z_thresh)
        else:
            lo, hi = -np.inf, np.inf

        bounds_by_year[y] = (lo, hi)
        out_mask = (x < lo) | (x > hi)
        is_out.loc[g_idx] = out_mask

        if action.startswith("Winsorize"):
            x_clipped = x.clip(lower=lo, upper=hi)
            pts.loc[g_idx, "weight"] = x_clipped
        # removal handled after loop

    if action == "Remove":
        cleaned = pts.loc[~is_out].drop(columns=["year"]).reset_index(drop=True)
        cleaned["is_outlier"] = False
        return cleaned, is_out, bounds_by_year
    else:
        pts = pts.drop(columns=["year"]).reset_index(drop=True)
        pts["is_outlier"] = is_out.reindex(pts.index, fill_value=False).values
        return pts, is_out, bounds_by_year


# =========================
# Main logic
# =========================
if uploaded is None:
    st.info("👆 Upload a CSV to get started.")
    st.stop()

# Read and parse
raw = _read_csv_flexible(uploaded.getvalue(), decimal_sep=decimal_sep)

# Column pickers
st.subheader("1) Map columns")
date_col = st.selectbox("Date column", options=_guess_date_cols(raw.columns))
weight_col = st.selectbox("Weight column", options=_guess_weight_cols(raw.columns))

# Convert
df = raw[[date_col, weight_col]].copy()
df[weight_col] = df[weight_col].str.strip()
if decimal_sep == ",":
    df[weight_col] = df[weight_col].str.replace(",", ".", regex=False)
df[weight_col] = _coerce_numeric(df[weight_col])
df[date_col] = _parse_dates(df[date_col], custom_date_format if custom_date_format else None)

# Clean base
df = df.dropna(subset=[date_col, weight_col]).copy()
df = df.rename(columns={date_col: "date", weight_col: "weight"})
df["date"] = df["date"].dt.normalize()

# Filter out impossible values (negative, zero)
df = df[df["weight"] > 0]

if df.empty:
    st.error("No valid rows after parsing. Check your date/weight columns and formats.")
    st.stop()

# One entry per day (last if multiple)
points_raw = _dedupe_last_per_day(df, "date").sort_values("date").reset_index(drop=True)

# ---- Outlier cleaning (on deduped points) ----
points_clean, out_mask, bounds_info = _apply_outlier_cleaning(
    points_raw,
    method=outlier_method,
    iqr_k=iqr_k,
    z_thresh=z_thresh,
    action=outlier_action,
    include_current_year=apply_to_current_year
)

# Quick summary + mini chart to visualize outliers/caps
with st.expander("🧪 Outlier cleaning summary", expanded=(outlier_method != "None")):
    total = len(points_raw)
    n_out = int(out_mask.sum())
    st.markdown(
        f"- **Method:** {outlier_method}  \n"
        f"- **Action:** {outlier_action}  \n"
        f"- **Affected points:** **{n_out}** / {total} ({(n_out/total*100):.1f}%)"
    )
    preview_df = points_raw.merge(points_clean, on=["date"], how="left", suffixes=("_raw", "_clean"))
    preview_df["year"] = preview_df["date"].dt.year
    preview_long = preview_df.melt(
        id_vars=["date", "year"],
        value_vars=["weight_raw", "weight_clean"],
        var_name="series",
        value_name="weight"
    )
    mini = alt.Chart(preview_long).mark_line().encode(
        x=alt.X("date:T", title="Date"),
        y=alt.Y("weight:Q", title=f"Weight ({units})"),
        color=alt.Color("series:N", title="Series",
                        scale=alt.Scale(domain=["weight_raw", "weight_clean"], range=["#d62728", "#1f77b4"])),
        opacity=alt.condition(alt.datum.series == "weight_clean", alt.value(1.0), alt.value(0.5)),
        tooltip=[alt.Tooltip("year:N"), alt.Tooltip("date:T"), alt.Tooltip("weight:Q", format=".2f")]
    ).properties(height=220)
    st.altair_chart(mini.interactive(bind_y=False), use_container_width=True)

# Daily interpolated per year (for consistent comparisons)
daily = _daily_interpolated(points_clean, "date", "weight", include_feb29=include_feb29)
if daily.empty:
    st.error("Could not build the daily series. Please check your data.")
    st.stop()

# Optional base smoothing (before plotting)
plot_data = _rolling(daily, smoothing_win)

# =========================
# 2) Interactive Year-over-year chart
# =========================
st.subheader("2) Year-over-year chart")

# Collect available years
all_years = sorted(daily["year"].unique().tolist())

# Persist selection across reruns
if "years_selected" not in st.session_state:
    st.session_state.years_selected = all_years  # default: show all

col_left, col_right = st.columns([3, 1])
with col_left:
    years_selected = st.multiselect(
        "Show years",
        options=all_years,
        default=st.session_state.years_selected,
        key="years_selected",
        help="Select which years to display. You can also click the legend to toggle."
    )
with col_right:
    if st.button("Select all"):
        st.session_state.years_selected = all_years
    if st.button("Clear selection"):
        st.session_state.years_selected = []

# Filter plot data to selected years
plot_subset = plot_data[plot_data["year"].isin(st.session_state.years_selected)]
if plot_subset.empty:
    st.info("No years selected. Choose one or more years to display the chart.")
else:
    # Control the minimum Y-axis value (default 75) to see differences better
    y_min = st.number_input("Minimum Y-axis value", value=75.0, step=0.5, key="y_min_chart")

    # Legend-based toggle — dims non-selected years (multiselect filters actual inclusion)
    year_legend_sel = alt.selection_multi(fields=["year"], bind="legend")

    # Compute y domain with a bit of headroom
    max_y = float(plot_subset["weight"].max()) if not plot_subset.empty else 0.0
    headroom = 0.5
    y_domain = [float(y_min), max_y + headroom] if max_y >= y_min else [float(y_min), float(y_min) + 1.0]

    base = alt.Chart(plot_subset)

    # --- Base lines (possibly pre-smoothed by 'smoothing_win') ---
    line = base.mark_line(size=2).encode(
        x=alt.X(
            "ref_date:T",
            axis=alt.Axis(
                title="Calendar date (Jan 1 → Dec 31)",
                format="%b",
                labelAngle=0,
                tickCount=12
            )
        ),
        y=alt.Y(
            "weight:Q",
            title=f"Weight ({units})",
            scale=alt.Scale(domain=y_domain),
        ),
        color=alt.Color("year:N", title="Year", scale=alt.Scale(scheme="category10")),
        opacity=alt.condition(year_legend_sel, alt.value(1.0), alt.value(0.2)),
        tooltip=[
            alt.Tooltip("year:N", title="Year"),
            alt.Tooltip("month_day:N", title="Date"),
            alt.Tooltip("weight:Q", title=f"Weight ({units})", format=".2f")
        ],
    ).add_selection(year_legend_sel).properties(height=420)

    # --- Trendline layer (optional) ---
    trend_layer = None
    if trend_method != "None":
        if trend_method == "Rolling mean (trend)":
            # Compute centered rolling mean per year as 'trend'
            trend_df = plot_subset.sort_values(["year", "ref_date"]).copy()
            trend_df["trend"] = (
                trend_df.groupby("year", group_keys=False)["weight"]
                        .apply(lambda s: s.rolling(trend_win, min_periods=1, center=True).mean())
            )
            trend_layer = alt.Chart(trend_df).mark_line(size=3, strokeDash=[6, 3]).encode(
                x=alt.X("ref_date:T"),
                y=alt.Y("trend:Q", title=f"Weight ({units})", scale=alt.Scale(domain=y_domain)),
                color=alt.Color("year:N", title="Year", scale=alt.Scale(scheme="category10")),
                opacity=alt.condition(year_legend_sel, alt.value(1.0), alt.value(0.2)),
                tooltip=[
                    alt.Tooltip("year:N", title="Year"),
                    alt.Tooltip("month_day:N", title="Date"),
                    alt.Tooltip("trend:Q", title=f"Trend ({units})", format=".2f"),
                ],
            )

        elif trend_method == "Exponential moving average (EMA)":
            trend_df = plot_subset.sort_values(["year", "ref_date"]).copy()
            # Compute EMA per year as 'trend'
            def _ema(group):
                return group.ewm(span=ema_span, adjust=False).mean()
            trend_df["trend"] = trend_df.groupby("year", group_keys=False)["weight"].apply(_ema)
            trend_layer = alt.Chart(trend_df).mark_line(size=3, strokeDash=[6, 3]).encode(
                x=alt.X("ref_date:T"),
                y=alt.Y("trend:Q", title=f"Weight ({units})", scale=alt.Scale(domain=y_domain)),
                color=alt.Color("year:N", title="Year", scale=alt.Scale(scheme="category10")),
                opacity=alt.condition(year_legend_sel, alt.value(1.0), alt.value(0.2)),
                tooltip=[
                    alt.Tooltip("year:N", title="Year"),
                    alt.Tooltip("month_day:N", title="Date"),
                    alt.Tooltip("trend:Q", title=f"Trend ({units})", format=".2f"),
                ],
            )

        elif trend_method == "LOESS (Altair)":
            # Use Altair's transform_loess grouped by year; outputs smoothed 'weight'
            trend_layer = base.transform_loess(
                on="ref_date",  # x
                loess="weight",  # y
                groupby=["year"],
                bandwidth=float(loess_bw),
            ).mark_line(size=3, strokeDash=[6, 3]).encode(
                x=alt.X("ref_date:T"),
                y=alt.Y("weight:Q", title=f"Weight ({units})", scale=alt.Scale(domain=y_domain)),
                color=alt.Color("year:N", title="Year", scale=alt.Scale(scheme="category10")),
                opacity=alt.condition(year_legend_sel, alt.value(1.0), alt.value(0.2)),
                tooltip=[
                    alt.Tooltip("year:N", title="Year"),
                    alt.Tooltip("ref_date:T", title="Date", format="%b %d"),
                    alt.Tooltip("weight:Q", title=f"LOESS ({units})", format=".2f"),
                ],
            )

    # Combine layers depending on overlay choice
    if trend_layer is not None:
        chart = (line + trend_layer) if trend_overlay else trend_layer
        # Ensure legend selection is active on the composite chart
        chart = chart.add_selection(year_legend_sel)
    else:
        chart = line

    st.altair_chart(chart.interactive(bind_y=False), use_container_width=True)

# =========================
# 3) Yearly statistics
# =========================
st.subheader("3) Yearly statistics")
stats = _yearly_stats(daily, points_clean)
fmt = {
    "start": "{:.2f}",
    "end": "{:.2f}",
    "change": "{:+.2f}",
    "pct_change": "{:+.1f}%",
    "min": "{:.2f}",
    "max": "{:.2f}",
    "mean": "{:.2f}",
    "median": "{:.2f}",
    "std": "{:.2f}",
    "measurements": "{:,.0f}",
    "unique_days": "{:,.0f}"
}
st.dataframe(
    stats.style.format(fmt),
    use_container_width=True,
    hide_index=True
)

# =========================
# 4) Current year/day vs historical
# =========================
st.subheader("4) Current year/day vs historical")

years = sorted(daily["year"].unique())
current_year = max(years)
ref_md = pd.Timestamp(year=2001, month=compare_date.month, day=compare_date.day)

# Values at selected ref date across years
values_on_day = []
ytd_changes = []
for y in years:
    v = _value_on_ref_date(daily, y, ref_md)
    c = _ytd_change(daily, y, ref_md)
    if v is not None:
        values_on_day.append({"Year": y, "Weight": v})
    if c is not None:
        ytd_changes.append({"Year": y, "YTD Change": c})

values_on_day_df = pd.DataFrame(values_on_day).sort_values("Year")
ytd_changes_df = pd.DataFrame(ytd_changes).sort_values("Year")

if values_on_day_df.empty:
    st.info("Not enough data to compute same-day comparisons across years.")
else:
    hist = values_on_day_df[values_on_day_df["Year"] < current_year]
    current_row = values_on_day_df[values_on_day_df["Year"] == current_year]

    col1, col2, col3, col4 = st.columns(4)

    if not current_row.empty:
        current_value = float(current_row["Weight"].iloc[0])
        col1.metric(f"{current_year} weight on {compare_date.strftime('%b %d')}", f"{current_value:.2f} {units}")
    else:
        current_value = None
        col1.metric(f"{current_year} weight on {compare_date.strftime('%b %d')}", "—")

    if not hist.empty:
        mean_hist = hist["Weight"].mean()
        median_hist = hist["Weight"].median()
        min_hist = hist["Weight"].min()
        max_hist = hist["Weight"].max()
        std_hist = hist["Weight"].std(ddof=0) if len(hist) > 1 else 0.0

        col2.metric("History mean (same day)", f"{mean_hist:.2f} {units}")
        col3.metric("History median (same day)", f"{median_hist:.2f} {units}")
        col4.metric("Range (min → max)", f"{min_hist:.2f} → {max_hist:.2f} {units}")

        if current_value is not None and std_hist > 0:
            pct = (hist["Weight"] <= current_value).mean() * 100.0
            z = (current_value - mean_hist) / std_hist
            st.markdown(
                f"**Current vs past (same day):** percentile ≈ **{pct:.1f}**; z-score ≈ **{z:.2f}**."
            )
    else:
        st.info("No prior years to compare against (history is empty).")

    st.markdown("**Year-to-date (Jan 1 → selected date) change vs history:**")
    if not ytd_changes_df.empty:
        current_ytd = ytd_changes_df.loc[ytd_changes_df["Year"] == current_year, "YTD Change"]
        hist_ytd = ytd_changes_df[ytd_changes_df["Year"] < current_year]["YTD Change"]

        c1, c2, c3 = st.columns(3)
        if not current_ytd.empty:
            c1.metric(f"{current_year} YTD change", f"{current_ytd.iloc[0]:+.2f} {units}")
        else:
            c1.metric(f"{current_year} YTD change", "—")

        if not hist_ytd.empty:
            c2.metric("History mean YTD change", f"{hist_ytd.mean():+.2f} {units}")
            c3.metric("History median YTD change", f"{hist_ytd.median():+.2f} {units}")

        ytd_hist_chart = alt.Chart(
            ytd_changes_df.rename(columns={"YTD Change": "ytd_change"})
        ).mark_bar().encode(
            x=alt.X("ytd_change:Q", bin=alt.Bin(maxbins=20), title=f"YTD change ({units})"),
            y=alt.Y("count():Q", title="Count"),
            tooltip=[alt.Tooltip("count():Q", title="Count")]
        ).properties(height=200)
        st.altair_chart(ytd_hist_chart, use_container_width=True)

# =========================
# Data views
# =========================
with st.expander("🔍 See processed daily data (interpolated)", expanded=False):
    st.dataframe(
        daily.sort_values(["year", "date"])
             .assign(ref_day=daily["ref_date"].dt.strftime("%b %d"))
             .rename(columns={"date": "actual_date"})
             [["actual_date", "year", "ref_day", "weight"]],
        use_container_width=True
    )

with st.expander("📄 Raw vs cleaned points (before interpolation)", expanded=False):
    compare_pts = points_raw.merge(points_clean, on="date", how="outer", suffixes=("_raw", "_clean")).sort_values("date")
    st.dataframe(compare_pts, use_container_width=True)
``