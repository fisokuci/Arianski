import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import pearsonr, wilcoxon
import statsmodels.formula.api as smf
from statsmodels.stats.anova import AnovaRM
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import unicodedata, warnings

# just a Helper to extract paired values without pivot_table
def get_pairs(df, var, id_col, shoe_col):
    vals = {'ID': [], 'Carbon': [], 'Spikes': []}
    for uid in df[id_col].dropna().unique():
        sub = df[df[id_col] == uid]
        c = sub.loc[sub[shoe_col] == 'Carbon', var]
        s = sub.loc[sub[shoe_col] == 'Spikes', var]
        if len(c) and len(s):
            vals['ID'].append(uid)
            vals['Carbon'].append(c.iloc[0])
            vals['Spikes'].append(s.iloc[0])
    return pd.DataFrame(vals).set_index('ID')

# ── Additional analyses ────

def nonparametric_wilcoxon(df, var, id_col, shoe_col):
    pairs = get_pairs(df, var, id_col, shoe_col)
    if len(pairs) < 3:
        return None, None
    stat, p = wilcoxon(pairs["Carbon"], pairs["Spikes"])
    return stat, p

def rm_anova_shoe_sex(df, var, id_col, shoe_col):
    if "Geschlecht" not in df.columns:
        return None
    aov = df[[id_col, shoe_col, "Geschlecht", var]].dropna().rename(
        columns={shoe_col: "Schuh", var: "Measurement"}
    )
    try:
        res = AnovaRM(aov, depvar="Measurement", subject=id_col,
                      within=["Schuh"], between=["Geschlecht"]).fit()
        return res.anova_table
    except Exception:
        return None

def pca_on_diffs(df, numeric_cols, id_col, shoe_col, n_components=2):
    diff_mat = {}
    for var in numeric_cols:
        pairs = get_pairs(df, var, id_col, shoe_col)
        if len(pairs) >= 3:
            diff_mat[var] = pairs["Carbon"] - pairs["Spikes"]
    diff_df = pd.DataFrame(diff_mat).dropna(axis=1, thresh=3)
    if diff_df.shape[1] < 2:
        return None, None
    pca = PCA(n_components=n_components)
    comps = pca.fit_transform(diff_df)
    comps_df = pd.DataFrame(comps, index=diff_df.index,
                            columns=[f"PC{i+1}" for i in range(n_components)])
    return pca, comps_df

def cluster_diffs(df, numeric_cols, id_col, shoe_col, k=2):
    rows, athletes = [], []
    for uid in df[id_col].dropna().unique():
        sub = df[df[id_col] == uid]
        row = {}
        for var in numeric_cols:
            c = sub.loc[sub[shoe_col]=="Carbon", var]
            s = sub.loc[sub[shoe_col]=="Spikes", var]
            if len(c) and len(s):
                row[var] = float(c.iloc[0] - s.iloc[0])
        if len(row) > 1:
            rows.append(row); athletes.append(uid)
    diff_df = pd.DataFrame(rows, index=athletes).fillna(0)
    if diff_df.shape[0] < k:
        return None
    kmeans = KMeans(n_clusters=k, random_state=0).fit(diff_df)
    return pd.Series(kmeans.labels_, index=athletes, name="Cluster")

def simulate_power(effect_size, alpha=0.05, n_sim=1000, n_per_group=14, seed=0):
    rng = np.random.default_rng(seed)
    rejections = 0
    for _ in range(n_sim):
        data = rng.normal(loc=effect_size, scale=1.0, size=n_per_group)
        _, p = stats.ttest_1samp(data, 0.0)
        if p < alpha:
            rejections += 1
    return rejections / n_sim


# Seiteneinstellungen & Titel

st.set_page_config(page_title="Carbon-Spike Analyse", layout="centered")
st.title("🔬 Carbon- vs. Standard-Spikes – Dashboard Arianski")

st.markdown(
    """
    **Schritte:**  
    1. Arianski lade deine Excel- oder CSV-Datei hoch.  
    2. Wähle im Sidebar den **Testmodus** (Lauf / Sprung).  
    3. Dropdown → Variable wählen → Kennzahlen, Boxplot **und** vertiefte Analysen.  
    """
)

# ---------- Sidebar ----------
with st.sidebar:
    st.header("⚙️ Einstellungen")
    data_file = st.file_uploader(
        "Datendatei (.xlsx / .csv)",
        type=["xlsx", "xls", "csv"],
        help="Muss die Spalten 'RandomisierungSchuh', 'Identität', 'Lauf oder Sprung' enthalten"
    )

    if data_file is not None and data_file.name.endswith(".csv"):
        df_raw = pd.read_csv(data_file)
    elif data_file is not None:
        df_raw = pd.read_excel(data_file)
    else:
        st.stop()

    # Spalten bereinigen
    if isinstance(df_raw.columns, pd.MultiIndex):
        df_raw.columns = [
            "_".join(str(c) for c in col if str(c) != "nan").strip()
            for col in df_raw.columns
        ]
    df_raw = df_raw.loc[:, ~df_raw.columns.duplicated()]

    if "Lauf oder Sprung" not in df_raw.columns:
        for col in df_raw.columns:
            if "Lauf" in col and "Sprung" in col:
                df_raw.rename(columns={col: "Lauf oder Sprung"}, inplace=True)
                break

    modes = df_raw["Lauf oder Sprung"].dropna().unique().tolist()
    mode_choice = st.selectbox("Testmodus wählen", options=modes)

# Datenvorbereitung
warnings.filterwarnings("ignore", category=RuntimeWarning, message="Precision loss occurred")

df = df_raw[df_raw["Lauf oder Sprung"] == mode_choice].copy()

# Spaltennamen vereinheitlichen + Duplikate entfernen
df.columns = (df.columns
              .map(lambda x: unicodedata.normalize("NFKC", str(x)))
              .str.strip())
df = df.loc[:, ~df.columns.duplicated()].copy()

id_col   = next((c for c in df.columns if "identit" in c.lower() or c.lower() == "id"), None)
shoe_col = "RandomisierungSchuh"
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

if id_col is None or shoe_col not in df.columns or not numeric_cols:
    st.error("Pflichtspalten oder numerische Variablen fehlen."); st.stop()

# Statistik-Funktion
@st.cache_data
def compute_stats(df, num_cols, id_col, shoe_col):
    df = df.loc[:, ~df.columns.duplicated()].copy()
    from statsmodels.stats.multitest import multipletests

    results = []
    for var in num_cols:
        pairs = get_pairs(df, var, id_col, shoe_col)
        if len(pairs) < 3:
            continue
        diff = pairs["Carbon"] - pairs["Spikes"]
        t_stat, p_val = stats.ttest_rel(pairs["Carbon"], pairs["Spikes"])
        cohen_d = diff.mean() / diff.std(ddof=1) if diff.std(ddof=1) else np.nan
        results.append({
            "Variable": var,
            "n": len(pairs),
            "Δ-Mittel (Carbon−Spike)": diff.mean(),
            "t": t_stat,
            "p": p_val,
            "Cohen d": cohen_d
        })

    res = pd.DataFrame(results)
    if not res.empty:
        res["p_adj"] = multipletests(res["p"], method="fdr_bh")[1]
        res = res.sort_values("p_adj")
    return res

stats_df = compute_stats(df, numeric_cols, id_col, shoe_col)

# ---------- UI Hauptbereich ----------
st.subheader("Variable auswählen")
var_choice = st.selectbox(f"Variablen ({mode_choice})", options=stats_df["Variable"])

if var_choice:
    # Basisausgabe
    row = stats_df.loc[stats_df["Variable"] == var_choice].iloc[0]
    st.markdown(
        f"""**n Paare:** {int(row['n'])}  
**Δ-Mittel:** {row['Δ-Mittel (Carbon−Spike)']:.4g}  
**_p_-Wert:** {row['p']:.4f} (adj: {row['p_adj']:.4f})  
**Cohen-d:** {row['Cohen d']:.3f}"""
    )

    pairs = get_pairs(df, var_choice, id_col, shoe_col)
    diffs = pairs["Carbon"] - pairs["Spikes"]

    # Boxplot
    fig, ax = plt.subplots()
    ax.boxplot([pairs["Carbon"], pairs["Spikes"]],
               labels=["Carbon", "Spikes"], showfliers=True)
    ax.set_title(f"{var_choice} ({mode_choice})"); ax.set_ylabel(var_choice)
    st.pyplot(fig)

    # Vertiefte Analysen
    # 1. Bootstrap-CI
    def bootstrap_ci(series, n_boot=10_000, seed=0):
        rng = np.random.default_rng(seed)
        boot_means = rng.choice(series, (n_boot, len(series)), replace=True).mean(axis=1)
        return np.percentile(boot_means, [2.5, 97.5])
    ci_low, ci_high = bootstrap_ci(diffs)
    st.markdown(f"**Bootstrapped 95% CI:** [{ci_low:.1f} ; {ci_high:.1f}]")

    # 2. Mixed Model
    lmm_cols = ["Geschlecht", "Alter zu Messzeitpunkt"]
    if all(c in df.columns for c in lmm_cols):
        long_df = df[[id_col, shoe_col, *lmm_cols, var_choice]].dropna()
        long_df = long_df.rename(columns={var_choice: "Outcome", shoe_col: "Schuh",
                                          "Geschlecht": "Sex", "Alter zu Messzeitpunkt": "Age"})
        try:
            m = smf.mixedlm("Outcome ~ Schuh + Sex + Age", long_df, groups=long_df[id_col]) \
                   .fit(reml=False)
            st.markdown("**Linear Mixed Model (adjustiert)**")
            st.dataframe(pd.DataFrame({"Coef": m.params, "SE": m.bse, "p": m.pvalues})
                         .reset_index(names="Term"), use_container_width=True)
        except Exception as e:
            st.info(f"Mixed-Model Fehler: {e}")

    # 3. Bland-Altman
    fig2, ax2 = plt.subplots()
    mean_vals = pairs.mean(axis=1); bias = diffs.mean(); loa = 1.96*diffs.std(ddof=1)
    ax2.scatter(mean_vals, diffs)
    ax2.axhline(bias, ls='--'); ax2.axhline(bias+loa, ls=':'); ax2.axhline(bias-loa, ls=':')
    ax2.set_xlabel("Mittelwert"); ax2.set_ylabel("Differenz"); ax2.set_title("Bland-Altman-Plot")
    st.pyplot(fig2)

    # 4. Korrelationen (robust gegen unterschiedliche Längen)
    bf_cols  = [c for c in df.columns if "% BF" in c or "Body F" in c]
    smm_cols = [c for c in df.columns if "SMM" in c]
    corr_rows = []
    for comp_col, label in [(bf_cols[0] if bf_cols else None, "%BF"),
                            (smm_cols[0] if smm_cols else None, "SMM")]:
        if not comp_col:
            continue
        vec = df.set_index(id_col)[comp_col].dropna()
        idx = diffs.index.intersection(vec.index)
        if len(idx) < 3:
            continue
        x = diffs.loc[idx].dropna()
        y = vec.loc[idx].dropna()
        idx_clean = x.index.intersection(y.index)
        x = x.loc[idx_clean]; y = y.loc[idx_clean]
        try:
            r, p = pearsonr(x, y)
            corr_rows.append({"Pair": f"Δ vs {label}", "r": r, "p": p})
        except ValueError as e:
            st.info(f"Korrelation Δ vs {label} nicht berechnet: {e}")
    if corr_rows:
        st.markdown("**Korrelationen (Δ-Leistung vs Körperzusammensetzung)**")
        st.dataframe(pd.DataFrame(corr_rows), use_container_width=True)

    # 5. Wilcoxon signed‐rank
    try:
        stat_w, p_w = nonparametric_wilcoxon(df, var_choice, id_col, shoe_col)
        if stat_w is not None:
            st.markdown(f"**Wilcoxon signed‐rank:** W={stat_w:.2f}, p={p_w:.3f}")
    except Exception as e:
        st.info(f"Wilcoxon-Test Fehler: {e}")

    # 6. RM-ANOVA Shoe × Sex
    try:
        anova_table = rm_anova_shoe_sex(df, var_choice, id_col, shoe_col)
        if anova_table is not None:
            st.markdown("**RM-ANOVA (Schuh × Geschlecht)**")
            st.dataframe(anova_table)
    except Exception as e:
        st.info(f"RM-ANOVA Fehler: {e}")

    # 7. PCA auf Δ-Profile
    try:
        pca, comps_df = pca_on_diffs(df, numeric_cols, id_col, shoe_col)
        if comps_df is not None:
            st.markdown("**PCA of Δ-Profiles**")
            st.dataframe(comps_df)
    except Exception as e:
        st.info(f"PCA Fehler: {e}")

    # 8. K-Means Clustering
    try:
        clusters = cluster_diffs(df, numeric_cols, id_col, shoe_col, k=2)
        if clusters is not None:
            st.markdown("**Athlete Clusters (k=2) based on Δ-Profile**")
            st.dataframe(clusters)
    except Exception as e:
        st.info(f"Clustering-Fehler: {e}")

    # 9. Power Simulation
    try:
        d = row['Δ-Mittel (Carbon−Spike)'] / np.std(diffs, ddof=1)
        power = simulate_power(d, alpha=0.05, n_sim=1000, n_per_group=len(diffs))
        st.markdown(f"**Estimated Power** for d={d:.2f} with n={len(diffs)}: {power:.2%}")
    except Exception as e:
        st.info(f"Power-Simulation Fehler: {e}")

# Gesamttabelle
with st.expander("Alle Ergebnisse (FDR-sortiert)"):
    st.dataframe(
        stats_df[["Variable", "n", "Δ-Mittel (Carbon−Spike)", "t", "p", "p_adj", "Cohen d"]],
        use_container_width=True
    )

st.caption("©️ Auto-generiertes Dashboard – robust gegen ID-Duplikate.")
