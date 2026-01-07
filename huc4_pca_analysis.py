"""Independent PCA analysis for HUC4 per-basin monthly series.

This script performs PCA only and writes:
 - explained_variance.csv (percent variance per PC)
 - pca_scores.csv (basin x PC scores)
 - pca_loadings.csv (PC x feature loadings)
 - pca_scatter.png (PC1 vs PC2 with basin labels)

This is intentionally independent from the KMeans script.
"""
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Save PCA outputs in a dedicated folder (not under clustering_huc4)
HERE = Path(__file__).parent
INPUT_DIR = HERE / 'HUC4_mm_averages_csvs'
OUT_DIR = HERE / 'pca_huc4'  # separate folder for PCA outputs
OUT_DIR.mkdir(parents=True, exist_ok=True)

# If older PCA outputs were accidentally written into clustering_huc4, move them
# to the new OUT_DIR so clustering_huc4 contains only KMeans outputs.
OLD_DIR = HERE / 'clustering_huc4'
if OLD_DIR.exists():
    for name in ('explained_variance.csv', 'pca_scores.csv', 'pca_loadings.csv', 'pca_scatter.png'):
        src = OLD_DIR / name
        if src.exists():
            dst = OUT_DIR / name
            try:
                shutil.move(str(src), str(dst))
                print(f"Moved existing PCA file {src.name} -> {dst}")
            except Exception as e:
                print(f"Failed to move {src}: {e}")


def load_combined(input_dir: Path) -> pd.DataFrame:
    files = sorted(input_dir.glob('basin_*.csv'))
    if not files:
        raise FileNotFoundError(f'No basin_*.csv files in {input_dir}')
    series = {}
    for f in files:
        df = pd.read_csv(f)
        if 'year' in df.columns and 'month' in df.columns:
            try:
                df['date'] = pd.to_datetime(dict(year=df['year'].astype(int), month=df['month'].astype(int), day=1))
            except Exception:
                continue
        elif 'time' in df.columns:
            df['date'] = pd.to_datetime(df['time'])
        else:
            continue
        valcol = None
        cols_lower = [c.lower() for c in df.columns]
        for cand in ['average', 'mean', 'value', 'runoff']:
            if cand in cols_lower:
                valcol = df.columns[cols_lower.index(cand)]
                break
        if valcol is None:
            numeric = df.select_dtypes(include=[np.number]).columns.tolist()
            numeric = [c for c in numeric if c not in ('year', 'month')]
            if not numeric:
                continue
            valcol = numeric[0]
        s = df.set_index('date')[valcol].sort_index()
        name = f.stem.replace('basin_', '')
        series[name] = s
    combined = pd.concat(series, axis=1)
    combined.sort_index(inplace=True)
    return combined


def main():
    combined = load_combined(INPUT_DIR)
    combined = combined.interpolate(method='time').bfill().ffill()

    # features: rows=basins, cols=timepoints
    X = combined.T.values

    # standardize features (per column of X -> timepoints are features)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA - follow a two-step process:
    # 1) compute PCA with no component limit to inspect explained variance
    # 2) choose n_components that explain ~90% (or nearest threshold) and re-run PCA
    pca_full = PCA()  # no n_components -> keep full SVD
    X_pca_full = pca_full.fit_transform(X_scaled)

    # explained variance (fraction) and cumulative
    explained = pca_full.explained_variance_ratio_
    cumulative = np.cumsum(explained)

    # Save explained + cumulative as percent
    df_ev = pd.DataFrame({
        'pc': np.arange(1, len(explained) + 1),
        'explained_pct': 100.0 * explained,
        'cumulative_pct': 100.0 * cumulative
    })
    df_ev.to_csv(OUT_DIR / 'explained_variance.csv', index=False)

    # Plot cumulative explained variance and save
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(explained) + 1), cumulative, marker='o', linestyle='--')
    plt.title('Cumulative Explained Variance by PCA Components')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'pca_cumulative_variance.png', dpi=150)
    plt.close()

    # Choose number of components to retain (e.g., explaining >= 90% variance)
    try:
        n_components = int(np.argmax(cumulative >= 0.90) + 1)
    except Exception:
        n_components = min(10, X_scaled.shape[1])
    print(f"Number of components explaining >=90% variance: {n_components}")

    # Save a small summary CSV including the chosen number of components at 90%
    # and the actual cumulative variance captured by that many components.
    try:
        cumulative_at_n = float(cumulative[n_components - 1])
    except Exception:
        cumulative_at_n = float(cumulative[-1])

    df_summary = pd.DataFrame({
        'threshold': [0.90],
        'n_components_at_threshold': [n_components],
        'cumulative_variance_at_n': [float(cumulative_at_n)],
        'cumulative_variance_at_n_pct': [100.0 * float(cumulative_at_n)]
    })
    df_summary.to_csv(OUT_DIR / 'pca_summary.csv', index=False)

    # Recompute PCA with selected number of components
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)

    # save scores: one row per basin (only retained components)
    basins = combined.columns.tolist()
    df_scores = pd.DataFrame(X_pca, index=basins, columns=[f'PC{i+1}' for i in range(X_pca.shape[1])])
    df_scores.to_csv(OUT_DIR / 'pca_scores.csv')

    # NOTE: We intentionally DO NOT save PCA loadings or the PCA scatter here.
    # The user requested only the explained-variance PNG, the summary CSV,
    # the explained_variance.csv, and the pca_scores.csv. Removing loadings
    # and scatter reduces clutter and keeps PCA outputs minimal.

    print('PCA outputs written to', OUT_DIR)


if __name__ == '__main__':
    main()
