from pathlib import Path  # import Path class from pathlib module for filesystem paths
import numpy as np  # import numpy for numerical operations on arrays
import pandas as pd  # import pandas for DataFrame manipulation and CSV reading
import matplotlib.pyplot as plt  # import pyplot from matplotlib for plotting
from sklearn.preprocessing import StandardScaler  # import StandardScaler to standardize features
from sklearn.cluster import KMeans  # import KMeans for clustering
from sklearn.metrics import silhouette_score  # import silhouette_score to evaluate clustering
import os  # import os module for OS-related operations

# define base directory where this script lives, fallback for notebooks
HERE = Path(__file__).parent if '__file__' in globals() else Path.cwd()  # Path object representing current directory
INPUT_DIR = HERE / 'HUC4_mm_averages_csvs'  # define folder with input basin CSVs, Path / operator joins paths
OUT_DIR = HERE / 'kmeans_clustering_huc4' / 'outputs'  # define output folder path
OUT_DIR.mkdir(parents=True, exist_ok=True)  # create output directory, parents=True creates all intermediate dirs, exist_ok=True avoids error if it exists

# function to load all basin CSV files and combine into a single DataFrame
def load_huc4_matrix(input_dir: Path) -> pd.DataFrame:  # input_dir is a Path object, returns a DataFrame
    files = sorted(input_dir.glob('basin_*.csv'))  # glob finds all files matching pattern, sorted returns list
    if not files:  # check if list is empty
        raise FileNotFoundError(f"No basin CSVs found in {input_dir}")  # raise error with formatted string

    series = {}  # initialize empty dictionary to hold each basin's time series
    for f in files:  # iterate over each file in files
        df = pd.read_csv(f)  # read CSV into a DataFrame, df is now a pandas DataFrame

        # check for year and month columns to build datetime
        if {'year', 'month'}.issubset(df.columns):  # set.issubset checks if columns exist in DataFrame
            try:
                df['date'] = pd.to_datetime(dict(
                    year=df['year'].astype(int),  # convert year column to integers
                    month=df['month'].astype(int),  # convert month column to integers
                    day=1  # set day to 1 for all rows
                ))  # pd.to_datetime converts dict of arrays to datetime objects
            except Exception:  # catch any error in parsing
                continue  # skip this file if error occurs
        elif 'time' in df.columns:  # check if 'time' column exists
            df['date'] = pd.to_datetime(df['time'])  # parse 'time' column into datetime
        else:  # if no date information
            continue  # skip this file

        # detect value column to use
        cols_lower = [c.lower() for c in df.columns]  # list comprehension to convert all column names to lowercase
        valcol = next((df.columns[i] for i, c in enumerate(cols_lower) if c in ['average', 'mean', 'value', 'runoff']), None)  
        # next returns first matching column, enumerate gives index and value, fallback to None if not found

        if valcol is None:  # if no standard value column found
            numeric = df.select_dtypes(include=[np.number]).columns  # select numeric columns only
            numeric = [c for c in numeric if c not in ('year', 'month')]  # remove year/month from candidates
            if not numeric:  # if no numeric column remains
                continue  # skip file
            valcol = numeric[0]  # take first numeric column

        s = df.set_index('date')[valcol].sort_index()  # set date as index, select value column, sort by date
        series[f.stem.replace('basin_', '')] = s  # use file stem as basin name (remove 'basin_')

    combined = pd.concat(series, axis=1)  # combine all series horizontally into a DataFrame, columns = basins
    combined.sort_index(inplace=True)  # ensure rows are sorted by date
    return combined  # return combined DataFrame

# function to compute silhouette scores for KMeans clustering
def compute_silhouette_scores(X_scaled: np.ndarray, k_range: range):
    scores, labels_by_k = [], {}  # initialize list for scores and dict for labels
    n_samples = X_scaled.shape[0]  # number of rows (basins) in feature matrix

    for k in k_range:  # iterate over range of cluster numbers
        if k >= n_samples:  # skip if k >= number of basins
            scores.append(np.nan)  # append NaN to maintain index alignment
            labels_by_k[k] = np.array([])  # store empty array for labels
            print(f"Skipping k={k} (k >= n_samples={n_samples})")  # print warning
            continue  # skip to next k

        kmeans = KMeans(n_clusters=k, random_state=42, n_init=50)  # create KMeans object, n_init=50 runs algorithm 50 times
        labels = kmeans.fit_predict(X_scaled)  # fit model and predict cluster labels

        try:  
            score = silhouette_score(X_scaled, labels)  # calculate silhouette score
        except Exception:  # catch errors
            score = np.nan  # set score to NaN

        scores.append(score)  # append score to list
        labels_by_k[k] = labels  # save labels for this k
        print(f"Silhouette score for k={k}: {score:.4f}")  # print score with 4 decimal places

    return scores, labels_by_k  # return scores list and labels dictionary

# function to plot mean series per cluster
def plot_cluster_mean_series(combined_df: pd.DataFrame, labels, out_png):
    clusters = {}  # dictionary to group basins by cluster
    for basin, lab in zip(combined_df.columns, labels):  # zip columns and labels together
        clusters.setdefault(lab, []).append(basin)  # setdefault initializes list if key not present

    plt.figure(figsize=(8, 4))  # create figure with width 8, height 4 inches
    for lab, basins in sorted(clusters.items()):  # iterate over clusters in sorted order
        mean_series = combined_df[basins].mean(axis=1)  # calculate mean across basins (axis=1)
        plt.plot(combined_df.index, mean_series, label=f'Cluster {lab}')  # plot mean series

    plt.legend()  # add legend
    plt.title('Cluster Mean Monthly Series')  # add title
    plt.xlabel('Time')  # label x-axis
    plt.ylabel('Mean Value')  # label y-axis
    plt.tight_layout()  # adjust layout to prevent clipping
    plt.savefig(out_png, dpi=150)  # save figure to file with 150 dpi
    plt.close()  # close figure to free memory

# function to save plots of all individual basin series per cluster
def save_per_cluster_timeseries(combined_df, cluster_labels, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)  # ensure output directory exists
    clusters = {}  # initialize cluster dictionary
    for basin, lab in zip(combined_df.columns, cluster_labels):  # assign basins to clusters
        clusters.setdefault(lab, []).append(basin)

    for lab, basins in sorted(clusters.items()):  # iterate clusters
        plt.figure(figsize=(8, 4))  # create figure
        for b in basins:  # plot each basin
            plt.plot(combined_df.index, combined_df[b], alpha=0.6)  # alpha sets transparency
        plt.title(f'Cluster {lab} Series (n={len(basins)})')  # title includes cluster and count
        plt.xlabel('Time')  # label x-axis
        plt.ylabel('Value')  # label y-axis
        plt.tight_layout()  # adjust layout
        plt.savefig(out_dir / f'cluster_{lab}_timeseries.png', dpi=150)  # save figure
        plt.close()  # close figure

# main function controlling workflow
def main():
    combined = load_huc4_matrix(INPUT_DIR)  # load all basin CSVs into DataFrame
    combined = combined.interpolate(method='time').bfill().ffill()  # fill missing data using time interpolation, backfill, forward fill
    X = combined.T.values  # transpose DataFrame: rows = basins, columns = timepoints, convert to numpy array

    scaler = StandardScaler()  # create StandardScaler object
    X_scaled = scaler.fit_transform(X)  # fit scaler to data and transform, standardizes features to mean 0, std 1

    k_values = list(range(2, 11))  # list of k values to test
    silhouette_scores, _ = compute_silhouette_scores(X_scaled, k_values)  # compute silhouette scores

    # save silhouette scores to CSV
    df_scores = pd.DataFrame({'k': k_values, 'silhouette': silhouette_scores})  # create DataFrame
    df_scores.to_csv(OUT_DIR / 'silhouette_scores.csv', index=False)  # save to CSV without index

    # plot silhouette scores
    plt.figure(figsize=(8, 5))  # create figure
    plt.plot(k_values, silhouette_scores, marker='o')  # plot scores with markers
    plt.title('Silhouette Scores for Different K')  # title
    plt.xlabel('Number of clusters (k)')  # x-axis label
    plt.ylabel('Silhouette Score')  # y-axis label
    plt.grid(True)  # add grid
    plt.tight_layout()  # adjust layout
    plt.savefig(OUT_DIR / 'silhouette_scores.png', dpi=150)  # save figure
    plt.close()  # close figure

    valid_scores = [s if not np.isnan(s) else -np.inf for s in silhouette_scores]  # replace NaNs with -inf
    best_k = k_values[int(np.argmax(valid_scores))]  # select k with max silhouette score
    print(f"Best number of clusters by silhouette score: {best_k}")  # print best k

    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=50)  # create KMeans for best k
    cluster_labels = kmeans.fit_predict(X_scaled)  # fit and get cluster labels

    basins = combined.columns.tolist()  # get list of basin names
    pd.DataFrame({'basin': basins, 'cluster': cluster_labels}).to_csv(  # save cluster assignments
        OUT_DIR / 'clusters_huc4.csv', index=False
    )

    plot_cluster_mean_series(combined, cluster_labels, OUT_DIR / 'cluster_mean_timeseries.png')  # plot mean series per cluster
    save_per_cluster_timeseries(combined, cluster_labels, OUT_DIR / 'cluster_timeseries')  # save all individual basin plots

    print(f"Clustering outputs written to {OUT_DIR}")  # print completion message

if __name__ == '__main__':  # ensures main() runs only if script is executed directly
    main()  # call main function


