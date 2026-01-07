import re
import os
from glob import glob
import h5py
import numpy as np
import pandas as pd
import geopandas as gpd
from rasterio.features import rasterize
from rasterio.transform import from_origin
from datetime import datetime
import calendar
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# ---------- USER CONFIG ----------
IMERG_DIR = r"C:\Users\ethan\Documents\josh_code\precip_imerg"
BASINS_VECTOR = r"C:\Users\ethan\Documents\josh_code\HUC_2_shape\HUC_2_shape.shp"
BASIN_NAME_FIELD = "name"
OUTPUT_CSV = r"C:\Users\ethan\Documents\josh_code\imerg_basin_monthly_timeseries_huc2.csv"
# ----------------------------------

def days_in_month(date):
    return calendar.monthrange(date.year, date.month)[1]

def filename_to_date(fname):
    m = re.search(r'(\d{8})', os.path.basename(fname))
    if not m:
        raise ValueError("Couldn't parse date from filename: " + fname)
    return datetime.strptime(m.group(1), "%Y%m%d")

def compute_monthly_timeseries(imerg_dir, basins_vector, basin_name_field):
    print("Reading basins file...")
    basins = gpd.read_file(basins_vector)
    
    if basins.crs != "EPSG:4326":
        print("Reprojecting basins to EPSG:4326...")
        basins = basins.to_crs("EPSG:4326")

    files = sorted(glob(os.path.join(imerg_dir, "*.HDF5")) + glob(os.path.join(imerg_dir, "*.h5")) + glob(os.path.join(imerg_dir, "*.HDF")))
    if not files:
        raise ValueError("No IMERG files found in " + imerg_dir)
    
    print("Reading grid info from first IMERG file...")
    sample_file = files[0]
    with h5py.File(sample_file, "r") as f:
        # Detect variable paths
        if "Grid/precipitation" in f:
            precip_path = "Grid/precipitation"
            lat_path = "Grid/lat"
            lon_path = "Grid/lon"
        else:
            precip_path = "precipitation"
            lat_path = "lat"
            lon_path = "lon"

        lat = f[lat_path][()]
        lon = f[lon_path][()]

    # --- 1. COORDINATE FIXES ---
    # Convert from Pixel Centers (HDF5 default) to Pixel Corners (Rasterio requirement)
    # GPM IMERG is 0.1 degree resolution. 
    # If lat min is -89.95, the edge is -90.0.
    dx = np.abs(np.mean(np.diff(lon)))
    dy = np.abs(np.mean(np.diff(lat)))
    
    west = lon.min() - (dx / 2)
    north = lat.max() + (dy / 2)
    
    print("\n--- Transform Diagnostics ---")
    print(f"Grid Resolution: {dx:.3f} x {dy:.3f}")
    print(f"Computed Top-Left Corner: West={west:.3f}, North={north:.3f}")
    
    # 'from_origin' expects POSITIVE xsize and ysize. It handles the negative Y-step internally.
    transform = from_origin(west, north, dx, dy)
    print(f"Affine Transform:\n{transform}")

    print("Rasterizing basin masks...")
    out_shape = (len(lat), len(lon)) # (Rows, Cols)
    
    masks = {}
    for idx, row in basins.iterrows():
        name = row[basin_name_field]
        mask = rasterize(
            [(row.geometry, 1)],
            out_shape=out_shape,
            transform=transform,
            fill=0,
            dtype='uint8'
        )
        masks[name] = mask
    
    # Check if masks are empty
    total_pixels = sum([np.sum(m) for m in masks.values()])
    print(f"Total masked pixels across all basins: {total_pixels}")
    if total_pixels == 0:
        raise ValueError("Mask generation failed. No pixels were inside the basin geometries. Check CRS or Transform logic.")

    dates = []
    results = {name: [] for name in masks.keys()}
    
    # Flag to generate one debug plot
    debug_plotted = False

    for fpath in tqdm(files, desc="Processing IMERG files"):
        try:
            date = filename_to_date(fpath)
            dates.append(date)
            
            with h5py.File(fpath, "r") as f:
                precip = f[precip_path][()]
                
                # Standardize to (lat, lon)
                if precip.ndim == 3:
                    precip = np.squeeze(precip, axis=0).T 
                else:
                    precip = precip.T # (lon, lat) -> (lat, lon)
                
                # --- 2. ARRAY ORIENTATION FIX ---
                # HDF5 lat goes Low->High (Bottom-up). 
                # Rasterio/Transform goes High->Low (Top-down).
                # We must flip the array vertically.
                precip = np.flipud(precip)

                # Cleaning
                precip = precip * 24 * days_in_month(date)
                precip[precip < 0] = np.nan
                
                # --- DEBUG PLOT (First file only) ---
                if not debug_plotted:
                    print(f"\nSaving debug alignment image for {date.strftime('%Y-%m')}...")
                    plt.figure(figsize=(10, 6))
                    # Plot precip background (log scale for visibility)
                    plt.imshow(precip, extent=[west, lon.max()+(dx/2), lat.min()-(dy/2), north], 
                               vmin=0, vmax=300, cmap='Blues', alpha=0.6)
                    # Plot one mask contour to check alignment
                    first_basin = list(masks.keys())[0]
                    plt.imshow(masks[first_basin], extent=[west, lon.max()+(dx/2), lat.min()-(dy/2), north], 
                               cmap='Reds', alpha=0.5, interpolation='none')
                    plt.title(f"Alignment Check: {first_basin} (Red) over Precip (Blue)")
                    plt.savefig("debug_alignment.png")
                    plt.close()
                    debug_plotted = True

            for basin_name, mask in masks.items():
                masked_vals = precip[mask == 1]
                
                if masked_vals.size == 0 or np.all(np.isnan(masked_vals)):
                    results[basin_name].append(np.nan)
                else:
                    results[basin_name].append(np.nanmean(masked_vals))
                    
        except Exception as e:
            print(f"Error processing {fpath}: {e}")
            # Keep lists aligned
            if len(dates) > len(results[list(results.keys())[0]]):
                dates.pop()

    df = pd.DataFrame(results, index=pd.to_datetime(dates)).sort_index()
    df.index.name = "time"
    return df
if __name__ == "__main__":
    # Force recompute if you suspect cached data is bad, otherwise set to False
    recompute = False
    
    # 1. Load or Compute Data
    if os.path.exists(OUTPUT_CSV) and not recompute:
        print(f"Found existing data. Loading from {OUTPUT_CSV}...")
        df = pd.read_csv(OUTPUT_CSV, index_col='time', parse_dates=True)
        # Simple check to ensure we don't load a file full of NaNs
        if df.isnull().all().all():
            print("Cached data contains all NaNs. Recomputing...")
            recompute = True
    else:
        recompute = True

    if recompute:
        print("Running full time series analysis...")
        df = compute_monthly_timeseries(IMERG_DIR, BASINS_VECTOR, BASIN_NAME_FIELD)
        df.to_csv(OUTPUT_CSV)
        print("Save complete.")

    print("\n--- Processed Data Head ---")
    print(df.head())

    # Check validity before plotting
    if df.isnull().all().all():
        print("\nCRITICAL: All data is NaN. Cannot calculate correlations.")
    else:
        # ==========================================
        # 1. RAW DATA CORRELATION (New Addition)
        # ==========================================
        print("\nCalculating correlation matrix for RAW data...")
        raw_corr = df.corr().dropna(how='all', axis=0).dropna(how='all', axis=1)
        
        raw_corr_csv = OUTPUT_CSV.replace(".csv", "_raw_correlation_matrix.csv")
        raw_corr.to_csv(raw_corr_csv)
        print(f"Saved raw correlation matrix to {raw_corr_csv}")

        if not raw_corr.empty:
            plt.figure(figsize=(10, 8))
            sns.heatmap(
                raw_corr, 
                annot=True, 
                fmt=".2f", 
                cmap="viridis",  # Different colormap to distinguish from anomalies
                vmin=0,          # Raw precip correlation is rarely negative
                vmax=1
            )
            plt.title("Raw Precipitation Correlation (Seasonality Included)", fontsize=16)
            plt.xticks(rotation=45, ha="right")
            plt.yticks(rotation=0)
            plt.tight_layout()
            plt.show()

        # ==========================================
        # 2. ANOMALY CORRELATION (Existing Logic)
        # ==========================================
        print("\nCalculating anomalies (removing seasonality)...")
        climatology = df.groupby(df.index.month).mean()
        anomalies = df.copy()
        
        for month in climatology.index:
            if month in df.index.month:
                mask = df.index.month == month
                anomalies.loc[mask] = df.loc[mask] - climatology.loc[month].values

        anomalies_csv = OUTPUT_CSV.replace(".csv", "_anomalies.csv")
        anomalies.to_csv(anomalies_csv)
        print(f"Saved anomalies to {anomalies_csv}")

        print("Calculating correlation matrix for ANOMALIES...")
        anomaly_corr = anomalies.corr().dropna(how='all', axis=0).dropna(how='all', axis=1)
        
        anomaly_corr_csv = OUTPUT_CSV.replace(".csv", "_anomaly_correlation_matrix.csv")
        anomaly_corr.to_csv(anomaly_corr_csv)
        print(f"Saved anomaly correlation matrix to {anomaly_corr_csv}")

        if not anomaly_corr.empty:
            plt.figure(figsize=(10, 8))
            sns.heatmap(
                anomaly_corr, 
                annot=True, 
                fmt=".2f", 
                cmap="coolwarm", 
                vmin=-1, 
                vmax=1
            )
            plt.title("Precipitation Anomaly Correlation (Seasonality Removed)", fontsize=16)
            plt.xticks(rotation=45, ha="right")
            plt.yticks(rotation=0)
            plt.tight_layout()
            plt.show()