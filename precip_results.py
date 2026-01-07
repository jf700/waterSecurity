import pandas as pd
import matplotlib.pyplot as plt
import os
import re


CSV_PATH = r"C:\Users\ethan\Documents\josh_code\imerg_basin_monthly_timeseries_anomalies.csv"
OUTPUT_DIR = r"C:\Users\ethan\Documents\josh_code\anomaly_plots"


# Load the CSV
df = pd.read_csv(CSV_PATH, parse_dates=["time"], index_col="time")

# Ensure output folder exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def sanitize_filename(name: str) -> str:
    """Replace invalid filename characters with underscores."""
    return re.sub(r'[<>:"/\\|?*]', "_", name)

# Loop through each basin
for basin in df.columns:
    plt.figure(figsize=(10, 4))
    plt.plot(df.index, df[basin], color="black", linewidth=1)
    plt.axhline(0, color="red", linestyle="--", linewidth=1)
    plt.title(f"Monthly Precipitation Anomalies – {basin}")
    plt.ylabel("Anomaly (mm/month)")
    plt.xlabel("Time")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    safe_name = sanitize_filename(basin)
    outfile = os.path.join(OUTPUT_DIR, f"{safe_name}_anomalies.png")
    plt.savefig(outfile, dpi=150)
    plt.close()

print(f"Saved plots to {OUTPUT_DIR}")
