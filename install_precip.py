import requests
from pathlib import Path

link_file = r"C:\Users\ethan\Documents\josh_code\subset_GPM_3IMERGM_07_20250912_070628_.txt"
out_dir = Path(r"C:\Users\ethan\Documents\josh_code\precip_imerg")
out_dir.mkdir(parents=True, exist_ok=True)

with open(link_file, "r") as f:
    urls = [line.strip() for line in f if line.strip()]

session = requests.Session()
session.auth = requests.utils.get_netrc_auth("https://urs.earthdata.nasa.gov")

for url in urls:
    fname = out_dir / Path(url).name
    if fname.exists():
        print(f"Already downloaded: {fname}")
        continue
    print(f"Downloading {url} → {fname}")
    with session.get(url, stream=True) as r:
        if r.status_code == 401:
            raise Exception("❌ Unauthorized. Check your .netrc file and Earthdata credentials.")
        r.raise_for_status()
        with open(fname, "wb") as f_out:
            for chunk in r.iter_content(chunk_size=8192):
                f_out.write(chunk)

print("✅ All downloads complete!")
