# marswalk-model
Marswalk Model for NASA Space Apps (Jakarta) - challenge: **Interplanetary Survival Guide: Martian Map**.

A planner for one Mars surface excursion (a Marswalk): it layers NASA data on a local grid, finds a safe route to science stops, and gives a GO / CAUTION / NO-GO verdict.

## NASA data this project needs
The full list, with dataset IDs, source pages and status, is in [`data/manifest.yaml`](data/manifest.yaml).

| Layer | Data type | Source | Used for |
| --- | --- | --- | --- |
| `Z` elevation (-> slope, hillshade) | topographic | MRO HiRISE DTM (`MRO-M-HIRISE-5-DTM-V1.0`); regional fallbacks: HRSC/MOLA blend, 30 m Gale HRSC mosaic | slope limit, route cost, terrain shielding |
| `SPEC` hydrated minerals | spectral | MRO CRISM MTRDR (targeted), MRDR (fallback) | science value of stops |
| `TI` thermal inertia | thermal | Mars Odyssey THEMIS (via MarsTherm), MGS TES (coarse) | surface hardness, temperature |
| `ROCK` boulders | optical | HiRISE RDR images; IRTM/TES rock abundance | blocked steps |
| `DOSE` radiation | radiation | MSL RAD RDR (`MSL-M-RAD-3-RDR-V1.0`) | dose estimate, verdict |
| Context | atmospheric, geology, basemap, POV | MARCI/MCS, REMS, MAVEN SEP, USGS geologic map, Mars Trek WMTS, rover raw images | dust, temperature, space weather, UI basemap, stop imagery |

The brief asks for at least three data types. This set covers topographic, spectral, thermal, optical and radiation.

## Target grid
All layers are clipped, reprojected and resampled to one grid: 8 m cells, 180 x 130 (1.44 x 1.04 km), centred on a chosen site (Mount Sharp foothills in Gale crater is suggested, not final).

## What is and isn't in this repo
- **Is:** the data manifest, a URL downloader (`scripts/fetch.py`), and this README.
- **Is not:** the actual rasters. The environment that prepared this branch could not reach NASA, USGS or PDS servers, so nothing was downloaded. `data/raw/` and `data/processed/` are git-ignored because the files are large.
- `status: verified` in the manifest means the source page showed up in a web search on 2026-11-14. `to-confirm` means it came from project notes or general knowledge. Resolutions are left as `TODO` until someone reads each product page.
- `scripts/fetch.py` has only been checked with `--dry-run`, not against the real servers.

## How to get the data
1. Open the pages in `data/manifest.yaml` and pick files covering your chosen site.
2. Copy `data/urls.example.txt` to `data/urls.txt` and add one `<layer_id> <url>` line per file.
3. `python scripts/fetch.py data/urls.txt` (add `--dry-run` first to preview).
4. Next step (not written yet): a script that resamples each layer onto the 8 m grid and saves `Z, TI, SPEC, ROCK, DOSE` arrays plus metadata (source, resolution, date).

## Known gaps
- No NASA source yet for EVA / suit numbers (walking speed, time cap, temperature limits).
- Atmosphere, space weather, geologic map and rover-image sources are named but not verified.
- Judging criteria, deadline and submission format for the Jakarta event are unknown.
