# SIH26227 Analyst Prototype

## What this prototype does

- Uploads two temporal satellite/EO images (T1/T2)
- Optional ECC-based co-registration
- Image quality and alignment indicators
- Hybrid RGB + edge change map
- Edge comparison
- Connected-component change-object extraction
- Bounding-box visualization
- Change-object CSV export
- Analyst review status + notes
- JSON evidence/audit record
- ChangeFormer checkpoint download path
- Explicit false-alarm warnings

## Run

    python -m venv .venv
    .venv\Scripts\activate
    pip install -r requirements.txt
    streamlit run app.py

## Important model note

The guaranteed path in `app.py` is deterministic CV evidence. This is
intentional: a random or unrelated pretrained checkpoint must not be presented
as a validated satellite change detector.

The official ChangeFormer LEVIR-CD checkpoint is referenced in
`models/download_models.py`. After downloading it, connect the official model
implementation through `models/changeformer_adapter.py`.

## Input guidance

Best results require:
- same AOI
- same or comparable GSD/resolution
- accurate temporal registration
- limited cloud/haze
- preferably comparable season and illumination
- GeoTIFF/COG support can be added with rasterio/GDAL in the next version

## Scope

This is a research/demo prototype for SIH26227. It does not claim classified
imagery access or operational intelligence accuracy.
