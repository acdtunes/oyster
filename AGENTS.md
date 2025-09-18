# Repository Guidelines

## Project Structure & Module Organization
- `R/` hosts the larval connectivity pipeline (configuration, data loading, dispersal modeling) executed through orchestrators in `scripts/`.
- Streamlit apps (`streamlit_app_clean.py`, `streamlit_app.py`) pull shared UI components from `app_modules/` and numerical kernels from `python_dispersal_model.py`.
- Raw inputs (NetCDF, GeoJSON, spreadsheets) live in `data/`, while derived CSVs and HTML diagnostics stay under `output/` grouped by region.

## Build, Test, and Development Commands
- Bootstrap Python tooling: `python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`.
- Launch Streamlit locally with `python3 -m streamlit run streamlit_app_clean.py`; switch to `streamlit_app.py` when extending the modular layout.
- Install the R stack (`install.packages(c("ncdf4","dplyr","ggplot2", ...))`) then run `Rscript scripts/run_full_analysis.R` or `Rscript scripts/run_st_marys_analysis.R`; quick map smoke tests run via `python3 -m pytest test_map_styles.py` (or `python3 test_map_styles.py`).

## Coding Style & Naming Conventions
- Follow PEP 8: four-space indentation, `snake_case` modules and functions, `CamelCase` classes, and `UPPER_CASE` constants; document units and coordinate frames in docstrings.
- Keep Streamlit components declarative—compose dashboards from helpers in `app_modules/` rather than embedding long logic in callbacks.
- Mirror tidyverse conventions in R: pipeline verbs, explicit column selections, and shared path variables at the top of each script to keep renders reproducible.

## Testing Guidelines
- Prefer `pytest` for new Python coverage; place `test_*.py` beside the feature or in a future `tests/` package so discovery still includes existing files.
- Attach HTML previews or screenshots when map styling changes, and describe reproduction steps (e.g., `python3 test_map_styles.py`).
- For R updates, rerun the relevant `Rscript` workflow and note changes in `output/<region>/reef_metrics.csv`; cache lightweight sample CSVs when full NetCDF runs are impractical.

## Commit & Pull Request Guidelines
- Write imperative, scoped commit subjects (`Enhance water boundary masking`) with optional bodies that explain data sources or parameter shifts.
- Bundle related changes per PR, reference issues (`Refs #123`), and list the commands or tests you ran with approximate runtimes.
- Flag regenerated artifacts, dependency updates, or schema changes, and keep large datasets out of version control—point reviewers to shared storage instead.

## Data Handling & Configuration Notes
- The main NetCDF file (`data/109516.nc`, ~114 MB) should be reused, not duplicated; document alternate sources if you introduce them.
- Secrets are not required today; load any future API tokens via environment variables and extend `.gitignore` before adding local config files.
