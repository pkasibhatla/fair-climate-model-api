from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import numpy as np
import pandas as pd
import io
import os
from fair.forward import fair_scm
from fair.RCPs import rcp45, rcp85
from fair.SSPs import ssp119, ssp126, ssp245, ssp370, ssp460, ssp585
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="FaIR Climate Model API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

F_2XCO2 = 3.7
START_YEAR = 1765

# ---------------------------------------------------------------------------
# Scenario lookup
# All modules have identical (736, 40) emissions arrays, same column order.
# ---------------------------------------------------------------------------
SCENARIO_MODULES = {
    "ssp119": ssp119,
    "ssp126": ssp126,
    "ssp245": ssp245,
    "ssp370": ssp370,
    "ssp460": ssp460,
    "ssp585": ssp585,
    "rcp45":  rcp45,
    "rcp85":  rcp85,
}
SSP_SCENARIOS = ["ssp119", "ssp126", "ssp245", "ssp370", "ssp460", "ssp585"]
RCP_SCENARIOS = ["rcp45", "rcp85"]

def get_scenario_emissions(scenario: str) -> np.ndarray:
    mod = SCENARIO_MODULES.get(scenario)
    if mod is None:
        raise ValueError(f"Unknown scenario '{scenario}'")
    return mod.Emissions.emissions.copy()

# ---------------------------------------------------------------------------
# Species info — FaIR column index -> metadata
# Column order verified from rcp45.py source earlier in this session.
# ---------------------------------------------------------------------------
SPECIES_INFO = {
    1:  {"name": "co2_fossil", "label": "CO2 Fossil",   "units": "GtC/yr"},
    2:  {"name": "co2_land",   "label": "CO2 Land Use", "units": "GtC/yr"},
    3:  {"name": "ch4",        "label": "CH4",          "units": "MtCH4/yr"},
    4:  {"name": "n2o",        "label": "N2O",          "units": "MtN2/yr"},
    5:  {"name": "sox",        "label": "SOx",          "units": "MtS/yr"},
    6:  {"name": "co",         "label": "CO",           "units": "MtCO/yr"},
    7:  {"name": "nmvoc",      "label": "NMVOC",        "units": "Mt/yr"},
    8:  {"name": "nox",        "label": "NOx",          "units": "MtN/yr"},
    9:  {"name": "bc",         "label": "BC",           "units": "Mt/yr"},
    10: {"name": "oc",         "label": "OC",           "units": "Mt/yr"},
    11: {"name": "nh3",        "label": "NH3",          "units": "Mt/yr"},
    12: {"name": "cf4",        "label": "CF4",          "units": "kt/yr"},
    13: {"name": "c2f6",       "label": "C2F6",         "units": "kt/yr"},
    14: {"name": "c6f14",      "label": "C6F14",        "units": "kt/yr"},
    15: {"name": "hfc23",      "label": "HFC-23",       "units": "kt/yr"},
    16: {"name": "hfc32",      "label": "HFC-32",       "units": "kt/yr"},
    17: {"name": "hfc43_10",   "label": "HFC-43-10",    "units": "kt/yr"},
    18: {"name": "hfc125",     "label": "HFC-125",      "units": "kt/yr"},
    19: {"name": "hfc134a",    "label": "HFC-134a",     "units": "kt/yr"},
    20: {"name": "hfc143a",    "label": "HFC-143a",     "units": "kt/yr"},
    21: {"name": "hfc227ea",   "label": "HFC-227ea",    "units": "kt/yr"},
    22: {"name": "hfc245fa",   "label": "HFC-245fa",    "units": "kt/yr"},
    23: {"name": "sf6",        "label": "SF6",          "units": "kt/yr"},
    24: {"name": "cfc11",      "label": "CFC-11",       "units": "kt/yr"},
    25: {"name": "cfc12",      "label": "CFC-12",       "units": "kt/yr"},
    26: {"name": "cfc113",     "label": "CFC-113",      "units": "kt/yr"},
    27: {"name": "cfc114",     "label": "CFC-114",      "units": "kt/yr"},
    28: {"name": "cfc115",     "label": "CFC-115",      "units": "kt/yr"},
    29: {"name": "carb_tet",   "label": "Carbon Tet",   "units": "kt/yr"},
    30: {"name": "mcf",        "label": "MCF",          "units": "kt/yr"},
    31: {"name": "hcfc22",     "label": "HCFC-22",      "units": "kt/yr"},
    32: {"name": "hcfc141b",   "label": "HCFC-141b",    "units": "kt/yr"},
    33: {"name": "hcfc142b",   "label": "HCFC-142b",    "units": "kt/yr"},
    34: {"name": "halon1211",  "label": "Halon-1211",   "units": "kt/yr"},
    35: {"name": "halon1202",  "label": "Halon-1202",   "units": "kt/yr"},
    36: {"name": "halon1301",  "label": "Halon-1301",   "units": "kt/yr"},
    37: {"name": "halon2402",  "label": "Halon-2402",   "units": "kt/yr"},
    38: {"name": "ch3br",      "label": "CH3Br",        "units": "kt/yr"},
    39: {"name": "ch3cl",      "label": "CH3Cl",        "units": "kt/yr"},
}

# FaIR C output column order — verified against 2020 RCP4.5 values
CONCENTRATION_INFO = {
    0:  {"name": "co2",      "label": "CO2",        "units": "ppm"},
    1:  {"name": "ch4",      "label": "CH4",        "units": "ppb"},
    2:  {"name": "n2o",      "label": "N2O",        "units": "ppb"},
    3:  {"name": "cf4",      "label": "CF4",        "units": "ppt"},
    4:  {"name": "c2f6",     "label": "C2F6",       "units": "ppt"},
    5:  {"name": "c6f14",    "label": "C6F14",      "units": "ppt"},
    6:  {"name": "hfc23",    "label": "HFC-23",     "units": "ppt"},
    7:  {"name": "hfc32",    "label": "HFC-32",     "units": "ppt"},
    8:  {"name": "hfc43_10", "label": "HFC-43-10",  "units": "ppt"},
    9:  {"name": "hfc125",   "label": "HFC-125",    "units": "ppt"},
    10: {"name": "hfc134a",  "label": "HFC-134a",   "units": "ppt"},
    11: {"name": "hfc143a",  "label": "HFC-143a",   "units": "ppt"},
    12: {"name": "hfc227ea", "label": "HFC-227ea",  "units": "ppt"},
    13: {"name": "hfc245fa", "label": "HFC-245fa",  "units": "ppt"},
    14: {"name": "sf6",      "label": "SF6",        "units": "ppt"},
    15: {"name": "cfc11",    "label": "CFC-11",     "units": "ppt"},
    16: {"name": "cfc12",    "label": "CFC-12",     "units": "ppt"},
    17: {"name": "cfc113",   "label": "CFC-113",    "units": "ppt"},
    18: {"name": "cfc114",   "label": "CFC-114",    "units": "ppt"},
    19: {"name": "cfc115",   "label": "CFC-115",    "units": "ppt"},
    20: {"name": "carb_tet", "label": "Carbon Tet", "units": "ppt"},
    21: {"name": "mcf",      "label": "MCF",        "units": "ppt"},
    22: {"name": "hcfc22",   "label": "HCFC-22",    "units": "ppt"},
    23: {"name": "hcfc141b", "label": "HCFC-141b",  "units": "ppt"},
    24: {"name": "hcfc142b", "label": "HCFC-142b",  "units": "ppt"},
    25: {"name": "halon1211","label": "Halon-1211",  "units": "ppt"},
    26: {"name": "halon1202","label": "Halon-1202",  "units": "ppt"},
    27: {"name": "halon1301","label": "Halon-1301",  "units": "ppt"},
    28: {"name": "halon2402","label": "Halon-2402",  "units": "ppt"},
    29: {"name": "ch3br",    "label": "CH3Br",      "units": "ppt"},
    30: {"name": "ch3cl",    "label": "CH3Cl",      "units": "ppt"},
}

# ---------------------------------------------------------------------------
# Climate model helpers
# ---------------------------------------------------------------------------
class EmissionsInput(BaseModel):
    scenario: str = "ssp245"
    variability_sigma: float = 0.1
    n_ensemble: int = 100
    lambda_mean: float = 1.2
    lambda_std: float = 0.3
    gamma_mean: float = 0.7
    gamma_std: float = 0.2
    ar2_phi1: float = 0.6
    ar2_phi2: float = -0.2


def lambda_gamma_to_tcrecs(lam, gam):
    ecs = F_2XCO2 / lam
    tcr = F_2XCO2 / (lam + gam)
    return tcr, ecs


def generate_ar2(n, phi1, phi2, sigma, rng):
    V = np.zeros(n)
    eps = rng.normal(0, sigma, n)
    V[0] = eps[0]
    V[1] = phi1 * V[0] + eps[1]
    for t in range(2, n):
        V[t] = phi1 * V[t-1] + phi2 * V[t-2] + eps[t]
    return V


def zero_out_anthropogenic(emissions: np.ndarray) -> np.ndarray:
    """Zero all species columns (1-39). FaIR's own natural CH4/N2O
    parameter is left at its default in both actual and counterfactual
    runs, so any residual drift is identical and cancels in the difference."""
    zeroed = emissions.copy()
    zeroed[:, 1:] = 0.0
    return zeroed


def run_ensemble(emissions: np.ndarray, inputs_dict: dict) -> dict:
    """Run FaIR ensemble for both actual and zero-anthropogenic-emissions
    counterfactual, with paired AR(2) variability so the difference panel
    cleanly shows the forced anthropogenic signal."""
    lambda_mean = inputs_dict["lambda_mean"]
    lambda_std  = inputs_dict["lambda_std"]
    gamma_mean  = inputs_dict["gamma_mean"]
    gamma_std   = inputs_dict["gamma_std"]
    n_ensemble  = inputs_dict["n_ensemble"]
    ar2_phi1    = inputs_dict["ar2_phi1"]
    ar2_phi2    = inputs_dict["ar2_phi2"]
    variability_sigma = inputs_dict["variability_sigma"]

    zero_emissions = zero_out_anthropogenic(emissions)
    nt = emissions.shape[0]
    years = list(range(START_YEAR, START_YEAR + nt))

    # Deterministic forced signals at default lambda/gamma
    tcr_def, ecs_def = lambda_gamma_to_tcrecs(lambda_mean, gamma_mean)
    C,  F,  T  = fair_scm(emissions=emissions,      tcrecs=np.array([tcr_def, ecs_def]))
    C0, F0, T0 = fair_scm(emissions=zero_emissions, tcrecs=np.array([tcr_def, ecs_def]))

    # Sample lambda and gamma jointly with correlation
    correlation = 0.5
    cov = [
        [lambda_std**2,                              correlation * lambda_std * gamma_std],
        [correlation * lambda_std * gamma_std,        gamma_std**2],
    ]
    samples = np.random.multivariate_normal([lambda_mean, gamma_mean], cov, n_ensemble)
    lambda_values = np.clip(samples[:, 0], 0.5, 3.0)
    gamma_values  = np.clip(samples[:, 1], 0.1, 2.0)

    ensemble, ensemble_zero, ecs_list, tcr_list = [], [], [], []

    for i, (lam, gam) in enumerate(zip(lambda_values, gamma_values)):
        tcr, ecs = lambda_gamma_to_tcrecs(lam, gam)
        try:
            Ce,  Fe,  Te  = fair_scm(emissions=emissions,      tcrecs=np.array([tcr, ecs]))
            Ce0, Fe0, Te0 = fair_scm(emissions=zero_emissions, tcrecs=np.array([tcr, ecs]))
            rng = np.random.default_rng(seed=int(1_000_003 * (i + 1)) % (2**32 - 1))
            V = generate_ar2(nt, ar2_phi1, ar2_phi2, variability_sigma, rng)
            ensemble.append((Te  + V).tolist())
            ensemble_zero.append((Te0 + V).tolist())
            ecs_list.append(round(ecs, 2))
            tcr_list.append(round(tcr, 2))
        except Exception:
            continue

    arr  = np.array(ensemble)
    arr0 = np.array(ensemble_zero)
    diff = arr - arr0

    def pct(a, p):
        return np.percentile(a, p, axis=0).tolist()

    # Per-species emissions and concentrations for both runs
    emissions_by_species      = {}
    emissions_zero_by_species = {}
    for col, info in SPECIES_INFO.items():
        emissions_by_species[info["name"]]      = emissions[:, col].tolist()
        emissions_zero_by_species[info["name"]] = zero_emissions[:, col].tolist()
    emissions_by_species["co2_total"]      = (emissions[:, 1] + emissions[:, 2]).tolist()
    emissions_zero_by_species["co2_total"] = [0.0] * nt

    concentrations_by_species      = {}
    concentrations_zero_by_species = {}
    for col, info in CONCENTRATION_INFO.items():
        concentrations_by_species[info["name"]]      = C[:,  col].tolist()
        concentrations_zero_by_species[info["name"]] = C0[:, col].tolist()

    return {
        "years": years,
        "emissions_by_species":            emissions_by_species,
        "emissions_zero_by_species":       emissions_zero_by_species,
        "concentrations_by_species":       concentrations_by_species,
        "concentrations_zero_by_species":  concentrations_zero_by_species,
        "temperature_C":      T.tolist(),
        "temperature_C_zero": T0.tolist(),
        "forcing_Wm2":        F.tolist(),
        "ensemble":           ensemble,
        "ensemble_zero":      ensemble_zero,
        "percentile_5":  pct(arr, 5),
        "percentile_25": pct(arr, 25),
        "percentile_50": pct(arr, 50),
        "percentile_75": pct(arr, 75),
        "percentile_95": pct(arr, 95),
        "percentile_5_zero":  pct(arr0, 5),
        "percentile_25_zero": pct(arr0, 25),
        "percentile_50_zero": pct(arr0, 50),
        "percentile_75_zero": pct(arr0, 75),
        "percentile_95_zero": pct(arr0, 95),
        "diff_percentile_5":  pct(diff, 5),
        "diff_percentile_25": pct(diff, 25),
        "diff_percentile_50": pct(diff, 50),
        "diff_percentile_75": pct(diff, 75),
        "diff_percentile_95": pct(diff, 95),
        "diff_forced":        (T - T0).tolist(),
        "ecs_default": round(ecs_def, 2),
        "tcr_default": round(tcr_def, 2),
        "ecs_ensemble": ecs_list,
        "tcr_ensemble": tcr_list,
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.post("/run")
def run_model(inputs: EmissionsInput):
    try:
        emissions = get_scenario_emissions(inputs.scenario)
    except ValueError as e:
        return {"error": str(e)}
    return run_ensemble(emissions, inputs.dict())


@app.post("/run_custom")
async def run_custom(
    file: UploadFile = File(...),
    variability_sigma: float = 0.1,
    n_ensemble: int = 100,
    lambda_mean: float = 1.2,
    lambda_std: float = 0.3,
    gamma_mean: float = 0.7,
    gamma_std: float = 0.2,
    ar2_phi1: float = 0.6,
    ar2_phi2: float = -0.2,
):
    contents = await file.read()
    try:
        df = pd.read_excel(io.BytesIO(contents))
    except Exception:
        return {"error": "Could not read Excel file"}

    if "year" not in df.columns:
        return {"error": "Excel file must have a 'year' column"}

    name_to_col = {info["name"]: col for col, info in SPECIES_INFO.items()}
    provided = [c for c in df.columns if c != "year" and c in name_to_col]
    unknown  = [c for c in df.columns if c != "year" and c not in name_to_col]
    if unknown:
        return {"error": f"Unknown species columns: {unknown}. See /species for valid names."}
    if not provided:
        return {"error": "No recognised species columns found. See /species for valid names."}

    # Start from SSP2-4.5 as baseline
    emissions = get_scenario_emissions("ssp245")
    fair_years = np.arange(START_YEAR, START_YEAR + emissions.shape[0])

    for species_name in provided:
        col_idx = name_to_col[species_name]
        student_years = df["year"].values
        student_vals  = df[species_name].values.astype(float)
        mask = ~np.isnan(student_vals)
        if mask.sum() < 2:
            continue
        year_min = max(int(student_years[mask].min()), START_YEAR)
        year_max = int(student_years[mask].max())
        for i, year in enumerate(fair_years):
            if year_min <= year <= year_max:
                emissions[i, col_idx] = np.interp(
                    year, student_years[mask], student_vals[mask]
                )

    params = {
        "variability_sigma": variability_sigma,
        "n_ensemble": n_ensemble,
        "lambda_mean": lambda_mean,
        "lambda_std":  lambda_std,
        "gamma_mean":  gamma_mean,
        "gamma_std":   gamma_std,
        "ar2_phi1":    ar2_phi1,
        "ar2_phi2":    ar2_phi2,
    }
    return run_ensemble(emissions, params)


@app.get("/template")
def download_template():
    """Template Excel file: SSP2-4.5 values for all species, 1765-2100."""
    emissions = get_scenario_emissions("ssp245")
    fair_years = np.arange(START_YEAR, START_YEAR + emissions.shape[0])

    data = {"year": fair_years}
    for col, info in SPECIES_INFO.items():
        data[info["name"]] = emissions[:, col].round(6)

    df_out = pd.DataFrame(data)

    units_df = pd.DataFrame([
        {"species": info["name"], "label": info["label"], "units": info["units"]}
        for info in SPECIES_INFO.values()
    ])

    instructions_df = pd.DataFrame({"Instructions": [
        "Edit any species column to create your custom scenario.",
        "Only include columns for species you want to change — others keep SSP2-4.5 values.",
        "Do not change the 'year' column.",
        "Covers 1765-2100 so historical as well as future emissions are editable.",
        "Upload via the Run Custom button.",
        "See the 'units' sheet for units of each species.",
    ]})

    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df_out.to_excel(writer, index=False, sheet_name="emissions")
        instructions_df.to_excel(writer, index=False, sheet_name="instructions")
        units_df.to_excel(writer, index=False, sheet_name="units")
    buffer.seek(0)

    return StreamingResponse(
        buffer,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": "attachment; filename=fair_template.xlsx"},
    )


@app.get("/species")
def list_species():
    emissions_species = {
        info["name"]: {"label": info["label"], "units": info["units"]}
        for info in SPECIES_INFO.values()
    }
    emissions_species["co2_total"] = {"label": "CO2 Total (fossil+land)", "units": "GtC/yr"}
    concentration_species = {
        info["name"]: {"label": info["label"], "units": info["units"]}
        for info in CONCENTRATION_INFO.values()
    }
    return {"emissions": emissions_species, "concentrations": concentration_species}


@app.get("/scenarios")
def list_scenarios():
    return {"ssp": SSP_SCENARIOS, "rcp": RCP_SCENARIOS}


@app.get("/favicon.ico")
def favicon():
    return Response(status_code=204)


@app.get("/")
def index():
    return FileResponse(os.path.join(os.path.dirname(__file__), "index.html"))
