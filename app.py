# app.py — Ghost Worker/ Fraud Detection API on Hugging Face Spaces

import asyncio
import io
import json
import os
import pickle

import httpx
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

# ── Load models from Space repo ────────────────────────────────────────────
with open("iso_forest_model.pkl", "rb") as f:
    iso_model = pickle.load(f)
with open("xgb_model.pkl", "rb") as f:
    xgb_full = pickle.load(f)
with open("xgb_rinse_model.pkl", "rb") as f:
    xgb_rinse = pickle.load(f)
with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

print("All models loaded.")

# ── Keys from HF Space Secrets ─────────────────────────────────────────────
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
DIVINE_WEBHOOK = os.environ.get("WEBHOOK", "")

# ── Feature definitions ────────────────────────────────────────────────────
FULL_FEATURE_COLUMNS = [
    "Gender",
    "State",
    "Department",
    "Job Title",
    "Grade",
    "monthly_salary",
    "band_min",
    "band_max",
    "Bank Name",
    "Acct Age (Mths)",
    "attendance_pct",
    "Monthly Txns",
    "BVN Acct Count",
    "days_tenure",
    "days_since_active",
    "Months on Record",
    "Total Working Days",
    "Total Days Present",
    "Total Days Absent",
    "Total Days Leave",
    "Avg Presence %",
    "Months w/ Zero Attend.",
    "Max Consec. Zero Mths",
    "Avg CI Variance (mins)",
    "Avg Unique Locations",
]

RINSE_BASE_FEATURES = [
    "Gender",
    "State",
    "Department",
    "Job Title",
    "Grade",
    "monthly_salary",
    "band_min",
    "band_max",
    "Bank Name",
    "Acct Age (Mths)",
    "attendance_pct",
    "Monthly Txns",
    "BVN Acct Count",
    "days_tenure",
    "days_since_active",
]

RINSE_FEATURE_COLUMNS = RINSE_BASE_FEATURES + [
    "same_day_joiners",
    "salary_roundness",
    "salary_band_violation",
    "salary_deviation",
]

CATEGORICAL_COLS = ["Gender", "State", "Department", "Job Title", "Bank Name"]

ATTENDANCE_COLS = [
    "Months on Record",
    "Total Working Days",
    "Total Days Present",
    "Total Days Absent",
    "Total Days Leave",
    "Avg Presence %",
    "Months w/ Zero Attend.",
    "Max Consec. Zero Mths",
    "Avg CI Variance (mins)",
    "Avg Unique Locations",
]


# ── Column normalisation — exact match only (demo mode) ───────────────────
async def normalise_columns(df, target_cols):
    """Strip/Delete 'unknown' cols from the uploaded data
    @df:
    @target_cols: target no of cols to be left after deletion
    """
    df.columns = df.columns.str.strip()
    return df


# ── Salary normalisation ───────────────────────────────────────────────────
def normalise_salary(df):
    if "monthly_salary" not in df.columns:
        return df

    freq_col = None
    for col in df.columns:
        if col.lower().strip() in (
            "pay frequency",
            "frequency",
            "payment type",
            "pay type",
            "pay cycle",
            "salary type",
        ):
            freq_col = col
            break

    if freq_col:

        def convert(row):
            freq = str(row[freq_col]).lower().strip()
            sal = row["monthly_salary"]
            if freq in ("bi-weekly", "twice monthly", "bi-monthly", "fortnightly"):
                return sal * 2
            elif freq in ("hourly", "per hour", "hour"):
                hours = row.get("hours_per_week", 40)
                return sal * hours * 4.33
            elif freq in ("weekly",):
                return sal * 4.33
            else:
                return sal

        df["monthly_salary"] = df.apply(convert, axis=1)

    if df["monthly_salary"].dtype == object:
        df["monthly_salary"] = df["monthly_salary"].str.replace(",", "", regex=False)
        df["monthly_salary"] = pd.to_numeric(
            df["monthly_salary"], errors="coerce"
        ).fillna(0)

    return df


# ── Rinse feature engineering ──────────────────────────────────────────────
def engineer_rinse_features(df):
    """ """
    join_date_counts = df["days_tenure"].value_counts()
    df["same_day_joiners"] = df["days_tenure"].map(join_date_counts)
    df["salary_roundness"] = (df["monthly_salary"] % 10000 == 0).astype(int)
    df["salary_above_band"] = (df["monthly_salary"] > df["band_max"] * 1.3).astype(int)
    df["salary_below_band"] = (df["monthly_salary"] < df["band_min"] * 0.7).astype(int)
    df["salary_band_violation"] = (
        df["salary_above_band"] | df["salary_below_band"]
    ).astype(int)
    df["salary_deviation"] = (df["monthly_salary"] - df["band_min"]) / (
        df["band_max"] - df["band_min"] + 1
    )
    return df


# ── Tier assignment ────────────────────────────────────────────────────────
def assign_tier(score):
    """Assigns tier to employees. Employess can hold one of three statuses "VERIFIED", "REVIEW", or "GHOST FLAG"
    @score: score given to employee viz model analysis
    """

    if score >= 75:
        return "VERIFIED"
    elif score >= 50:
        return "REVIEW"
    else:
        return "GHOST FLAG"


# ── Risk signal extractor ──────────────────────────────────────────────────
def get_top_signals(row):
    signals = []
    if row.get("Total Days Absent", 0) > 100:
        signals.append("high number of absent days")
    if row.get("Avg Presence %", 100) < 20:
        signals.append("critically low attendance rate")
    if row.get("Months w/ Zero Attend.", 0) >= 3:
        signals.append("multiple months with zero attendance")
    if row.get("Acct Age (Mths)", 99) < 3:
        signals.append("very new bank account")
    if row.get("BVN Acct Count", 1) > 2:
        signals.append("BVN linked to multiple accounts")
    if row.get("salary_band_violation", 0) == 1:
        signals.append("salary outside grade band")
    if row.get("same_day_joiners", 0) > 5:
        signals.append("joined on same date as many others")
    if row.get("salary_roundness", 0) == 1:
        signals.append("suspiciously round salary amount")
    if not signals:
        signals.append("statistical anomaly across multiple payroll signals")
    return signals


# ── Gemini summariser (async) ──────────────────────────────────────────────
async def generate_explanation(employee_id, trust_score, tier, top_signals):
    """Fetches the summary from a googlke gemini model
    @employee_id: id of employee
    @trust_score: the trust score result for the employee (from the XBosst and IsolationForest models)
    @top_signals: top 2-3 suspicious signals detected from employee data
    """

    prompt = f"""You are a payroll audit assistant for an African HR platform.
Write a concise 2-3 sentence explanation for an HR manager about why this employee was flagged.
Be factual and professional. Do not use technical jargon.

Employee ID: {employee_id}
Trust Score: {trust_score}/100
Status: {tier}
Risk signals detected: {", ".join(top_signals)}

Write only the explanation. No headers, no bullet points."""

    try:
        async with httpx.AsyncClient(timeout=20) as client:
            response = await client.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}",
                headers={"Content-Type": "application/json"},
                json={
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {
                        "maxOutputTokens": 500,
                        "temperature": 0.3,
                        "stopSequences": ["\n\n"],  # stops after first paragraph
                    },
                },
            )
        raw = response.json()
        return raw["candidates"][0]["content"]["parts"][0]["text"].strip()
    except Exception as e:
        print(f"Gemini summariser failed for {employee_id}: {e} — using template")
        return (
            f"Employee {employee_id} has been flagged with a trust score of "
            f"{round(trust_score, 1)}/100. Key risk signals: {', '.join(top_signals)}. "
            f"HR review is recommended before processing this payment."
        )


# ── File parser ────────────────────────────────────────────────────────────
def parse_file(content: bytes, filename: str) -> pd.DataFrame:
    filename = filename.lower()
    if filename.endswith(".csv"):
        return pd.read_csv(io.BytesIO(content))
    elif filename.endswith((".xlsx", ".xls")):
        return pd.read_excel(io.BytesIO(content))
    else:
        raise ValueError("Unsupported file type. Send CSV or Excel.")


# ── Main scoring engine (async) ────────────────────────────────────────────
async def score_payroll(df_raw: pd.DataFrame):

    # Preserve employee ID
    id_col = None
    for col in df_raw.columns:
        if col.lower().strip() in (
            "employee id",
            "employee_id",
            "emp_id",
            "staff id",
            "staff_id",
        ):
            id_col = col
            break
    employee_ids = (
        df_raw[id_col].values if id_col else [f"EMP-{i}" for i in range(len(df_raw))]
    )

    # Detect scan type
    has_attendance = any(
        col.lower().strip() in [c.lower() for c in ATTENDANCE_COLS]
        for col in df_raw.columns
    )
    scan_type = "Full Scan" if has_attendance else "Rinse Scan"
    target_cols = FULL_FEATURE_COLUMNS if has_attendance else RINSE_FEATURE_COLUMNS

    # Normalise columns (async — includes DeepSeek fallback)
    df = await normalise_columns(df_raw.copy(), target_cols)

    # Normalise salary
    df = normalise_salary(df)

    # Engineer Rinse features if needed
    if not has_attendance:
        df = engineer_rinse_features(df)

    # Impute missing columns
    for col in target_cols:
        if col not in df.columns:
            df[col] = 0

    # Encode categoricals
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].astype(str)
            try:
                df[col] = le.fit_transform(df[col])
            except:
                df[col] = pd.Categorical(df[col]).codes

    # Score
    if has_attendance:
        df_features = df[FULL_FEATURE_COLUMNS].fillna(0)
        xgb_proba = xgb_full.predict_proba(df_features)[:, 1]
        iso_scores = iso_model.decision_function(df_features)
        iso_proba = 1 - (
            (iso_scores - iso_scores.min()) / (iso_scores.max() - iso_scores.min())
        )
        ensemble = (0.70 * xgb_proba) + (0.30 * iso_proba)
    else:
        df_features = df[RINSE_FEATURE_COLUMNS].fillna(0)
        ensemble = xgb_rinse.predict_proba(df_features)[:, 1]

    trust = ((1 - ensemble) * 100).round(1)

    # Generate explanations concurrently for flagged employees
    explanation_tasks = {}
    for i, (eid, ts) in enumerate(zip(employee_ids, trust)):
        tier = assign_tier(float(ts))
        if tier in ("GHOST FLAG", "REVIEW"):
            row = df.iloc[i].to_dict()
            signals = get_top_signals(row)
            explanation_tasks[i] = (
                asyncio.create_task(
                    generate_explanation(str(eid), float(ts), tier, signals)
                ),
                signals,
            )

    # Await all explanation tasks concurrently
    explanations = {}
    for i, (task, signals) in explanation_tasks.items():
        explanations[i] = (await task, signals)

    # Build final results
    results = []
    for i, (eid, ts) in enumerate(zip(employee_ids, trust)):
        tier = assign_tier(float(ts))
        record = {
            "employee_id": str(eid),
            "trust_score": float(ts),
            "payment_tier": tier,
            "action_required": tier != "VERIFIED",
            "scan_type": scan_type,
        }
        if i in explanations:
            record["explanation"] = explanations[i][0]
            record["risk_signals"] = explanations[i][1]

        results.append(record)

    verified = [r for r in results if r["payment_tier"] == "VERIFIED"]
    review = [r for r in results if r["payment_tier"] == "REVIEW"]
    flagged = [r for r in results if r["payment_tier"] == "GHOST FLAG"]

    return {
        "status": "success",
        "scan_type": scan_type,
        "total_records": len(results),
        "summary": {
            "verified_count": len(verified),
            "review_count": len(review),
            "ghost_flag_count": len(flagged),
            "interception_rate": round(
                (len(review) + len(flagged)) / len(results) * 100, 1
            )
            if results
            else 0,
        },
        "verified": verified,
        "review": review,
        "ghost_flags": flagged,
    }


# ── FastAPI app ────────────────────────────────────────────────────────────
app = FastAPI(
    title="Ghost Worker Detection API",
    description="Payroll anomaly detection — Isolation Forest + XGBoost + Gemini",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {
        "status": "online",
        "message": "Ghost Worker Detection API is live",
        "models": "Full Scan (Ensemble) + Rinse Scan (XGBoost)",
    }


@app.post("/score")
async def score(
    file: UploadFile = File(...),
    company_name: str = Form(default="Unknown Company"),
    location: str = Form(default="Unknown Location"),
    branch_name: str = Form(default="Unknown Branch"),
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    try:
        content = await file.read()
        df_raw = parse_file(content, file.filename)
        results = await score_payroll(df_raw)
        results["company_name"] = company_name
        results["location"] = location
        results["branch_name"] = branch_name
        return results
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
