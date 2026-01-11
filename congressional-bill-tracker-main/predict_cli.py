import argparse
import importlib
import os
import sys
from datetime import datetime
from functools import lru_cache
from typing import Any, Optional, Sequence

import joblib
import numpy as np
import pandas as pd

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


def find_project_root(start_dir):
    candidates = [start_dir]
    nested = os.path.join(start_dir, "congressional-bill-tracker-main")
    if os.path.isdir(nested):
        candidates.append(nested)

    try:
        for name in os.listdir(start_dir):
            candidate = os.path.join(start_dir, name)
            if candidate in candidates:
                continue
            if os.path.isdir(candidate) and os.path.isfile(
                os.path.join(candidate, "src", "data_fetch.py")
            ):
                candidates.append(candidate)
    except OSError:
        pass

    for candidate in candidates:
        if os.path.isdir(os.path.join(candidate, "src")) and os.path.isdir(
            os.path.join(candidate, "models")
        ):
            return candidate

    for candidate in candidates:
        if os.path.isdir(os.path.join(candidate, "src")):
            return candidate

    return start_dir


PROJECT_ROOT = find_project_root(ROOT_DIR)
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

load_dotenv = None
try:
    from dotenv import load_dotenv as _load_dotenv

    load_dotenv = _load_dotenv
except ModuleNotFoundError:
    pass

if load_dotenv:
    load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

try:
    fetch_comprehensive_bill_data = importlib.import_module(
        "data_fetch"
    ).fetch_comprehensive_bill_data
except ModuleNotFoundError as exc:
    if exc.name == "dotenv":
        raise ModuleNotFoundError(
            "Missing dependency 'python-dotenv'. Install it with: "
            "python -m pip install python-dotenv"
        ) from exc
    raise ModuleNotFoundError(
        "Could not import data_fetch. Run from the repo root or move predict_cli.py "
        "into the project folder containing src/."
    ) from exc


class ReconstructedVotingEnsemble:
    def __init__(
        self,
        estimators: Sequence[Any],
        voting: str = "soft",
        weights: Optional[Sequence[float]] = None,
        classes: Optional[np.ndarray] = None,
    ):
        if voting != "soft":
            raise ValueError("Only soft voting is supported for reconstructed ensemble.")
        self.estimators = list(estimators)
        self.voting = voting
        self.weights = None if weights is None else np.asarray(list(weights), dtype=float)
        self.classes_ = classes

    def predict_proba(self, X):
        probas = [est.predict_proba(X) for est in self.estimators]
        stacked = np.stack(probas, axis=0)  # (n_estimators, n_samples, n_classes)

        if self.weights is None:
            return stacked.mean(axis=0)

        if len(self.weights) != stacked.shape[0]:
            raise ValueError(
                f"weights length ({len(self.weights)}) does not match "
                f"number of estimators ({stacked.shape[0]})."
            )

        weight_sum = float(self.weights.sum())
        if weight_sum <= 0:
            raise ValueError("weights must sum to a positive value.")

        w = (self.weights / weight_sum).reshape(-1, 1, 1)
        return (stacked * w).sum(axis=0)


def reconstruct_ensemble(rf_model, gb_model, lr_model, ensemble_config):
    return ReconstructedVotingEnsemble(
        estimators=[rf_model, gb_model, lr_model],
        voting=ensemble_config["voting"],
        weights=ensemble_config.get("weights"),
        classes=getattr(rf_model, "classes_", None),
    )


def load_model_stage(model_root, model_type, stage):
    model_dir = os.path.join(model_root, f"{model_type}_{stage}")
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    rf_model = joblib.load(os.path.join(model_dir, "rf_model.pkl"))
    components = joblib.load(os.path.join(model_dir, "components.pkl"))
    ensemble_config = joblib.load(os.path.join(model_dir, "ensemble_config.pkl"))

    ensemble = reconstruct_ensemble(
        rf_model,
        components["gb_model"],
        components["lr_model"],
        ensemble_config,
    )

    metadata = components["metadata"]
    return {
        "model": ensemble,
        "ensemble": ensemble,
        "rf_model": rf_model,
        "gb_model": components["gb_model"],
        "lr_model": components["lr_model"],
        "scaler": components["scaler"],
        "selector": components["selector"],
        "features": metadata["features"],
        "selected_features": metadata["selected_features"],
        "threshold": metadata["threshold"],
        "performance": metadata["performance"],
    }


@lru_cache(maxsize=1)
def load_models(project_root):
    models_dir = os.path.join(project_root, "models")
    if not os.path.exists(models_dir):
        raise FileNotFoundError("Models directory not found. Train models first.")

    metadata_package = joblib.load(os.path.join(models_dir, "metadata.pkl"))

    viability_models = {}
    passage_models = {}
    for stage in ["new_bill", "early_stage", "progressive"]:
        viability_models[stage] = load_model_stage(models_dir, "viability", stage)
        passage_models[stage] = load_model_stage(models_dir, "passage", stage)

    return {
        "viability_models": viability_models,
        "passage_models": passage_models,
        "label_encoders": metadata_package["label_encoders"],
    }


def determine_days_active(actions_df):
    if actions_df is None or actions_df.empty:
        return 1
    actions_df = actions_df.copy()
    actions_df["date"] = pd.to_datetime(actions_df["date"])
    first_action = actions_df["date"].min()
    if pd.isna(first_action):
        return 1
    days_active = (datetime.now() - first_action).days
    return max(days_active, 1)


def detect_progress_status(df, actions_df):
    has_passed_house = False
    has_passed_senate = False
    has_become_law = False

    if not df.empty and "status" in df.columns:
        status_text = str(df["status"].values[0]).lower()
        if "passed house" in status_text or "received in the senate" in status_text:
            has_passed_house = True
        if "passed senate" in status_text or "received in the house" in status_text:
            has_passed_senate = True
        if "became law" in status_text or "public law" in status_text:
            has_become_law = True

    if actions_df is not None and not actions_df.empty:
        action_texts = " ".join(actions_df["text"].str.lower().tolist())
        if "became public law" in action_texts or "became law" in action_texts:
            has_become_law = True
        if "passed house" in action_texts:
            has_passed_house = True
        if "passed senate" in action_texts:
            has_passed_senate = True

    return has_passed_house, has_passed_senate, has_become_law


def build_feature_row(df, subjects_data, metrics, days_active, activity_rate, congress):
    feature_data = {
        "sponsor_party": df["sponsor_parties"].values[0] if not df.empty else "Unknown",
        "sponsor_party_encoded": 0,
        "sponsor_count": len(df["sponsors"].values[0].split(",")) if not df.empty else 1,
        "original_cosponsor_count": metrics.get("original_cosponsor_count", 0),
        "cosponsor_count": df["cosponsor_count"].values[0] if not df.empty else 0,
        "month_introduced": datetime.now().month,
        "quarter_introduced": (datetime.now().month - 1) // 3 + 1,
        "is_election_year": int(datetime.now().year % 4 == 0),
        "title_length": (
            len(df["short_title"].values[0])
            if not df.empty
            and "short_title" in df.columns
            and df["short_title"].values[0]
            else len(df["title"].values[0])
            if not df.empty
            and "title" in df.columns
            and df["title"].values[0]
            else 100
        ),
        "title_word_count": (
            len(df["short_title"].values[0].split())
            if not df.empty
            and "short_title" in df.columns
            and df["short_title"].values[0]
            else len(df["title"].values[0].split())
            if not df.empty
            and "title" in df.columns
            and df["title"].values[0]
            else 20
        ),
        "title_complexity": 0,
        "subject_count": len(subjects_data.get("subjects", [])),
        "policy_area": df["policy_area"].values[0] if not df.empty else "Unknown",
        "policy_area_encoded": 0,
        "dem_total": metrics.get("dem_total", 0),
        "rep_total": metrics.get("rep_total", 0),
        "party_balance": 0,
        "party_dominance": 0,
        "bipartisan_score": metrics.get("bipartisan_score", 0),
        "has_bipartisan_support": int(df["is_bipartisan"].values[0]) if not df.empty else 0,
        "total_sponsors": 0,
        "is_fresh": int(days_active <= 30),
        "support_velocity": 0,
        "cosponsor_growth": 0,
        "days_active": days_active,
        "log_days_active": np.log1p(days_active),
        "sqrt_days_active": np.sqrt(days_active),
        "action_count": metrics.get("total_actions", 0),
        "activity_rate": activity_rate,
        "normalized_activity": metrics.get("total_actions", 0) / np.log1p(days_active),
        "early_activity": metrics.get("total_actions", 0) / (min(days_active, 30) + 1),
        "sustained_activity": metrics.get("total_actions", 0) / (min(days_active, 180) + 1),
        "is_active": int(days_active <= 90),
        "is_stale": int(days_active > 180),
        "committee_count": metrics.get("committee_count", 0),
        "has_committee": int(metrics.get("committee_count", 0) > 0),
        "multi_committee": int(metrics.get("committee_count", 0) >= 2),
        "committee_density": metrics.get("committee_count", 0) / max(days_active / 30, 1),
        "bipartisan_momentum": 0,
        "committee_activity": 0,
        "congress_numeric": congress,
        "is_recent_congress": int(congress >= 117),
    }

    feature_data["total_sponsors"] = (
        feature_data["sponsor_count"] + feature_data["cosponsor_count"]
    )
    feature_data["title_complexity"] = feature_data["title_length"] / (
        feature_data["title_word_count"] + 1
    )
    feature_data["party_balance"] = (
        feature_data["dem_total"] - feature_data["rep_total"]
    ) / (feature_data["total_sponsors"] + 1)
    feature_data["party_dominance"] = abs(feature_data["party_balance"])
    feature_data["support_velocity"] = feature_data["total_sponsors"] / np.sqrt(
        days_active
    )
    feature_data["cosponsor_growth"] = (
        feature_data["cosponsor_count"] - feature_data["original_cosponsor_count"]
    ) / max(days_active / 30, 1)
    feature_data["bipartisan_momentum"] = (
        feature_data["bipartisan_score"] * feature_data["normalized_activity"]
    )
    feature_data["committee_activity"] = (
        feature_data["committee_count"] * feature_data["activity_rate"]
    )

    return feature_data


def predict_bill(bill_id, congress, bill_type):
    if not os.getenv("CONGRESS_API_KEY"):
        raise ValueError(
            "CONGRESS_API_KEY is not set. Add it to the environment or "
            f"create {os.path.join(PROJECT_ROOT, '.env')} with that key."
        )

    model_package = load_models(PROJECT_ROOT)
    viability_models = model_package["viability_models"]
    passage_models = model_package["passage_models"]
    label_encoders = model_package["label_encoders"]

    comprehensive_data = fetch_comprehensive_bill_data(
        bill_id, congress=congress, bill_type=bill_type
    )
    if not comprehensive_data:
        raise ValueError("Could not fetch bill data. Check bill id/type/congress.")

    df = comprehensive_data["bill_info"]
    actions_df = comprehensive_data["actions"]
    subjects_data = comprehensive_data["subjects"]
    metrics = comprehensive_data["metrics"]

    days_active = determine_days_active(actions_df)
    activity_rate = metrics.get("total_actions", 0) / max(days_active, 1)

    if days_active <= 1:
        stage = "new_bill"
    elif days_active <= 30:
        stage = "early_stage"
    else:
        stage = "progressive"

    has_passed_house, has_passed_senate, has_become_law = detect_progress_status(
        df, actions_df
    )

    feature_data = build_feature_row(
        df, subjects_data, metrics, days_active, activity_rate, congress
    )

    if has_become_law:
        feature_data["action_count"] = max(feature_data["action_count"], 50)
        feature_data["committee_count"] = max(feature_data["committee_count"], 5)
        feature_data["is_stale"] = 0
        feature_data["activity_rate"] = max(feature_data["activity_rate"], 1.0)
        feature_data["normalized_activity"] = max(feature_data["normalized_activity"], 5.0)

    try:
        feature_data["sponsor_party_encoded"] = label_encoders["party"].transform(
            [feature_data["sponsor_party"]]
        )[0]
    except Exception:
        feature_data["sponsor_party_encoded"] = 0

    try:
        feature_data["policy_area_encoded"] = label_encoders["policy"].transform(
            [feature_data["policy_area"]]
        )[0]
    except Exception:
        feature_data["policy_area_encoded"] = 0

    bill_df = pd.DataFrame([feature_data])

    viability_model = viability_models[stage]
    passage_model = passage_models[stage]

    X_features = bill_df[viability_model["features"]].fillna(0)
    X_features = X_features.replace([np.inf, -np.inf], 0)
    X_scaled = viability_model["scaler"].transform(X_features)
    X_selected = X_scaled[:, viability_model["selector"].get_support()]
    X_selected_df = pd.DataFrame(
        X_selected, columns=viability_model["selected_features"]
    )

    try:
        rf_viability = viability_model["rf_model"].predict_proba(X_selected_df)[0, 1]
        gb_viability = viability_model["gb_model"].predict_proba(X_selected_df)[0, 1]
        lr_viability = viability_model["lr_model"].predict_proba(X_selected_df)[0, 1]
        viability_scores = [rf_viability, gb_viability, lr_viability]
    except Exception:
        viability_scores = []

    ensemble_viability = viability_model["model"].predict_proba(X_selected_df)[0, 1]
    is_viable = ensemble_viability >= 0.5

    passage_result = None
    if is_viable:
        X_passage = bill_df[passage_model["features"]].fillna(0)
        X_passage = X_passage.replace([np.inf, -np.inf], 0)
        X_passage_scaled = passage_model["scaler"].transform(X_passage)
        X_passage_selected = X_passage_scaled[:, passage_model["selector"].get_support()]
        X_passage_selected_df = pd.DataFrame(
            X_passage_selected, columns=passage_model["selected_features"]
        )

        try:
            rf_passage = passage_model["rf_model"].predict_proba(X_passage_selected_df)[
                0, 1
            ]
            gb_passage = passage_model["gb_model"].predict_proba(X_passage_selected_df)[
                0, 1
            ]
            lr_passage = passage_model["lr_model"].predict_proba(X_passage_selected_df)[
                0, 1
            ]
            passage_scores = [rf_passage, gb_passage, lr_passage]
        except Exception:
            passage_scores = []

        ensemble_passage = passage_model["model"].predict_proba(X_passage_selected_df)[
            0, 1
        ]

        passage_result = {
            "ensemble": ensemble_passage,
            "scores": passage_scores,
        }

    if has_become_law:
        ensemble_viability = 1.0
        passage_result = {"ensemble": 1.0, "scores": []}

    return {
        "stage": stage,
        "days_active": days_active,
        "has_passed_house": has_passed_house,
        "has_passed_senate": has_passed_senate,
        "has_become_law": has_become_law,
        "viability": {"ensemble": ensemble_viability, "scores": viability_scores},
        "passage": passage_result,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Predict bill viability and passage using local models."
    )
    parser.add_argument("bill_id", help="Bill number, e.g., 1234")
    parser.add_argument("--type", default="hr", choices=["hr", "s"], help="Bill type")
    parser.add_argument("--congress", type=int, default=118, help="Congress number")
    args = parser.parse_args()

    result = predict_bill(args.bill_id, args.congress, args.type)

    print(f"Stage: {result['stage']} (days_active={result['days_active']})")
    print(f"Viability: {result['viability']['ensemble']:.2%}")

    if result["passage"] is None:
        print("Passage: skipped (viability below 0.5)")
    else:
        print(f"Passage: {result['passage']['ensemble']:.2%}")

    if result["has_become_law"]:
        print("Note: bill already became law; scores forced to 100%.")
    elif result["has_passed_house"] or result["has_passed_senate"]:
        print("Note: bill already passed a chamber; features reflect current stage.")


if __name__ == "__main__":
    main()
