"""Central configuration shared by training, explainability and the web app."""
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parent
DATA_DIR, MODEL_DIR, REPORT_DIR = ROOT / "data", ROOT / "models", ROOT / "reports"
SEED = 42
DATA_SOURCE = os.getenv("INSUREXPLAIN_DATA_SOURCE", "analytics_vidhya")

NUMERIC = ["customer_age", "vehicle_age", "annual_mileage", "driving_experience",
           "previous_claims", "previous_accidents", "years_insured", "vehicle_value"]
CATEGORICAL = ["vehicle_type", "policy_type", "region_risk"]
FEATURES = NUMERIC + CATEGORICAL
TARGET = "claim_next_12m"
AUDIT_ONLY = ["gender"]          # kept ONLY for fairness auditing, never a model input

CATEGORY_VALUES = {
    "vehicle_type": ["Hatchback", "Sedan", "SUV", "Sports", "Van"],
    "policy_type": ["Third-Party", "Comprehensive"],
    "region_risk": ["Low", "Medium", "High"],
}
# (min, max) validation ranges used by the API
RANGES = {
    "customer_age": (18, 90), "vehicle_age": (0, 30), "annual_mileage": (500, 60000),
    "driving_experience": (0, 70), "previous_claims": (0, 10), "previous_accidents": (0, 10),
    "years_insured": (0, 60), "vehicle_value": (500, 150000),
}
LABELS = {
    "customer_age": "Customer age", "vehicle_age": "Vehicle age", "annual_mileage": "Annual mileage",
    "driving_experience": "Driving experience", "previous_claims": "Previous claim count",
    "previous_accidents": "Previous accident count", "years_insured": "Years insured",
    "vehicle_value": "Vehicle value", "vehicle_type": "Vehicle type", "policy_type": "Policy type",
    "region_risk": "Geographic risk category",
}

# Documented, configurable rule that converts probability -> category (edit here).
RISK_THRESHOLDS = {"low_below": 0.08, "high_from": 0.20}

# Features that are NOT used as customer-facing "reasons" in the explanation notice, because they
# are sensitive or proxy-prone in many jurisdictions. They still appear in the technical explanation.
NOTICE_EXCLUDED = ["customer_age", "region_risk"]

if DATA_SOURCE == "analytics_vidhya":
    TARGET = "is_claim"
    DATA_FILE = DATA_DIR / "analytics_vidhya" / "train.csv"
    NUMERIC = [
        "policy_tenure", "age_of_car", "age_of_policyholder", "population_density",
        "make", "airbags", "displacement", "cylinder", "gear_box",
        "turning_radius", "length", "width", "height", "gross_weight", "ncap_rating",
    ]
    CATEGORICAL = [
        "area_cluster", "segment", "model", "fuel_type", "max_torque", "max_power",
        "engine_type", "is_esc", "is_adjustable_steering", "is_tpms",
        "is_parking_sensors", "is_parking_camera", "rear_brakes_type",
        "transmission_type", "steering_type", "is_front_fog_lights",
        "is_rear_window_wiper", "is_rear_window_washer", "is_rear_window_defogger",
        "is_brake_assist", "is_power_door_locks", "is_central_locking",
        "is_power_steering", "is_driver_seat_height_adjustable",
        "is_day_night_rear_view_mirror", "is_ecw", "is_speed_alert",
    ]
    FEATURES = NUMERIC + CATEGORICAL
    AUDIT_ONLY = []
    CATEGORY_VALUES = {
        "area_cluster": [f"C{i}" for i in range(1, 23)],
        "segment": ["A", "C1", "C2", "B2", "B1", "Utility"],
        "model": [f"M{i}" for i in range(1, 12)],
        "fuel_type": ["CNG", "Petrol", "Diesel"],
        "max_torque": ["60Nm@3500rpm", "113Nm@4400rpm", "91Nm@4250rpm", "250Nm@2750rpm", "200Nm@3000rpm", "82.1Nm@3400rpm", "200Nm@1750rpm", "85Nm@3000rpm", "170Nm@4000rpm"],
        "max_power": ["40.36bhp@6000rpm", "88.50bhp@6000rpm", "67.06bhp@5500rpm", "113.45bhp@4000rpm", "88.77bhp@4000rpm", "55.92bhp@5300rpm", "97.89bhp@3600rpm", "61.68bhp@6000rpm", "118.36bhp@5500rpm"],
        "engine_type": ["F8D Petrol Engine", "1.2 L K12N Dualjet", "1.0 SCe", "1.5 L U2 CRDi", "1.5 Turbocharged Revotorq", "K Series Dual jet", "1.2 L K Series Engine", "K10C", "i-DTEC", "G12B", "1.5 Turbocharged Revotron"],
        "is_esc": ["No", "Yes"], "is_adjustable_steering": ["No", "Yes"], "is_tpms": ["No", "Yes"],
        "is_parking_sensors": ["Yes", "No"], "is_parking_camera": ["No", "Yes"],
        "rear_brakes_type": ["Drum", "Disc"], "transmission_type": ["Manual", "Automatic"],
        "steering_type": ["Power", "Electric", "Manual"],
    }
    for _name in [
        "is_esc", "is_adjustable_steering", "is_tpms", "is_parking_sensors",
        "is_parking_camera", "is_front_fog_lights", "is_rear_window_wiper",
        "is_rear_window_washer", "is_rear_window_defogger", "is_brake_assist",
        "is_power_door_locks", "is_central_locking", "is_power_steering",
        "is_driver_seat_height_adjustable", "is_day_night_rear_view_mirror",
        "is_ecw", "is_speed_alert",
    ]:
        CATEGORY_VALUES[_name] = ["No", "Yes"]
    LABELS = {f: f.replace("_", " ").title() for f in FEATURES}
    RANGES = {
        "policy_tenure": (0, 2), "age_of_car": (0, 1), "age_of_policyholder": (0, 1),
        "population_density": (290, 73430), "make": (1, 5), "airbags": (1, 6),
        "displacement": (796, 1498), "cylinder": (3, 4), "gear_box": (5, 6),
        "turning_radius": (4.5, 5.2), "length": (3445, 4300), "width": (1475, 1811),
        "height": (1475, 1825), "gross_weight": (1051, 1720), "ncap_rating": (0, 5),
    }
    RISK_THRESHOLDS = {"low_below": 0.05, "high_from": 0.15}
    NOTICE_EXCLUDED = []
    DATA_NOTE = "Trained on the Analytics Vidhya vehicle-insurance claim dataset."
else:
    DATA_FILE = DATA_DIR / "insurance_dataset.csv"
    DATA_NOTE = "SYNTHETIC demonstration data - results show the workflow, not real-world insurance accuracy."
