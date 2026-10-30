"""Generate a clearly-labelled SYNTHETIC motor-insurance dataset (academic demonstration only).

LABEL GENERATION (documented assumption): claim_next_12m ~ Bernoulli(sigmoid(z)), where z is a hand-written
linear risk score of the features below + noise. The model can therefore only learn these invented rules, not
real-world insurance behaviour. 'gender' does NOT enter z; it is slightly correlated with annual_mileage and
vehicle_type (a deliberate proxy effect so the fairness audit has something to examine).
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATA_DIR, SEED

def generate(n=6000, seed=SEED):
    r = np.random.default_rng(seed)
    age = np.clip(r.normal(40, 14, n), 18, 85).round().astype(int)
    gender = r.choice(["Female", "Male"], n, p=[0.48, 0.52])
    exp = np.clip(age - 18 - r.integers(0, 6, n), 0, None)
    veh_age = np.clip(r.gamma(2.2, 3, n), 0, 25).round().astype(int)
    vtype = np.array([r.choice(["Hatchback", "Sedan", "SUV", "Sports", "Van"],
                       p=[.27, .3, .22, .09, .12] if g == "Female" else [.2, .27, .22, .17, .14]) for g in gender])
    mileage = np.clip(r.normal(11000, 4500, n) + (gender == "Male") * 1200, 800, 55000).round(-2)
    prev_claims = np.minimum(r.poisson(np.where(age < 26, .6, .35)), 6)
    prev_acc = np.minimum(prev_claims + r.poisson(.25, n), 8)
    yrs_ins = np.minimum(np.clip(exp - r.integers(0, 4, n), 0, None), 40)
    value = np.clip(r.lognormal(9.6, .55, n) * np.where(vtype == "Sports", 1.8, 1) *
                    np.exp(-.07 * veh_age), 800, 140000).round(-2)
    policy = r.choice(["Third-Party", "Comprehensive"], n, p=[.35, .65])
    region = r.choice(["Low", "Medium", "High"], n, p=[.35, .4, .25])

    z = (-3.1 + 0.55 * prev_claims + 0.30 * prev_acc + 0.000028 * mileage
         + 0.9 * (age < 25) + 0.25 * (age > 70) - 0.025 * exp - 0.04 * yrs_ins
         + 0.5 * (vtype == "Sports") + 0.12 * (vtype == "SUV") + 0.015 * veh_age
         + 0.15 * (policy == "Comprehensive")
         + np.select([region == "Medium", region == "High"], [.25, .6], 0) + r.normal(0, .35, n))
    y = (r.random(n) < 1 / (1 + np.exp(-z))).astype(int)

    df = pd.DataFrame(dict(customer_age=age, vehicle_age=veh_age, vehicle_type=vtype, annual_mileage=mileage,
                           driving_experience=exp, previous_claims=prev_claims, previous_accidents=prev_acc,
                           years_insured=yrs_ins, vehicle_value=value, policy_type=policy, region_risk=region,
                           gender=gender, claim_next_12m=y))
    # realistic messiness: ~3% missing values in a few columns + a few duplicate rows
    for c in ["annual_mileage", "vehicle_value", "region_risk", "vehicle_type"]:
        df.loc[r.random(n) < .03, c] = np.nan
    df = pd.concat([df, df.sample(25, random_state=seed)], ignore_index=True)
    return df

if __name__ == "__main__":
    DATA_DIR.mkdir(exist_ok=True)
    d = generate(); d.to_csv(DATA_DIR / "insurance_dataset.csv", index=False)
    print(d.shape, "claim rate:", round(d.claim_next_12m.mean(), 3))
