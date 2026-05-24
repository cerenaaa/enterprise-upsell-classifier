import numpy as np
import pandas as pd

INDUSTRIES = ["Financial Services","Healthcare","Manufacturing","Retail","Technology","Government","Education","Energy"]
REGIONS = ["NA","EMEA","APAC","LATAM"]
SEGMENTS = ["Enterprise","Mid-Market","SMC"]

def generate_accounts(n: int = 10_000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame()
    df["account_id"] = [f"ACC_{i:06d}" for i in range(n)]
    df["industry"] = rng.choice(INDUSTRIES, n)
    df["region"] = rng.choice(REGIONS, n, p=[0.45, 0.30, 0.18, 0.07])
    df["segment"] = rng.choice(SEGMENTS, n, p=[0.25, 0.35, 0.40])
    df["employee_count"] = np.where(
        df["segment"] == "Enterprise", rng.integers(1000, 50000, n),
        np.where(df["segment"] == "Mid-Market", rng.integers(100, 1000, n),
                 rng.integers(10, 100, n)))
    df["m365_seats"] = (df["employee_count"] * rng.uniform(0.6, 1.1, n)).astype(int)
    df["security_seats_current"] = (df["m365_seats"] * rng.uniform(0.0, 0.5, n)).astype(int)
    df["license_gap_pct"] = ((df["m365_seats"] - df["security_seats_current"]) / df["m365_seats"]).clip(0, 1)
    df["has_any_security"] = rng.binomial(1, 0.45, n)
    df["monthly_active_users_pct"] = rng.beta(5, 2, n).round(3)
    df["feature_adoption_score"] = rng.beta(3, 3, n).round(3)
    df["usage_velocity_30d"] = rng.normal(0.02, 0.05, n).round(4)
    df["support_tickets_6m"] = rng.poisson(2, n)
    df["days_since_last_login"] = rng.integers(0, 90, n)
    df["security_event_alerts_30d"] = rng.poisson(3, n)
    df["exec_contacts"] = rng.integers(0, 5, n)
    df["marketing_touchpoints_90d"] = rng.integers(0, 20, n)
    df["webinar_attendee"] = rng.binomial(1, 0.15, n)
    df["renewal_days_out"] = rng.integers(-30, 365, n)
    df["account_age_months"] = rng.integers(1, 120, n)
    logit = (
        -2.5
        + 1.8 * df["license_gap_pct"]
        + 0.8 * df["has_any_security"]
        + 1.2 * df["feature_adoption_score"]
        + 0.5 * df["monthly_active_users_pct"]
        + 0.04 * df["security_event_alerts_30d"]
        - 0.01 * df["days_since_last_login"]
        + 0.3 * (df["segment"] == "Enterprise").astype(float)
        + 0.5 * df["webinar_attendee"]
        + rng.normal(0, 0.4, n)
    )
    prob = 1 / (1 + np.exp(-logit))
    df["upsold"] = (rng.uniform(size=n) < prob).astype(int)
    print(f"Generated {n:,} accounts | Upsell rate: {df['upsold'].mean():.1%}")
    return df

if __name__ == "__main__":
    df = generate_accounts()
    df.to_csv("data/accounts.csv", index=False)