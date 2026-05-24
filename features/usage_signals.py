"""
Usage-based feature engineering for enterprise upsell classification.
Computes velocity, adoption momentum, and engagement signals.
"""
import pandas as pd
import numpy as np


def add_usage_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Adoption momentum: are they growing into the product?
    df["adoption_momentum"] = df["feature_adoption_score"] * df["monthly_active_users_pct"]
    # Security exposure score: more alerts = more upsell urgency
    df["security_urgency"] = np.log1p(df["security_event_alerts_30d"])
    # Recency engagement
    df["is_recently_active"] = (df["days_since_last_login"] <= 14).astype(int)
    # License expansion headroom
    df["expansion_headroom"] = df["license_gap_pct"] * df["m365_seats"]
    # Support burden (high tickets = pain points that security solves)
    df["support_burden"] = np.log1p(df["support_tickets_6m"])
    # Renewal proximity (closer to renewal = higher upsell window)
    df["in_renewal_window"] = (df["renewal_days_out"].between(0, 90)).astype(int)
    # Exec engagement as a proxy for deal readiness
    df["exec_engaged"] = (df["exec_contacts"] >= 2).astype(int)
    # Usage velocity signal
    df["positive_velocity"] = (df["usage_velocity_30d"] > 0.01).astype(int)
    return df


def get_feature_cols() -> list[str]:
    return [
        "license_gap_pct", "has_any_security", "feature_adoption_score",
        "monthly_active_users_pct", "usage_velocity_30d", "support_tickets_6m",
        "days_since_last_login", "security_event_alerts_30d", "exec_contacts",
        "marketing_touchpoints_90d", "webinar_attendee", "renewal_days_out",
        "account_age_months", "adoption_momentum", "security_urgency",
        "is_recently_active", "expansion_headroom", "support_burden",
        "in_renewal_window", "exec_engaged", "positive_velocity",
    ]