import pandas as pd 
import numpy as np
import logging

logger = logging.getLogger(__name__)

def compute_vol_history_features(df, config_features=None):
    """
    Take the raw volatility_history DataFrame (from DoltHub) and compute
    all derived features

    Input columns (from DoltHub schema):
    - date, act_symbol,
    - hv_current, hv_week_ago, hv_month_ago, hv_year_high, hv_year_high_date, hv_year_low, hv_year_low_date,
    - iv_current, iv_week_ago, iv_month_ago, iv_year_high, iv_year_high_date, iv_year_low, iv_year_low_date

    Output: same DataFrame with additional columns for each enabled feature

    Args: 
    - df: raw volatility_history DataFrame with dtypes already castx
    - config_features: teh vol_history.features section from data.yaml
        (if None, compute all features)
    
    Returns:
    - DataFrame keyed on (date, act_symbol) with ~11 new feature columns
    """

    df = df.copy()

    if config_features is None:
        config_features = {k: True for k in [
            "iv_rank", "hv_rank", "vol_risk_premium", "iv_momentum_short",
            "iv_momentum_medium","hv_momentum_short", "hv_momentum_medium",
            "days_since_iv_year_high", "days_since_iv_year_low",
            "days_since_hv_year_high", "days_since_hv_year_low"
        ]}

        # IV Rank
        # (iv_current - iv_year_low) / (iv_year_high - iv_year_low)
        # bounded [0,1] NaN when year_high  == year_low
        if config_features.get("iv_rank"):
            iv_range = df["iv_year_high"] - df["iv_year_low"]
            df["iv_rank"] = np.where(iv_range>0, (df['iv_current'] - df['iv_year_low'])/iv_range, np.nan)
            df['iv_rank'] = df['iv_rank'].clip(0.0,1.0)
            logger.info("Computed iv_rank")

        # HV Rank
        # (hv_current - hv_year_low) / (hv_year_high - hv_year_low)
        # bounded [0,1] NaN when year_high  == year_low
        if config_features.get("hv_rank"):
            hv_range = df["hv_year_high"] - df["hv_year_low"]
            df["hv_rank"] = np.where(hv_range>0, (df['hv_current'] - df['hv_year_low'])/hv_range, np.nan)
            df['hv_rank'] = df['hv_rank'].clip(0.0,1.0)
            logger.info("Computed hv_rank")
        
        # Volatility Risk Premium
        # positive = options are "expensive" relative to realized vol
        # negative = options are "cheap" relative to realized vol
        if config_features.get("vol_risk_premium"):
            df["vol_risk_premium"] = df["iv_current"] - df["hv_current"]
            logger.info("Computed vol_risk_premium")

        # IV momentum
        if config_features.get("iv_momentum_short"):
            df["iv_momentum_short"] = df["iv_current"] - df["iv_week_ago"]
            logger.info("Computed iv_momentum_short")
                
        if config_features.get("iv_momentum_medium"):
            df["iv_momentum_medium"] = df["iv_current"] - df["iv_month_ago"]
            logger.info("Computed iv_momentum_medium")

        # HV momentum
        if config_features.get("hv_momentum_short"):
            df["hv_momentum_short"] = df["hv_current"] - df["hv_week_ago"]
            logger.info("Computed hv_momentum_short")

        if config_features.get("hv_momentum_medium"):
            df["hv_momentum_medium"] = df["hv_current"] - df["hv_month_ago"]
            logger.info("Computed hv_momentum_medium")

        # days since extremes
        # convert date column to pandas timestamp for subtraction
        if any(config_feature.get(k) for k in ["days_since_iv_year_high", "days_since_iv_year_low", "days_since_hv_year_high", "days_since_hv_year_low"]):
            date_ts = pd.to_datetime(pd.Series(df["date"].values))

        if config_features.get("days_since_iv_year_high"):
            iv_high_ts = pd.to_datetime(pd.Series(df["iv_year_high"].values))
            df["days_since_iv_year_high"] = (date_ts - iv_high_ts).dt.days
            df['days_since_iv_year_high'] = df['days_since_iv_year_high'].clip(lower=0)
            logger.info("Computed days_since_iv_year_high")

        if config_features.get("days_since_iv_year_low"):
            iv_low_ts = pd.to_datetime(pd.Series(df["iv_year_low"].values))
            df["days_since_iv_year_low"] = (date_ts - iv_low_ts).dt.days
            df['days_since_iv_year_low'] = df['days_since_iv_year_low'].clip(lower=0)
            logger.info("Computed days_since_iv_year_low")

        if config_features.get("days_since_hv_year_high"):
            hv_high_ts = pd.to_datetime(pd.Series(df["hv_year_high"].values))
            df["days_since_hv_year_high"] = (date_ts - hv_high_ts).dt.days
            df['days_since_hv_year_high'] = df['days_since_hv_year_high'].clip(lower=0)
            logger.info("Computed days_since_hv_year_high")

        if config_features.get("days_since_hv_year_low"):
            hv_low_ts = pd.to_datetime(pd.Series(df["hv_year_low"].values))
            df["days_since_hv_year_low"] = (date_ts - hv_low_ts).dt.days
            df['days_since_hv_year_low'] = df['days_since_hv_year_low'].clip(lower=0)
            logger.info("Computed days_since_hv_year_low")

        feature_cols = [c for c in df.columns if c not in [
            "date", "act_symbol",
            "hv_current", "hv_week_ago", "hv_month_ago", "hv_year_high", "hv_year_high_date",
            "hv_year_low", "hv_year_low_date",
            "iv_current", "iv_week_ago", "iv_month_ago", "iv_year_high", "iv_year_high_date",
            "iv_year_low", "iv_year_low_date",
        ]]

        logger.info("Vol history:  computed %d features: %s", len(feature_cols), feature_cols)
        return df
        
            

    