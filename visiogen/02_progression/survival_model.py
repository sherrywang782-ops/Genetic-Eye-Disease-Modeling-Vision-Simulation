"""Survival analysis baseline models.

Provides a Cox proportional hazards baseline model for disease onset and progression.
"""

from __future__ import annotations

import pandas as pd


def fit_cox_model(df: pd.DataFrame, duration_col: str, event_col: str):
    """Fit a Cox proportional hazards model.

    Parameters
    ----------
    df:
        Dataframe containing survival/time-to-event data.
    duration_col:
        Column name with duration until event.
    event_col:
        Column name indicating event occurrence.
    """
    raise NotImplementedError("Survival modeling is not implemented yet.")
