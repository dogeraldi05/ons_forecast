"""Shared pipeline. The only format-specific piece is sql/00_raw_<fmt>.sql."""
import time
from pathlib import Path

import duckdb
import holidays
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
SQL = ROOT / "sql"

FEATURES_7D = ["cod_subsistema", "lag_7d", "lag_14d", "lag_52w", "media_movel_7d",
               "dia_semana", "mes", "dia_ano", "feriado"]
FEATURES_CAL = ["cod_subsistema", "dia_semana", "mes", "dia_ano", "feriado"]
XGB_PARAMS = dict(n_estimators=600, learning_rate=0.03, max_depth=5,
                  subsample=0.8, colsample_bytree=0.8, random_state=42)

# Walk-forward folds (expanding window). Holdout is touched once, at the end.
FOLDS = [("2025-07-01", "2025-09-30"), ("2025-10-01", "2025-12-31"),
         ("2026-01-01", "2026-03-31"), ("2026-04-01", "2026-06-30"),
         ("2026-07-01", "2026-08-14")]
HOLDOUT = ("2026-08-15", "2026-11-14")


def connect(fmt: str) -> tuple[duckdb.DuckDBPyConnection, float]:
    """Build the warehouse in memory. Returns the connection and ingestion wall time (s)."""
    con = duckdb.connect()
    con.execute(f"SET file_search_path = '{ROOT.as_posix()}'")
    br = holidays.Brazil(years=range(2024, 2028), categories=("public", "optional"))
    con.register("feriados", pd.DataFrame({"data": list(br.keys()), "nome": list(br.values())}))

    t0 = time.perf_counter()
    con.execute((SQL / f"00_raw_{fmt}.sql").read_text())
    con.execute("CREATE OR REPLACE TABLE raw_materialized AS SELECT * FROM raw_demanda_maxima")
    ingest_s = time.perf_counter() - t0
    con.execute("CREATE OR REPLACE VIEW raw_demanda_maxima AS SELECT * FROM raw_materialized")

    for f in ["01_stg_demanda_maxima", "02_dim_calendario", "03_fct_demanda_diaria", "04_fct_demanda_features"]:
        con.execute((SQL / f"{f}.sql").read_text())
    return con, ingest_s


def wape(y, yhat) -> float:
    return float(np.abs(y - yhat).sum() / np.abs(y).sum())


def mape(y, yhat) -> float:
    return float(np.mean(np.abs((y - yhat) / y)))


def scores(df: pd.DataFrame, pred_cols: list[str], naive_col: str = "seasonal_naive") -> pd.DataFrame:
    """WAPE, MAPE and skill vs the naive column, one row per forecast column."""
    y = df["demanda_maxima_mw"]
    out = pd.DataFrame({c: {"WAPE": wape(y, df[c]), "MAPE": mape(y, df[c])} for c in pred_cols}).T
    out["skill_vs_naive"] = 1 - out["WAPE"] / out.loc[naive_col, "WAPE"]
    return out


def add_baselines(df: pd.DataFrame) -> pd.DataFrame:
    # At a 7-day horizon, "previous-day persistence" collapses to lag_7d (same as seasonal naive),
    # so the second Tier-0 reference is the mean of the last fully known week.
    return df.assign(seasonal_naive=df["lag_7d"], last_week_mean=df["media_movel_7d"])


def fit_predict_7d(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    model = XGBRegressor(**XGB_PARAMS).fit(train[FEATURES_7D], train["demanda_maxima_mw"])
    return model.predict(test[FEATURES_7D])


def walk_forward(feat: pd.DataFrame) -> pd.DataFrame:
    """7-day-ahead walk-forward CV. Returns per-fold scores."""
    feat = add_baselines(feat.dropna(subset=["demanda_maxima_mw", "lag_7d", "media_movel_7d"]))
    rows = []
    for i, (start, end) in enumerate(FOLDS, 1):
        train = feat[feat.din_instante < start]
        test = feat[(feat.din_instante >= start) & (feat.din_instante <= end)].copy()
        test["xgboost"] = fit_predict_7d(train, test)
        s = scores(test, ["seasonal_naive", "last_week_mean", "xgboost"])
        rows.append(s.assign(fold=i, test=f"{start} → {end}"))
    return pd.concat(rows).rename_axis("model").reset_index()


def holdout(feat: pd.DataFrame) -> pd.DataFrame:
    """Final 7-day-ahead evaluation on the untouched holdout. Returns predictions."""
    feat = add_baselines(feat.dropna(subset=["demanda_maxima_mw", "lag_7d", "media_movel_7d"]))
    train = feat[feat.din_instante < HOLDOUT[0]]
    test = feat[(feat.din_instante >= HOLDOUT[0]) & (feat.din_instante <= HOLDOUT[1])].copy()
    test["xgboost"] = fit_predict_7d(train, test)
    return test


class AnnualModel:
    """Long-horizon (calendar-only) model: linear trend per subsystem x XGBoost seasonal ratio.

    shortcut: linear trend fitted on <3 years; replace with an exogenous load-growth path (e.g. EPE PDE)
    when forecasting more than one year out.
    """

    def fit(self, hist: pd.DataFrame) -> "AnnualModel":
        hist = hist.dropna(subset=["demanda_maxima_mw"])
        self.trend = {s: np.polyfit(self._t(g), g["demanda_maxima_mw"], 1)
                      for s, g in hist.groupby("id_subsistema")}
        ratio = hist["demanda_maxima_mw"] / self._trend(hist)
        self.model = XGBRegressor(**XGB_PARAMS).fit(hist[FEATURES_CAL], ratio)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.model.predict(df[FEATURES_CAL]) * self._trend(df)

    @staticmethod
    def _t(df):
        return (pd.to_datetime(df["din_instante"]) - pd.Timestamp("2024-01-01")).dt.days.to_numpy()

    def _trend(self, df):
        return np.array([np.polyval(self.trend[s], t) for s, t in zip(df["id_subsistema"], self._t(df))])


def calendar_2027(con) -> pd.DataFrame:
    return con.sql("""
        SELECT c.din_instante, s.id_subsistema, s.nom_subsistema,
               CASE s.id_subsistema WHEN 'SE' THEN 0 WHEN 'S' THEN 1 WHEN 'NE' THEN 2 WHEN 'N' THEN 3 END AS cod_subsistema,
               c.dia_semana, c.mes, c.dia_ano, c.feriado
        FROM dim_calendario c
        CROSS JOIN (SELECT DISTINCT id_subsistema, nom_subsistema FROM fct_demanda_diaria) s
        WHERE c.ano = 2027
        ORDER BY s.id_subsistema, c.din_instante
    """).df()


if __name__ == "__main__":
    # Self-check: both formats must produce the identical feature table.
    a = connect("csv")[0].sql("SELECT * FROM fct_demanda_features ORDER BY 1, 2").df()
    b = connect("parquet")[0].sql("SELECT * FROM fct_demanda_features ORDER BY 1, 2").df()
    pd.testing.assert_frame_equal(a, b)
    assert a.groupby("id_subsistema").size().nunique() == 1, "uneven subsystem coverage"
    assert a["demanda_maxima_mw"].notna().all(), "gaps in history"
    print("ok:", a.shape, a.din_instante.min(), "→", a.din_instante.max())
