"""P4 MVP: usable-stock dynamic pool, causal OHLCV factors and ridge baseline.

RESEARCH ONLY. A pool selected from T-day and earlier bars is not a guarantee
of tradability or complete historical ST coverage. Never backfill current
stock_info.st into past selection; never build labels across suspension gaps.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import date, timedelta
from pathlib import Path
import json

import numpy as np
import pandas as pd
from sqlalchemy import text

from .daily import AShareDailyLoader
from .risk_events import _prepare_events

FEATURES = (
    "ret_1", "ret_5", "ret_10", "close_ma20_dev",
    "volume_ratio_20", "amount_ratio_20",
    "range_intraday", "return_intraday",
    "volatility_20", "log_liquidity_20",
)
RESEARCH_WARNING = (
    "Exploratory prediction only: no complete historical ST/suspension/corporate "
    "actions, adjusted prices, auction fills or audited profit. Prices are "
    "unadjusted and the five-session future open-to-open return is a research "
    "label, NOT executable PnL. Do not treat direction accuracy as profitable trading."
)


@dataclass(frozen=True)
class PoolConfig:
    top_n: int = 800
    min_price: float = 3.0
    min_amount: float = 0.0
    consecutive_sessions: int = 40
    hold_sessions: int = 5
    max_abs_label: float = 0.8

    def __post_init__(self):
        if (self.top_n < 1 or self.consecutive_sessions < 22
                or self.hold_sessions < 1 or self.min_price <= 0
                or self.min_amount < 0 or self.max_abs_label <= 0):
            raise ValueError("Invalid P4 pool configuration")


@dataclass
class PoolResult:
    data: pd.DataFrame
    calendar: pd.DatetimeIndex
    stats: dict


@dataclass
class RidgeModel:
    coefficients: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    intercept: float
    alpha: float
    features: tuple[str, ...] = FEATURES

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        if not set(self.features).issubset(frame):
            raise ValueError("Missing model features")
        x = frame.loc[:, list(self.features)].to_numpy(dtype=float)
        if not np.isfinite(x).all():
            raise ValueError("Model cannot predict non-finite features")
        return np.clip((x - self.mean) / self.std, -5.0, 5.0) @ self.coefficients + self.intercept

    def save(self, path: str | Path) -> None:
        np.savez(
            path, coefficients=self.coefficients, mean=self.mean, std=self.std,
            intercept=np.array(self.intercept), alpha=np.array(self.alpha),
            features=np.asarray(self.features),
        )

    @classmethod
    def load(cls, path: str | Path) -> "RidgeModel":
        with np.load(path, allow_pickle=False) as data:
            feats = tuple(str(x) for x in data["features"])
            if feats != FEATURES:
                raise ValueError("Model feature version incompatible with P4 pipeline")
            return cls(data["coefficients"], data["mean"], data["std"],
                       float(data["intercept"]), float(data["alpha"]), feats)


def load_formal_ohlcv(
    start: date, end: date, *, hold_sessions: int = 5, chunksize: int = 50000,
    engine=None, progress=None, include_label_tail: bool = True,
) -> pd.DataFrame:
    """DB SELECT only, including past warmup and future LABEL-ONLY days."""
    if end < start or chunksize < 1 or hold_sessions < 1:
        raise ValueError("Invalid load dates/chunksize/horizon")
    db = engine if engine is not None else AShareDailyLoader().engine
    sql = text("""
        SELECT stock_code, trade_date, open, high, low, close, volume,
               amount, not_billing
          FROM public.market
         WHERE not_billing = 1
           AND trade_date BETWEEN :begin AND :finish
           AND stock_code IS NOT NULL
    """)
    params = {
        "begin": start - timedelta(days=180),
        "finish": (end + timedelta(days=max(20, hold_sessions * 3 + 10))
                   if include_label_tail else end),
    }
    batches = []
    total = 0
    with db.connect() as conn:
        streaming = conn.execution_options(stream_results=True)
        for batch in pd.read_sql_query(sql, streaming, params=params,
                                       chunksize=chunksize):
            batches.append(batch)
            total += len(batch)
            if progress:
                progress(f"已读取正式日线 {total:,} 行")
    return pd.concat(batches, ignore_index=True) if batches else pd.DataFrame()


def _known_event_veto(
    universe: pd.DataFrame, events: pd.DataFrame, calendar: pd.DatetimeIndex,
) -> pd.Series:
    """T-close evidence for T+1 entry: published by T, active by order day.

    If T is the final observed session (prediction), next-session calendar
    cannot be known from future market bars. Conservatively block events
    effective up to T+7 calendar days if already announced by T.
    No future announcement is allowed to affect earlier decisions.
    """
    ev = _prepare_events(events)
    blocked = pd.Series(False, index=universe.index, dtype=bool)
    next_day = dict(zip(calendar[:-1], calendar[1:]))
    for code, group in ev.groupby("stock_code", sort=False):
        indices = universe.index[universe.stock_code.eq(code)]
        if len(indices) == 0:
            continue
        for idx in indices:
            asof = universe.at[idx, "trade_date"]
            order_day = next_day.get(asof, asof + pd.Timedelta(days=7))
            visible = group.loc[
                (group.published_on <= asof)
                & (group.effective_date <= order_day)
            ]
            if visible.empty:
                continue
            r = visible.loc[visible.scope == "RISK"]
            h = visible.loc[visible.scope == "HALT"]
            d = visible.loc[visible.scope == "DELISTED"]
            is_risk = (
                not r.empty
                and r.iloc[-1].event_type in ("RISK_ST", "RISK_STAR_ST")
            )
            is_halt = not h.empty and h.iloc[-1].event_type == "HALT"
            if is_risk or is_halt or not d.empty:
                blocked.at[idx] = True
    return blocked


def build_pool(raw: pd.DataFrame, start: date, end: date,
               cfg: PoolConfig, *, events: pd.DataFrame | None = None) -> PoolResult:
    """Compute all factors using <=T only and labels from exact future sessions.

    The pool is selected at each T on trailing history + trailing liquidity.
    Uses no current stock_info; unknown historical ST is allowed for MVP
    research (and remains explicitly unverified).
    """
    if raw.empty:
        raise ValueError("No OHLCV rows")
    need = {"stock_code", "trade_date", "open", "high", "low", "close",
            "volume", "amount", "not_billing"}
    if need - set(raw.columns):
        raise ValueError(f"Missing formal OHLCV columns: {sorted(need-set(raw.columns))}")
    df = raw.loc[pd.to_numeric(raw.not_billing, errors="coerce").eq(1)].copy()
    if df.empty:
        raise ValueError("No formal (not_billing=1) observations")
    df["trade_date"] = pd.to_datetime(df.trade_date, errors="raise").dt.normalize()
    df["stock_code"] = df.stock_code.astype("string").str.strip()
    if df.stock_code.isna().any() or not df.stock_code.str.fullmatch(r"\d{6}").all():
        raise ValueError("stock_code must be six digits")
    if df.duplicated(["stock_code", "trade_date"]).any():
        raise ValueError("Duplicate market stock/date; cannot silently choose a bar")
    for col in ("open", "high", "low", "close", "volume", "amount"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.sort_values(["stock_code", "trade_date"], kind="stable").reset_index(drop=True)
    calendar = pd.DatetimeIndex(sorted(df.trade_date.unique()))
    if len(calendar) < cfg.consecutive_sessions:
        raise ValueError("Insufficient market sessions")
    ordinal = pd.Series(range(len(calendar)), index=calendar)
    df["session_id"] = df.trade_date.map(ordinal).astype(int)
    good = (
        (df.open > 0) & (df.close > 0) & (df.low > 0)
        & (df.high >= df.low) & (df.volume > 0) & (df.amount > 0)
        & (df.high + 1e-8 >= df[["open", "close"]].max(axis=1))
        & (df.low <= df[["open", "close"]].min(axis=1) + 1e-8)
    ).fillna(False)
    df["bar_valid"] = good
    g = df.groupby("stock_code", sort=False)
    close = df.close
    prev = g.close.shift(1)
    df["ret_1"] = close.div(prev) - 1
    df["ret_5"] = close.div(g.close.shift(5)) - 1
    df["ret_10"] = close.div(g.close.shift(10)) - 1
    ma20 = g.close.transform(lambda s: s.rolling(20, min_periods=20).mean())
    amount20 = g.amount.transform(lambda s: s.rolling(20, min_periods=20).median())
    amount_mean20 = g.amount.transform(lambda s: s.rolling(20, min_periods=20).mean())
    vol20 = g.volume.transform(lambda s: s.rolling(20, min_periods=20).mean())
    df["close_ma20_dev"] = close.div(ma20) - 1
    df["volume_ratio_20"] = df.volume.div(vol20)
    df["amount_ratio_20"] = df.amount.div(amount_mean20)
    df["range_intraday"] = df.high.div(df.low) - 1
    df["return_intraday"] = close.div(df.open) - 1
    df["volatility_20"] = df.groupby("stock_code")["ret_1"].transform(
        lambda s: s.rolling(20, min_periods=20).std()
    )
    df["log_liquidity_20"] = np.log1p(amount20.clip(lower=0))
    df["liquidity_20"] = amount20

    steps = cfg.consecutive_sessions
    consecutive = (
        df.session_id - g.session_id.shift(steps - 1)
    ).eq(steps - 1)
    valid_streak = g.bar_valid.transform(
        lambda s: s.rolling(steps, min_periods=steps).sum()
    ).eq(steps)
    df["valid_past"] = consecutive & valid_streak

    # The label requires T+1 and T+(H+1) actual observed rows AND a
    # continuous session sequence. Never jump across suspended/missing bars.
    hold = cfg.hold_sessions
    next_open = g.open.shift(-1)
    future_open = g.open.shift(-(hold + 1))
    next_sid = g.session_id.shift(-1)
    future_sid = g.session_id.shift(-(hold + 1))
    label = future_open.div(next_open) - 1
    label_good = (
        next_sid.eq(df.session_id + 1)
        & future_sid.eq(df.session_id + hold + 1)
        & next_open.gt(0) & future_open.gt(0)
        & label.abs().le(cfg.max_abs_label)
    )
    df["target_return"] = label.where(label_good)
    df["label_end_session"] = future_sid.where(label_good)

    # Top-N is applied AFTER all causal quality/liquidity filters.
    for feat in FEATURES:
        df[feat] = df[feat].replace([np.inf, -np.inf], np.nan)
    mask = (
        df.trade_date.between(pd.Timestamp(start), pd.Timestamp(end))
        & df.valid_past & df.close.ge(cfg.min_price)
        & df.liquidity_20.ge(cfg.min_amount)
        & pd.Series(np.isfinite(df[list(FEATURES)].to_numpy(dtype=float)).all(axis=1), index=df.index)
    )
    qualified = df.loc[mask].copy()
    candidate_rows = len(qualified)
    veto_count = 0
    if events is not None and not qualified.empty:
        veto = _known_event_veto(qualified, events, calendar)
        veto_count = int(veto.sum())
        qualified = qualified.loc[~veto].copy()

    qualified = qualified.sort_values(
        ["trade_date", "liquidity_20", "stock_code"],
        ascending=[True, False, True], kind="stable"
    )
    qualified["liquidity_rank"] = qualified.groupby("trade_date").cumcount() + 1
    qualified = qualified.loc[qualified.liquidity_rank <= cfg.top_n].copy()
    cols = ["stock_code", "trade_date", "session_id", "liquidity_rank",
            "close", "liquidity_20", "target_return", "label_end_session", *FEATURES]
    selected = qualified[cols].reset_index(drop=True)
    stats = {
        "pool_mode": "DYNAMIC_CAUSAL_LIQUIDITY_TOP_N",
        "event_guard": "PARTIAL_KNOWN_EVENTS_ONLY" if events is not None else "NOT_APPLIED",
        "historical_st_complete": False,
        "current_st_backfilled": False,
        "selection_is_future_data_free": True,
        "raw_formal_bar_rows": int(len(df)),
        "eligible_before_event_veto": int(candidate_rows),
        "blocked_known_risk_event_rows": veto_count,
        "selected_rows": int(len(selected)),
        "selected_stock_codes": int(selected.stock_code.nunique()),
        "selected_days": int(selected.trade_date.nunique()),
        "rows_with_valid_label": int(selected.target_return.notna().sum()),
        "pool_cap": cfg.top_n,
        "pool_config": asdict(cfg),
        "warning": RESEARCH_WARNING,
    }
    return PoolResult(selected, calendar, stats)


def fit_ridge(frame: pd.DataFrame, alpha: float) -> RidgeModel:
    if alpha <= 0 or len(frame) < len(FEATURES) + 2:
        raise ValueError("Invalid ridge alpha or insufficient training samples")
    x = frame.loc[:, list(FEATURES)].to_numpy(dtype=float)
    y = frame.target_return.to_numpy(dtype=float)
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Ridge fitting received invalid values/labels")
    mu, sd = x.mean(axis=0), x.std(axis=0)
    sd = np.where(sd > 1e-9, sd, 1.0)
    z = np.clip((x - mu) / sd, -5, 5)
    center_y = float(y.mean())
    coef = np.linalg.solve(
        z.T @ z / len(z) + alpha * np.eye(len(FEATURES)),
        z.T @ (y - center_y) / len(z),
    )
    return RidgeModel(coef, mu, sd, center_y, alpha)


def rank_diagnostics(frame: pd.DataFrame, *, top_k: int = 20,
                     roundtrip_cost_bps: float = 25.0) -> dict:
    """Return ranking diagnostics, NOT portfolio backtest or realized PnL."""
    if top_k < 1 or roundtrip_cost_bps < 0:
        raise ValueError("Invalid evaluation settings")
    obs = frame.loc[
        np.isfinite(pd.to_numeric(frame.target_return, errors="coerce"))
        & np.isfinite(pd.to_numeric(frame.predicted_return, errors="coerce"))
    ]
    daily_ic, hit, gross, bench, counts = [], [], [], [], []
    for _, day in obs.groupby("trade_date", sort=True):
        if len(day) < 3:
            continue
        xr = day.predicted_return.rank()
        yr = day.target_return.rank()
        if xr.std() > 0 and yr.std() > 0:
            ic = xr.corr(yr)
            if np.isfinite(ic):
                daily_ic.append(float(ic))
        best = day.sort_values(
            ["predicted_return", "stock_code"], ascending=[False, True],
            kind="stable"
        ).head(top_k)
        hit.append(float((best.target_return > 0).mean()))
        gross.append(float(best.target_return.mean()))
        bench.append(float(day.target_return.mean()))
        counts.append(len(day))
    if not gross:
        return {"evaluated_days": 0, "evaluated_observations": 0,
                "mean_rank_ic": None, "topk_positive_share": None,
                "topk_mean_forward_return": None,
                "universe_mean_forward_return": None,
                "topk_net_return_proxy": None}
    cost = roundtrip_cost_bps / 10000
    return {
        "evaluated_days": len(gross),
        "evaluated_observations": int(sum(counts)),
        "mean_rank_ic": float(np.mean(daily_ic)) if daily_ic else None,
        "topk_positive_share": float(np.mean(hit)),
        "topk_mean_forward_return": float(np.mean(gross)),
        "universe_mean_forward_return": float(np.mean(bench)),
        "topk_excess_vs_universe": float(np.mean(np.asarray(gross)-np.asarray(bench))),
        "topk_net_return_proxy": float(np.mean(gross)-cost),
        "roundtrip_cost_bps_proxy": roundtrip_cost_bps,
        "topk": top_k,
        "warning": "Overlapping forward returns; NOT portfolio PnL, no fills or turnover model.",
    }


def fit_splits(pool: PoolResult, *, train_fraction: float = 0.6,
               val_fraction: float = 0.2, top_k: int = 20,
               roundtrip_cost_bps: float = 25.0):
    """Train->tune on validation->refit train+val->evaluate untouched test."""
    if train_fraction <= 0 or val_fraction <= 0 or train_fraction + val_fraction >= 0.9:
        raise ValueError("Invalid date fractions")
    data = pool.data.copy()
    dates = pd.DatetimeIndex(sorted(data.trade_date.unique()))
    if len(dates) < 80:
        raise ValueError("Need at least 80 eligible research dates")
    train_cut = int(len(dates) * train_fraction)
    test_cut = int(len(dates) * (train_fraction + val_fraction))
    val_start, test_start = dates[train_cut], dates[test_cut]
    sid_map = pd.Series(range(len(pool.calendar)), index=pool.calendar)
    v_sid, t_sid = int(sid_map[val_start]), int(sid_map[test_start])
    labelled = data.loc[data.target_return.notna()].copy()
    train = labelled.loc[
        (labelled.trade_date < val_start)
        & (labelled.label_end_session < v_sid)
    ]
    val = labelled.loc[
        labelled.trade_date.ge(val_start)
        & labelled.trade_date.lt(test_start)
        & labelled.label_end_session.lt(t_sid)
    ]
    refit = labelled.loc[
        labelled.trade_date.lt(test_start)
        & labelled.label_end_session.lt(t_sid)
    ]
    test = labelled.loc[labelled.trade_date.ge(test_start)]
    if (len(train) < 100 or len(val) < 40
            or len(refit) < 200 or len(test) < 40):
        raise ValueError("Insufficient clean labelled samples after chronological purging")

    best_alpha, best_score = None, -float("inf")
    tuning = []
    for alpha in (0.01, 0.1, 1.0):
        model = fit_ridge(train, alpha)
        scored = val.copy()
        scored["predicted_return"] = model.predict(scored)
        diags = rank_diagnostics(
            scored, top_k=top_k, roundtrip_cost_bps=roundtrip_cost_bps
        )
        score = diags["mean_rank_ic"]
        if score is None:
            score = -float(np.mean((scored.predicted_return-scored.target_return)**2))
        tuning.append({"alpha": alpha, "selection_score": float(score),
                       "validation_diagnostics": diags})
        if score > best_score:
            best_alpha, best_score = alpha, score

    model = fit_ridge(refit, float(best_alpha))
    test_pred = test.copy()
    test_pred["predicted_return"] = model.predict(test_pred)
    test_metrics = rank_diagnostics(
        test_pred, top_k=top_k, roundtrip_cost_bps=roundtrip_cost_bps
    )
    # Baseline NEG(ma20_dev) is a ranking comparison, not separate fitted model.
    baseline = test.copy()
    baseline["predicted_return"] = -baseline["close_ma20_dev"]
    baseline_metrics = rank_diagnostics(
        baseline, top_k=top_k, roundtrip_cost_bps=roundtrip_cost_bps
    )

    infer = data.loc[data.trade_date.eq(data.trade_date.max())].copy()
    infer["predicted_return"] = model.predict(infer)
    infer = infer.sort_values(["predicted_return", "stock_code"],
                              ascending=[False, True]).reset_index(drop=True)
    # Persist last historical TRAIN label date to block prediction-time
    # requests preceding the data used to fit model weights.
    latest_label_sid = int(refit.label_end_session.max())
    metadata = {
        "model_version": "P4_RIDGE_V1",
        "feature_names": list(FEATURES),
        "pool_config": pool.stats["pool_config"],
        "train_start_date": str(train.trade_date.min().date()),
        "train_last_signal_date": str(refit.trade_date.max().date()),
        "train_last_label_date": str(pool.calendar[latest_label_sid].date()),
        "validation_start": str(val_start.date()),
        "test_start": str(test_start.date()),
        "test_end": str(dates[-1].date()),
        "purge_rule": "label_end_session < next_split_first_session",
        "selection_alpha": best_alpha,
        "tuning": tuning,
        "refit_samples": len(refit),
        "test_samples": len(test),
        "pool_stats": pool.stats,
        "test_metrics": test_metrics,
        "baseline_metrics": baseline_metrics,
        "model_is_tradable_strategy": False,
        "warning": RESEARCH_WARNING,
    }
    return model, metadata, test_pred, infer
