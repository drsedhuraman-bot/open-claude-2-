"""
Scalp strategy — 8-condition scoring engine adapted from INDY BOT.

Conditions (each contributes to bull/bear score):
  C1: VWAP bias          — LTP vs VWAP ± ATR×0.3          (±2 pts)
  C2: StochRSI           — K vs D cross + zone              (±1.5 pts)
  C3: RSI momentum       — RSI(5) level zones               (±1 to ±1.5 pts)
  C4: MACD histogram     — direction                        (±1 pt)
  C5: Bollinger Band     — position vs bands                (±1 pt)
  C6: Candle pattern     — bullish/bearish/doji             (±1.5 pts)
  C7: Volume             — ratio vs rolling avg             (±2 pts)
  C8: Support/Resistance — proximity to pivot levels        (±1 pt)

Verdict:
  bull_score ≥ PASS_THRESHOLD  → LONG  (min 4 conditions must pass)
  bear_score ≥ PASS_THRESHOLD  → SHORT (currently only LONG traded)
  else                         → HOLD
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from .base import BaseStrategy, Signal, StrategyResult

# ── Scoring thresholds ─────────────────────────────────────────────────────────
PASS_THRESHOLD   = 6.0   # bull/bear score needed for a trade signal
MIN_CONDITIONS   = 4     # minimum number of conditions that must fire


@dataclass
class ConditionResult:
    name:       str
    bull_pts:   float
    bear_pts:   float
    label:      str           # short description shown in UI
    detail:     str = ""      # verbose detail


@dataclass
class ScalpResult:
    conditions:   list[ConditionResult]
    bull_score:   float
    bear_score:   float
    bull_pass:    int          # conditions that scored bull pts
    bear_pass:    int
    verdict:      str          # "LONG" | "SHORT" | "HOLD"
    signal:       Signal

    # Position management
    entry:        float = 0.0
    stop:         float = 0.0
    t1:           float = 0.0
    t2:           float = 0.0
    risk_reward:  float = 0.0
    risk_pts:     float = 0.0

    # Market context
    vwap:         float = 0.0
    atr:          float = 0.0
    pivot:        float = 0.0
    r1:           float = 0.0
    s1:           float = 0.0
    r2:           float = 0.0
    s2:           float = 0.0

    # Indicator snapshot
    rsi:          float = 0.0
    stoch_k:      float = 0.0
    stoch_d:      float = 0.0
    macd_hist:    float = 0.0
    bb_pct:       float = 0.0   # 0=lower band, 1=upper band
    volume_ratio: float = 0.0


# ── Preset configurations ──────────────────────────────────────────────────────

PRESETS: dict[str, dict] = {
    "Ultra Fast": {
        "rsi_period": 3, "stoch_period": 5, "stoch_smooth": 2,
        "macd_fast": 5, "macd_slow": 13, "macd_signal": 4,
        "bb_period": 10, "bb_std": 1.5, "atr_period": 5,
        "vol_period": 5, "pass_threshold": 5.5, "min_conditions": 3,
    },
    "Standard": {
        "rsi_period": 5, "stoch_period": 8, "stoch_smooth": 3,
        "macd_fast": 8, "macd_slow": 21, "macd_signal": 5,
        "bb_period": 15, "bb_std": 2.0, "atr_period": 10,
        "vol_period": 10, "pass_threshold": 6.0, "min_conditions": 4,
    },
    "Relaxed": {
        "rsi_period": 7, "stoch_period": 10, "stoch_smooth": 3,
        "macd_fast": 10, "macd_slow": 26, "macd_signal": 7,
        "bb_period": 20, "bb_std": 2.0, "atr_period": 14,
        "vol_period": 14, "pass_threshold": 5.0, "min_conditions": 3,
    },
    "Positional": {
        "rsi_period": 14, "stoch_period": 14, "stoch_smooth": 3,
        "macd_fast": 12, "macd_slow": 26, "macd_signal": 9,
        "bb_period": 20, "bb_std": 2.0, "atr_period": 14,
        "vol_period": 20, "pass_threshold": 6.5, "min_conditions": 5,
    },
    "Tight BB": {
        "rsi_period": 5, "stoch_period": 8, "stoch_smooth": 3,
        "macd_fast": 8, "macd_slow": 21, "macd_signal": 5,
        "bb_period": 10, "bb_std": 1.5, "atr_period": 10,
        "vol_period": 10, "pass_threshold": 6.0, "min_conditions": 4,
    },
    "VWAP Rev": {
        "rsi_period": 5, "stoch_period": 8, "stoch_smooth": 3,
        "macd_fast": 8, "macd_slow": 21, "macd_signal": 5,
        "bb_period": 15, "bb_std": 2.0, "atr_period": 10,
        "vol_period": 10, "pass_threshold": 5.5, "min_conditions": 4,
    },
    "Fast MACD": {
        "rsi_period": 5, "stoch_period": 8, "stoch_smooth": 3,
        "macd_fast": 5, "macd_slow": 13, "macd_signal": 4,
        "bb_period": 15, "bb_std": 2.0, "atr_period": 10,
        "vol_period": 10, "pass_threshold": 6.0, "min_conditions": 4,
    },
    "Explosive": {
        "rsi_period": 3, "stoch_period": 5, "stoch_smooth": 2,
        "macd_fast": 5, "macd_slow": 13, "macd_signal": 4,
        "bb_period": 10, "bb_std": 1.5, "atr_period": 5,
        "vol_period": 5, "pass_threshold": 7.0, "min_conditions": 5,
    },
}

DEFAULT_PRESET = "Standard"


# ── Helper math ────────────────────────────────────────────────────────────────

def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def _rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    avg_g = gain.ewm(com=period - 1, adjust=False).mean()
    avg_l = loss.ewm(com=period - 1, adjust=False).mean()
    rs    = avg_g / avg_l.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _stoch_rsi(close: pd.Series, period: int, smooth: int) -> tuple[pd.Series, pd.Series]:
    rsi   = _rsi(close, period)
    min_r = rsi.rolling(period).min()
    max_r = rsi.rolling(period).max()
    k_raw = (rsi - min_r) / (max_r - min_r + 1e-9) * 100
    k     = k_raw.rolling(smooth).mean()
    d     = k.rolling(smooth).mean()
    return k, d


def _macd(close: pd.Series, fast: int, slow: int, signal: int):
    macd_line = _ema(close, fast) - _ema(close, slow)
    sig_line  = _ema(macd_line, signal)
    hist      = macd_line - sig_line
    return macd_line, sig_line, hist


def _bollinger(close: pd.Series, period: int, std_mult: float):
    mid   = close.rolling(period).mean()
    std   = close.rolling(period).std()
    upper = mid + std_mult * std
    lower = mid - std_mult * std
    return upper, mid, lower


def _atr(df: pd.DataFrame, period: int) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def _vwap(df: pd.DataFrame) -> pd.Series:
    """Session VWAP — resets every day."""
    typical = (df["high"] + df["low"] + df["close"]) / 3
    vp      = typical * df["volume"]
    # group by date (uses open_time if available, else index)
    if "open_time" in df.columns:
        date_key = df["open_time"].dt.date
    else:
        date_key = pd.Series(df.index, index=df.index).apply(lambda x: x.date() if hasattr(x, "date") else 0)

    cum_vp  = vp.groupby(date_key).cumsum()
    cum_vol = df["volume"].groupby(date_key).cumsum()
    return cum_vp / cum_vol.replace(0, np.nan)


def _pivot_levels(df: pd.DataFrame) -> dict[str, float]:
    """Classic daily pivot from previous day's H/L/C."""
    prev_high  = float(df["high"].iloc[-2]) if len(df) > 1 else float(df["high"].iloc[-1])
    prev_low   = float(df["low"].iloc[-2])  if len(df) > 1 else float(df["low"].iloc[-1])
    prev_close = float(df["close"].iloc[-2]) if len(df) > 1 else float(df["close"].iloc[-1])
    pivot = (prev_high + prev_low + prev_close) / 3
    r1    = 2 * pivot - prev_low
    s1    = 2 * pivot - prev_high
    r2    = pivot + (prev_high - prev_low)
    s2    = pivot - (prev_high - prev_low)
    return {"pivot": pivot, "r1": r1, "s1": s1, "r2": r2, "s2": s2}


# ── 8-condition scoring engine ─────────────────────────────────────────────────

class ScalpEngine:
    """
    Stateless scoring engine — call analyze(df) to get ScalpResult.
    """

    def __init__(
        self,
        rsi_period:      int   = 5,
        stoch_period:    int   = 8,
        stoch_smooth:    int   = 3,
        macd_fast:       int   = 8,
        macd_slow:       int   = 21,
        macd_signal:     int   = 5,
        bb_period:       int   = 15,
        bb_std:          float = 2.0,
        atr_period:      int   = 10,
        vol_period:      int   = 10,
        pass_threshold:  float = PASS_THRESHOLD,
        min_conditions:  int   = MIN_CONDITIONS,
        sl_atr_mult:     float = 1.0,
        t1_atr_mult:     float = 1.5,
        t2_atr_mult:     float = 3.0,
    ) -> None:
        self.rsi_period     = rsi_period
        self.stoch_period   = stoch_period
        self.stoch_smooth   = stoch_smooth
        self.macd_fast      = macd_fast
        self.macd_slow      = macd_slow
        self.macd_signal    = macd_signal
        self.bb_period      = bb_period
        self.bb_std         = bb_std
        self.atr_period     = atr_period
        self.vol_period     = vol_period
        self.pass_threshold = pass_threshold
        self.min_conditions = min_conditions
        self.sl_atr_mult    = sl_atr_mult
        self.t1_atr_mult    = t1_atr_mult
        self.t2_atr_mult    = t2_atr_mult

    @classmethod
    def from_preset(cls, preset_name: str, **overrides) -> "ScalpEngine":
        params = PRESETS.get(preset_name, PRESETS[DEFAULT_PRESET]).copy()
        params.update(overrides)
        return cls(**params)

    def analyze(self, df: pd.DataFrame) -> ScalpResult:
        min_rows = max(self.macd_slow * 2, self.bb_period * 2, 60)
        if len(df) < min_rows:
            raise ValueError(f"Need at least {min_rows} candles, got {len(df)}")

        close  = df["close"].astype(float)
        high   = df["high"].astype(float)
        low    = df["low"].astype(float)
        volume = df["volume"].astype(float)

        # ── Pre-compute indicators ─────────────────────────────────────────
        rsi_val  = float(_rsi(close, self.rsi_period).iloc[-1])
        k, d     = _stoch_rsi(close, self.stoch_period, self.stoch_smooth)
        k_val    = float(k.iloc[-1])
        d_val    = float(d.iloc[-1])
        k_prev   = float(k.iloc[-2])
        d_prev   = float(d.iloc[-2])

        ml, sl_line, hist = _macd(close, self.macd_fast, self.macd_slow, self.macd_signal)
        hist_val  = float(hist.iloc[-1])
        hist_prev = float(hist.iloc[-2])

        bb_upper, bb_mid, bb_lower = _bollinger(close, self.bb_period, self.bb_std)
        bb_u = float(bb_upper.iloc[-1])
        bb_l = float(bb_lower.iloc[-1])
        bb_m = float(bb_mid.iloc[-1])
        ltp  = float(close.iloc[-1])
        bb_pct = (ltp - bb_l) / (bb_u - bb_l + 1e-9)

        atr_series = _atr(df, self.atr_period)
        atr_val    = float(atr_series.iloc[-1])

        try:
            vwap_series = _vwap(df)
            vwap_val    = float(vwap_series.iloc[-1])
        except Exception:
            vwap_val = float(close.rolling(20).mean().iloc[-1])

        vol_avg    = float(volume.rolling(self.vol_period).mean().iloc[-1])
        vol_now    = float(volume.iloc[-1])
        vol_ratio  = vol_now / (vol_avg + 1e-9)

        levels     = _pivot_levels(df)
        pivot_val  = levels["pivot"]
        r1_val     = levels["r1"]
        s1_val     = levels["s1"]
        r2_val     = levels["r2"]
        s2_val     = levels["s2"]

        # Candle body / wick analysis
        open_val   = float(df["open"].astype(float).iloc[-1])
        body       = ltp - open_val
        body_abs   = abs(body)
        candle_range = (float(high.iloc[-1]) - float(low.iloc[-1])) or 1.0
        body_ratio   = body_abs / candle_range

        # ── Condition scoring ──────────────────────────────────────────────
        conditions: list[ConditionResult] = []

        # C1 — VWAP bias (±2 pts)
        vwap_band = atr_val * 0.3
        if ltp > vwap_val + vwap_band:
            c1 = ConditionResult("C1:VWAP", bull_pts=2.0, bear_pts=0.0,
                                 label="Above VWAP", detail=f"LTP {ltp:.2f} > VWAP+band {vwap_val+vwap_band:.2f}")
        elif ltp < vwap_val - vwap_band:
            c1 = ConditionResult("C1:VWAP", bull_pts=0.0, bear_pts=2.0,
                                 label="Below VWAP", detail=f"LTP {ltp:.2f} < VWAP-band {vwap_val-vwap_band:.2f}")
        else:
            c1 = ConditionResult("C1:VWAP", bull_pts=0.0, bear_pts=0.0,
                                 label="At VWAP", detail=f"LTP {ltp:.2f} near VWAP {vwap_val:.2f}")
        conditions.append(c1)

        # C2 — StochRSI cross (±1.5 pts)
        bull_cross = (k_prev <= d_prev) and (k_val > d_val)
        bear_cross = (k_prev >= d_prev) and (k_val < d_val)
        if bull_cross and k_val < 80:
            c2 = ConditionResult("C2:StochRSI", bull_pts=1.5, bear_pts=0.0,
                                 label="Stoch Bull Cross", detail=f"K={k_val:.1f} crossed above D={d_val:.1f}")
        elif bear_cross and k_val > 20:
            c2 = ConditionResult("C2:StochRSI", bull_pts=0.0, bear_pts=1.5,
                                 label="Stoch Bear Cross", detail=f"K={k_val:.1f} crossed below D={d_val:.1f}")
        elif k_val < 20:
            c2 = ConditionResult("C2:StochRSI", bull_pts=1.0, bear_pts=0.0,
                                 label="Stoch Oversold", detail=f"K={k_val:.1f} in oversold zone")
        elif k_val > 80:
            c2 = ConditionResult("C2:StochRSI", bull_pts=0.0, bear_pts=1.0,
                                 label="Stoch Overbought", detail=f"K={k_val:.1f} in overbought zone")
        else:
            c2 = ConditionResult("C2:StochRSI", bull_pts=0.0, bear_pts=0.0,
                                 label="Stoch Neutral", detail=f"K={k_val:.1f} D={d_val:.1f}")
        conditions.append(c2)

        # C3 — RSI momentum (±1 to ±1.5 pts)
        if rsi_val < 30:
            c3 = ConditionResult("C3:RSI", bull_pts=1.5, bear_pts=0.0,
                                 label="RSI Oversold", detail=f"RSI={rsi_val:.1f} < 30")
        elif rsi_val > 70:
            c3 = ConditionResult("C3:RSI", bull_pts=0.0, bear_pts=1.5,
                                 label="RSI Overbought", detail=f"RSI={rsi_val:.1f} > 70")
        elif 40 <= rsi_val <= 60:
            c3 = ConditionResult("C3:RSI", bull_pts=0.0, bear_pts=0.0,
                                 label="RSI Neutral", detail=f"RSI={rsi_val:.1f}")
        elif rsi_val > 55:
            c3 = ConditionResult("C3:RSI", bull_pts=1.0, bear_pts=0.0,
                                 label="RSI Bullish", detail=f"RSI={rsi_val:.1f} in bull zone")
        else:
            c3 = ConditionResult("C3:RSI", bull_pts=0.0, bear_pts=1.0,
                                 label="RSI Bearish", detail=f"RSI={rsi_val:.1f} in bear zone")
        conditions.append(c3)

        # C4 — MACD histogram direction (±1 pt)
        if hist_val > 0 and hist_val > hist_prev:
            c4 = ConditionResult("C4:MACD", bull_pts=1.0, bear_pts=0.0,
                                 label="MACD Expanding Bull", detail=f"hist={hist_val:.4f} rising above 0")
        elif hist_val < 0 and hist_val < hist_prev:
            c4 = ConditionResult("C4:MACD", bull_pts=0.0, bear_pts=1.0,
                                 label="MACD Expanding Bear", detail=f"hist={hist_val:.4f} falling below 0")
        elif hist_val > 0:
            c4 = ConditionResult("C4:MACD", bull_pts=0.5, bear_pts=0.0,
                                 label="MACD Positive", detail=f"hist={hist_val:.4f} > 0")
        elif hist_val < 0:
            c4 = ConditionResult("C4:MACD", bull_pts=0.0, bear_pts=0.5,
                                 label="MACD Negative", detail=f"hist={hist_val:.4f} < 0")
        else:
            c4 = ConditionResult("C4:MACD", bull_pts=0.0, bear_pts=0.0,
                                 label="MACD Zero", detail="histogram at zero")
        conditions.append(c4)

        # C5 — Bollinger Band position (±1 pt)
        if bb_pct < 0.2:
            c5 = ConditionResult("C5:BB", bull_pts=1.0, bear_pts=0.0,
                                 label="Near Lower BB", detail=f"BB%={bb_pct:.2f}, near lower band")
        elif bb_pct > 0.8:
            c5 = ConditionResult("C5:BB", bull_pts=0.0, bear_pts=1.0,
                                 label="Near Upper BB", detail=f"BB%={bb_pct:.2f}, near upper band")
        elif bb_pct > 0.5:
            c5 = ConditionResult("C5:BB", bull_pts=0.5, bear_pts=0.0,
                                 label="Upper Half BB", detail=f"BB%={bb_pct:.2f}")
        else:
            c5 = ConditionResult("C5:BB", bull_pts=0.0, bear_pts=0.5,
                                 label="Lower Half BB", detail=f"BB%={bb_pct:.2f}")
        conditions.append(c5)

        # C6 — Candle pattern (±1.5 pts)
        if body > 0 and body_ratio > 0.6:
            c6 = ConditionResult("C6:Candle", bull_pts=1.5, bear_pts=0.0,
                                 label="Strong Bull Candle", detail=f"body_ratio={body_ratio:.2f}")
        elif body < 0 and body_ratio > 0.6:
            c6 = ConditionResult("C6:Candle", bull_pts=0.0, bear_pts=1.5,
                                 label="Strong Bear Candle", detail=f"body_ratio={body_ratio:.2f}")
        elif body_ratio < 0.2:
            # Doji — indecision, slight bias toward continuation
            c6 = ConditionResult("C6:Candle", bull_pts=0.0, bear_pts=0.0,
                                 label="Doji", detail=f"body_ratio={body_ratio:.2f} (indecision)")
        elif body > 0:
            c6 = ConditionResult("C6:Candle", bull_pts=0.75, bear_pts=0.0,
                                 label="Bull Candle", detail=f"body_ratio={body_ratio:.2f}")
        else:
            c6 = ConditionResult("C6:Candle", bull_pts=0.0, bear_pts=0.75,
                                 label="Bear Candle", detail=f"body_ratio={body_ratio:.2f}")
        conditions.append(c6)

        # C7 — Volume (±2 pts, adapted from breadth for crypto)
        if vol_ratio > 1.5:
            if body > 0:
                c7 = ConditionResult("C7:Volume", bull_pts=2.0, bear_pts=0.0,
                                     label="High Vol Bull", detail=f"vol_ratio={vol_ratio:.2f}x on up-move")
            else:
                c7 = ConditionResult("C7:Volume", bull_pts=0.0, bear_pts=2.0,
                                     label="High Vol Bear", detail=f"vol_ratio={vol_ratio:.2f}x on down-move")
        elif vol_ratio > 1.0:
            if body > 0:
                c7 = ConditionResult("C7:Volume", bull_pts=1.0, bear_pts=0.0,
                                     label="Above Avg Vol Bull", detail=f"vol_ratio={vol_ratio:.2f}x")
            else:
                c7 = ConditionResult("C7:Volume", bull_pts=0.0, bear_pts=1.0,
                                     label="Above Avg Vol Bear", detail=f"vol_ratio={vol_ratio:.2f}x")
        else:
            c7 = ConditionResult("C7:Volume", bull_pts=0.0, bear_pts=0.0,
                                 label="Low Volume", detail=f"vol_ratio={vol_ratio:.2f}x (below avg)")
        conditions.append(c7)

        # C8 — S/R proximity (±1 pt)
        s_r_threshold = atr_val * 0.5
        near_support  = abs(ltp - s1_val) < s_r_threshold or abs(ltp - s2_val) < s_r_threshold
        near_resist   = abs(ltp - r1_val) < s_r_threshold or abs(ltp - r2_val) < s_r_threshold
        near_pivot    = abs(ltp - pivot_val) < s_r_threshold

        if near_support and not near_resist:
            c8 = ConditionResult("C8:S/R", bull_pts=1.0, bear_pts=0.0,
                                 label="Near Support", detail=f"LTP {ltp:.2f} near S1={s1_val:.2f}")
        elif near_resist and not near_support:
            c8 = ConditionResult("C8:S/R", bull_pts=0.0, bear_pts=1.0,
                                 label="Near Resistance", detail=f"LTP {ltp:.2f} near R1={r1_val:.2f}")
        elif near_pivot:
            c8 = ConditionResult("C8:S/R", bull_pts=0.0, bear_pts=0.0,
                                 label="At Pivot", detail=f"LTP {ltp:.2f} at pivot {pivot_val:.2f}")
        else:
            c8 = ConditionResult("C8:S/R", bull_pts=0.0, bear_pts=0.0,
                                 label="Between Levels", detail=f"P={pivot_val:.2f} R1={r1_val:.2f} S1={s1_val:.2f}")
        conditions.append(c8)

        # ── Aggregate scores ────────────────────────────────────────────────
        bull_score = sum(c.bull_pts for c in conditions)
        bear_score = sum(c.bear_pts for c in conditions)
        bull_pass  = sum(1 for c in conditions if c.bull_pts > 0)
        bear_pass  = sum(1 for c in conditions if c.bear_pts > 0)

        # Determine verdict
        long_ok  = bull_score >= self.pass_threshold and bull_pass >= self.min_conditions
        short_ok = bear_score >= self.pass_threshold and bear_pass >= self.min_conditions

        if long_ok and (not short_ok or bull_score > bear_score):
            verdict = "LONG"
            signal  = Signal.BUY
        elif short_ok:
            verdict = "SHORT"
            signal  = Signal.SELL
        else:
            verdict = "HOLD"
            signal  = Signal.HOLD

        # ── Position management ─────────────────────────────────────────────
        if verdict == "LONG":
            stop_loss = ltp - self.sl_atr_mult * atr_val
            t1        = ltp + self.t1_atr_mult * atr_val
            t2        = ltp + self.t2_atr_mult * atr_val
            risk_pts  = ltp - stop_loss
            rr        = (t1 - ltp) / (risk_pts + 1e-9)
        elif verdict == "SHORT":
            stop_loss = ltp + self.sl_atr_mult * atr_val
            t1        = ltp - self.t1_atr_mult * atr_val
            t2        = ltp - self.t2_atr_mult * atr_val
            risk_pts  = stop_loss - ltp
            rr        = (ltp - t1) / (risk_pts + 1e-9)
        else:
            stop_loss = t1 = t2 = risk_pts = rr = 0.0

        return ScalpResult(
            conditions   = conditions,
            bull_score   = round(bull_score, 2),
            bear_score   = round(bear_score, 2),
            bull_pass    = bull_pass,
            bear_pass    = bear_pass,
            verdict      = verdict,
            signal       = signal,
            entry        = round(ltp, 4),
            stop         = round(stop_loss, 4),
            t1           = round(t1, 4),
            t2           = round(t2, 4),
            risk_reward  = round(rr, 2),
            risk_pts     = round(risk_pts, 4),
            vwap         = round(vwap_val, 4),
            atr          = round(atr_val, 4),
            pivot        = round(pivot_val, 4),
            r1           = round(r1_val, 4),
            s1           = round(s1_val, 4),
            r2           = round(r2_val, 4),
            s2           = round(s2_val, 4),
            rsi          = round(rsi_val, 2),
            stoch_k      = round(k_val, 2),
            stoch_d      = round(d_val, 2),
            macd_hist    = round(hist_val, 6),
            bb_pct       = round(bb_pct, 3),
            volume_ratio = round(vol_ratio, 2),
        )


# ── Strategy adapter (BaseStrategy interface) ──────────────────────────────────

class ScalpStrategy(BaseStrategy):
    """Wraps ScalpEngine as a BaseStrategy for the bot loop."""

    name = "scalp"

    def __init__(self, preset: str = DEFAULT_PRESET) -> None:
        self.engine = ScalpEngine.from_preset(preset)

    def analyze(self, df: pd.DataFrame) -> StrategyResult:
        result = self.engine.analyze(df)
        indicators = {
            "rsi":          result.rsi,
            "stoch_k":      result.stoch_k,
            "stoch_d":      result.stoch_d,
            "macd_hist":    result.macd_hist,
            "bb_pct":       result.bb_pct,
            "volume_ratio": result.volume_ratio,
            "bull_score":   result.bull_score,
            "bear_score":   result.bear_score,
            "vwap":         result.vwap,
            "atr":          result.atr,
        }
        return StrategyResult(
            signal     = result.signal,
            reason     = f"{result.verdict} score={result.bull_score if result.signal==Signal.BUY else result.bear_score:.1f}",
            indicators = indicators,
        )
