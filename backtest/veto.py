"""
Walk-forward ML entry veto.

The trend rule decides WHEN an entry is possible; a gradient-boosted model scores
each candidate and vetoes the ones it expects to lose. Labels come from a
baseline run's own completed trades, net of costs. Each walk-forward segment's
model is trained only on baseline trades that EXITED before the segment began,
which is what a live retrain on the paper book would see. Model: median imputer +
gradient boosting (120 trees, depth 2, learning rate 0.06, subsample 0.8, min leaf
30; huber loss for the regressor), refit every 40 sessions, at least 150 training
trades.

The label is net return on risk (P&L / max loss) and the features are scale-free
(engine.FEATURES), because QQQ's price rose ~7x over the data.

mode "reg": keep when the predicted return on risk >= ev_min
mode "clf": keep when P(net loss) is below the training quantile keep_frac
"""

import bisect

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from .engine import FEATURES
from .market import ET


def _gb_reg():
    return Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("m", GradientBoostingRegressor(
                         n_estimators=120, max_depth=2, learning_rate=0.06, subsample=0.8,
                         min_samples_leaf=30, loss="huber", random_state=42))])


def _gb_clf():
    return Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("m", GradientBoostingClassifier(
                         n_estimators=120, max_depth=2, learning_rate=0.06, subsample=0.8,
                         min_samples_leaf=30, random_state=42))])


class WalkForwardVeto:
    def __init__(self, base_trades: pd.DataFrame, days: list, mode: str = "reg",
                 ev_min: float = 0.0, keep_frac: float = 0.8,
                 retrain_sessions: int = 40, min_train: int = 150):
        tr = base_trades[base_trades["status"] == "closed"]
        self.X = tr[[f"f_{k}" for k in FEATURES]].to_numpy(float)
        self.y = (tr["pnl"] / tr["max_loss"]).to_numpy(float)
        exit_t = pd.to_datetime(tr["exit_time"], utc=True)
        self.exit_ns = exit_t.dt.as_unit("ns").astype("int64").to_numpy()
        self.bounds = [pd.Timestamp(f"{d} 00:00", tz=ET) for d in days[::retrain_sessions]]
        self.bound_ns = [b.value for b in self.bounds]
        self.mode, self.ev_min, self.keep_frac = mode, ev_min, keep_frac
        self.min_train = min_train
        self.models = {}
        self.active_from = None   # first segment start with a trained model
        self.seen = self.skipped = 0

    def _segment(self, t: pd.Timestamp) -> int:
        return bisect.bisect_right(self.bound_ns, t.value) - 1

    def _model(self, seg: int):
        if seg in self.models:
            return self.models[seg]
        out = None
        if seg >= 0:
            m = self.exit_ns < self.bound_ns[seg]
            if m.sum() >= self.min_train:
                if self.mode == "reg":
                    mod = _gb_reg().fit(self.X[m], self.y[m])
                    out = (mod, self.ev_min)
                else:
                    yl = (self.y[m] < 0).astype(int)
                    if len(np.unique(yl)) == 2:
                        mod = _gb_clf().fit(self.X[m], yl)
                        p = mod.predict_proba(self.X[m])[:, 1]
                        out = (mod, float(np.quantile(p, self.keep_frac)))
                if out is not None and (self.active_from is None or self.bounds[seg] < self.active_from):
                    self.active_from = self.bounds[seg]
        self.models[seg] = out
        return out

    def __call__(self, feat: dict, t: pd.Timestamp) -> bool:
        self.seen += 1
        got = self._model(self._segment(t))
        if got is None:
            return True   # not enough closed trades yet: keep everything
        mod, thr = got
        x = np.array([[feat.get(k, np.nan) for k in FEATURES]], float)
        if self.mode == "reg":
            keep = mod.predict(x)[0] >= thr
        else:
            keep = mod.predict_proba(x)[0, 1] < thr
        if not keep:
            self.skipped += 1
        return bool(keep)

    def importances(self, top: int = 8) -> list[tuple[str, float]]:
        """Feature importances of the most recent trained model."""
        fitted = [v for k, v in sorted(self.models.items()) if v is not None]
        if not fitted:
            return []
        imp = fitted[-1][0].named_steps["m"].feature_importances_
        order = np.argsort(imp)[::-1][:top]
        return [(FEATURES[i], round(float(imp[i]), 3)) for i in order]
