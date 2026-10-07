"""
같은 셀아웃 데이터로 7개 모델 비교
  1) 선형 회귀(OLS)  2) 릿지 회귀  3) GAM(스플라인 가법 모델)  4) LightGBM + SHAP
  5) XGBoost + SHAP  6) 구조적 시계열(변하는 기본수요)  7) 베이지안 MMM
비교 기준
  - 정확도: 학습 기간 R²/MAPE, 시계열 교차검증 MAPE(앞부분으로 학습 → 다음 8주 예측, 4회 반복)
  - 효과 추정: 요인별 효과 크기를 같은 정의로 계산해 가상 데이터의 정답과 비교
  - 주별 기여도: 모델별로 같은 주를 어떻게 설명하는지
실행: python src/compare_models.py
"""
import json
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import SplineTransformer, StandardScaler

from common import load_data, adstock, mape, effects, true_contrib, FACTORS, OUT_DIR

warnings.filterwarnings("ignore")
FOLD_STARTS = [72, 80, 88, 96]   # 이 주까지로 학습하고 다음 8주를 예측
FOLD_LEN = 8
DECAYS = np.arange(0, 0.91, 0.1)


# ---------------------------------------------------------------- 공통 입력
def base_cols(df, temp_mean):
    t = np.arange(len(df), dtype=float)
    return dict(t=t, s1=np.sin(2 * np.pi * t / 52), c1=np.cos(2 * np.pi * t / 52),
                disc=df.discount_pct.values.astype(float), promo=df.promo.values.astype(float),
                hol=df.holiday.values.astype(float), temp=df.temp.values - temp_mean,
                stock=df.stockout_days.values.astype(float))


GROUP = {"t": "기본 수요", "s1": "계절성", "c1": "계절성", "disc": "할인", "promo": "프로모션",
         "ad": "광고", "hol": "공휴일", "temp": "기온", "stock": "결품"}


# ---------------------------------------------------------------- 1) OLS, 2) Ridge
def linear_model(df, tr, ridge=False):
    """tr: 학습에 쓸 행 인덱스(앞부분). 반환: 전체 기간 예측, 요인별 기여도"""
    y = df.sellout.values.astype(float)
    temp_mean = df.temp.values[tr].mean()
    best = None
    for d in DECAYS:
        C = base_cols(df, temp_mean)
        C["ad"] = np.log1p(adstock(df.ad_spend.values, d))
        names = list(C)
        X = np.column_stack([C[k] for k in names])
        if ridge:
            sc = StandardScaler().fit(X[tr])
            m = RidgeCV(alphas=np.logspace(-3, 2, 30)).fit(sc.transform(X[tr]), y[tr])
            coef = m.coef_ / sc.scale_
            icpt = m.intercept_ - (coef * sc.mean_).sum()
        else:
            M = np.column_stack([np.ones(len(tr)), X[tr]])
            b, *_ = np.linalg.lstsq(M, y[tr], rcond=None)
            icpt, coef = b[0], b[1:]
        p = icpt + X @ coef
        r2 = 1 - ((y[tr] - p[tr]) ** 2).sum() / ((y[tr] - y[tr].mean()) ** 2).sum()
        if best is None or r2 > best[0]:
            best = (r2, p, icpt, coef, X, names, d)
    _, p, icpt, coef, X, names, d = best
    contrib = {f: np.zeros(len(df)) for f in FACTORS}
    contrib["기본 수요"] += icpt
    for j, k in enumerate(names):
        contrib[GROUP[k]] += coef[j] * X[:, j]
    return p, contrib, {"decay": round(float(d), 1)}


# ---------------------------------------------------------------- 3) GAM (스플라인 + 릿지 벌점)
def gam_model(df, tr):
    y = df.sellout.values.astype(float)
    temp_mean = df.temp.values[tr].mean()
    best = None
    for d in DECAYS:
        raw = base_cols(df, temp_mean)
        raw["ad"] = adstock(df.ad_spend.values, d)
        # 곡선으로 배울 요인: 추세, 할인, 기온, 광고 / 직선: 나머지
        spline = {"t": 4, "disc": 3, "temp": 5, "ad": 5}
        ref = {"t": None, "disc": 0.0, "temp": 0.0, "ad": 0.0}   # 효과 0으로 둘 기준점
        blocks, refs = {}, {}
        for k, v in raw.items():
            if k in spline:
                st = SplineTransformer(n_knots=spline[k], degree=3, extrapolation="linear").fit(v[tr, None])
                blocks[k] = st.transform(v[:, None])
                refs[k] = st.transform(np.array([[ref[k]]])) if ref[k] is not None else None
            else:
                blocks[k] = v[:, None]
                refs[k] = np.zeros((1, 1))
        names = list(blocks)
        X = np.column_stack([blocks[k] for k in names])
        sc = StandardScaler().fit(X[tr])
        sc.scale_[sc.scale_ == 0] = 1
        m = RidgeCV(alphas=np.logspace(-3, 2, 30)).fit(sc.transform(X[tr]), y[tr])
        coef = m.coef_ / sc.scale_
        icpt = m.intercept_ - (coef * sc.mean_).sum()
        p = icpt + X @ coef
        r2 = 1 - ((y[tr] - p[tr]) ** 2).sum() / ((y[tr] - y[tr].mean()) ** 2).sum()
        if best is None or r2 > best[0]:
            best = (r2, p, icpt, coef, blocks, refs, names, d)
    _, p, icpt, coef, blocks, refs, names, d = best
    contrib = {f: np.zeros(len(df)) for f in FACTORS}
    contrib["기본 수요"] += icpt
    j = 0
    for k in names:
        w = blocks[k].shape[1]
        c = blocks[k] @ coef[j:j + w]
        if refs[k] is not None:                      # 기준점 값은 기본 수요로 옮김
            r0 = float((refs[k] @ coef[j:j + w]).item())
            c = c - r0
            contrib["기본 수요"] += r0
        contrib[GROUP[k]] += c
        j += w
    return p, contrib, {"decay": round(float(d), 1)}


# ---------------------------------------------------------------- 4) LightGBM, 5) XGBoost (+ SHAP)
TREE_GROUP = {"t": "기본 수요", "s1": "계절성", "c1": "계절성", "disc": "할인", "promo": "프로모션",
              "ad": "광고", "ad_l1": "광고", "ad_l2": "광고", "ad_l3": "광고",
              "hol": "공휴일", "temp": "기온", "stock": "결품"}


def tree_features(df):
    """트리 모델 입력: 원래 값 그대로 + 광고 지난 1~3주 값(이월 효과를 스스로 배우도록)"""
    t = np.arange(len(df), dtype=float)
    ad = df.ad_spend.values.astype(float)
    lag = lambda x, k: np.concatenate([np.zeros(k), x[:-k]])
    return pd.DataFrame(dict(t=t, s1=np.sin(2 * np.pi * t / 52), c1=np.cos(2 * np.pi * t / 52),
                             disc=df.discount_pct.values, promo=df.promo.values, ad=ad,
                             ad_l1=lag(ad, 1), ad_l2=lag(ad, 2), ad_l3=lag(ad, 3),
                             hol=df.holiday.values, temp=df.temp.values, stock=df.stockout_days.values))


def shap_to_contrib(shap, columns, n):
    """SHAP 값(마지막 열 = 평균 예측)을 요인 그룹별 기여도로 묶기"""
    contrib = {f: np.zeros(n) for f in FACTORS}
    contrib["기본 수요"] += shap[:, -1]
    for j, c in enumerate(columns):
        contrib[TREE_GROUP[c]] += shap[:, j]
    return contrib


def lgb_model(df, tr):
    y = df.sellout.values.astype(float)
    F = tree_features(df)
    m = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.03, num_leaves=7, min_child_samples=5,
                          subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                          random_state=7, verbose=-1)
    m.fit(F.iloc[tr], y[tr])
    shap = m.predict(F, pred_contrib=True)            # TreeSHAP
    return m.predict(F), shap_to_contrib(shap, F.columns, len(df)), {}


def xgb_model(df, tr):
    import xgboost as xgb
    y = df.sellout.values.astype(float)
    F = tree_features(df)
    m = xgb.XGBRegressor(n_estimators=400, learning_rate=0.03, max_depth=3, subsample=0.8,
                         colsample_bytree=0.8, random_state=7)
    m.fit(F.iloc[tr], y[tr])
    shap = m.get_booster().predict(xgb.DMatrix(F), pred_contribs=True)
    return m.predict(F), shap_to_contrib(shap, F.columns, len(df)), {}


# ---------------------------------------------------------------- 6) 구조적 시계열 (Unobserved Components)
def sts_model(df, tr):
    """기본 수요(level + trend)가 시간에 따라 변할 수 있는 시계열 모델 + 외부 요인"""
    import statsmodels.api as sm
    y = df.sellout.values.astype(float)
    n = len(df)
    temp_mean = df.temp.values[tr].mean()
    # 광고 이월률은 학습 구간 OLS로 먼저 고름
    _, _, info = linear_model(df, tr)
    C = base_cols(df, temp_mean)
    C["ad"] = np.log1p(adstock(df.ad_spend.values, info["decay"]))
    names = ["s1", "c1", "disc", "promo", "ad", "hol", "temp", "stock"]
    X = np.column_stack([C[k] for k in names])
    k = len(tr)
    m = sm.tsa.UnobservedComponents(y[:k], level="local linear trend", exog=X[:k]).fit(disp=False, maxiter=500)
    beta = m.params[-len(names):]
    level = np.empty(n)
    level[:k] = m.level.smoothed
    p = np.empty(n)
    p[:k] = level[:k] + X[:k] @ beta
    if k < n:                                          # 학습 이후 구간은 예측
        p[k:] = m.forecast(n - k, exog=X[k:])
        level[k:] = p[k:] - X[k:] @ beta
    contrib = {f: np.zeros(n) for f in FACTORS}
    contrib["기본 수요"] = level
    for j, nm in enumerate(names):
        contrib[GROUP[nm]] += beta[j] * X[:, j]
    return p, contrib, {"decay": info["decay"]}


# ---------------------------------------------------------------- 5) 베이지안 MMM
def mmm_model(df, tr, full=False):
    import pymc as pm
    from sellout_mmm import design, build_model
    y = df.sellout.values.astype(float)
    temp_mean, ad_scale = df.temp.values[tr].mean(), df.ad_spend.max()
    X_tr = design(df.iloc[tr], 0, temp_mean, ad_scale)
    model = build_model(X_tr, y[tr])
    with model:
        idata = pm.sample(1000 if full else 500, tune=1000 if full else 700, chains=4 if full else 2,
                          target_accept=0.92, random_seed=11, progressbar=False)
        pm.set_data({"x_" + k: v for k, v in design(df, 0, temp_mean, ad_scale).items()})
        pp = pm.sample_posterior_predictive(idata, var_names=["mu"] + FACTORS, random_seed=11, progressbar=False)
    ppd = pp.posterior_predictive
    ppd = ppd.to_dataset() if hasattr(ppd, "to_dataset") else ppd
    s = ppd.stack(s=("chain", "draw"))
    p = s["mu"].values.mean(axis=1)
    contrib = {f: s[f].values.mean(axis=1) for f in FACTORS}
    return p, contrib, {}


MODELS = [
    ("ols", "선형 회귀", linear_model, {}),
    ("ridge", "릿지 회귀", linear_model, {"ridge": True}),
    ("gam", "GAM", gam_model, {}),
    ("lgb", "LightGBM + SHAP", lgb_model, {}),
    ("xgb", "XGBoost + SHAP", xgb_model, {}),
    ("sts", "구조적 시계열", sts_model, {}),
    ("mmm", "베이지안 MMM", mmm_model, {}),
]


def main():
    df = load_data()
    y = df.sellout.values.astype(float)
    n = len(df)
    allidx = np.arange(n)
    out_models = []
    truth = true_contrib(df)
    for mid, name, fn, kw in MODELS:
        # 시계열 교차검증
        folds = []
        for s0 in FOLD_STARTS:
            tr = np.arange(s0)
            te = np.arange(s0, min(s0 + FOLD_LEN, n))
            p, _, _ = fn(df, tr, **kw)
            folds.append(round(mape(y[te], p[te]), 1))
        # 전체 기간 학습
        kw_full = dict(kw, full=True) if mid == "mmm" else kw
        p, contrib, info = fn(df, allidx, **kw_full)
        r2 = 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        eff = effects(contrib, df)
        # 기여도 정확도: 요인별 기여도가 정답 기여도와 얼마나 다른지 (주당 평균 절대 오차, 기본 수요 제외)
        # 정답과 같은 '0 기준'을 쓰는 가법 모델만 의미 있음
        contrib_err = float(np.mean([np.mean(np.abs(contrib[f] - truth[f])) for f in FACTORS[2:]]))
        out_models.append(dict(
            id=mid, name=name, r2=round(float(r2), 3), mape=round(mape(y, p), 1),
            cv=folds, cv_mape=round(float(np.mean(folds)), 1),
            pred=np.round(p, 1).tolist(),
            contrib={f: np.round(v, 1).tolist() for f, v in contrib.items()},
            effects={k: round(v, 2) for k, v in eff.items()},
            contrib_err=round(contrib_err, 1), info=info))
        print(f"{name:16s} R2={r2:.3f} MAPE={mape(y, p):.1f} CV={np.mean(folds):.1f} {folds} 기여도오차={contrib_err:.1f}")
        print("   ", {k: round(v, 1) for k, v in eff.items()})

    t_eff = effects(truth, df)
    print("정답", {k: round(v, 1) for k, v in t_eff.items()})
    out = dict(
        weeks=df.week.dt.strftime("%Y-%m-%d").tolist(), actual=y.tolist(),
        inputs=df.drop(columns=["week", "sellout"]).to_dict(orient="list"),
        truth={"effects": {k: round(v, 2) for k, v in t_eff.items()},
               "contrib": {f: np.round(v, 1).tolist() for f, v in truth.items()}},
        models=out_models, folds={"starts": FOLD_STARTS, "len": FOLD_LEN})
    with open(OUT_DIR / "compare_data.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)


if __name__ == "__main__":
    main()
