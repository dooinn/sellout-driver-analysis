"""
셀아웃 베이지안 MMM (PyMC)
- 같은 구조(기본수요+계절성+할인+프로모션+광고+공휴일+기온+결품)를 베이지안으로 추정
- 광고: 기하 adstock(이월률 decay 학습) + 로지스틱 포화(lam 학습)
- 사전분포로 상식 반영: 할인·프로모션·광고·공휴일 효과 ≥ 0, 결품 효과 ≤ 0
- 결과: 계수·기여도의 평균과 90% 신용구간, 광고 반응곡선, 마지막 12주 홀드아웃 비교(OLS vs MMM)
실제 데이터: src/common.py의 load_data()만 교체 (컬럼: week, sellout, discount_pct, promo, ad_spend, holiday, temp, stockout_days)
"""
import json
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import arviz as az

from common import load_data, OUT_DIR

L_MAX = 8          # 광고 이월 최대 주수
HOLDOUT = 12       # 검증용으로 빼둘 마지막 주 수
SEED = 11


def lag_matrix(x, L):
    """x[t-l] 행렬 (n × L)"""
    n = len(x)
    M = np.zeros((n, L))
    for l in range(L):
        M[l:, l] = x[: n - l]
    return M


def design(df, t0=0, temp_mean=None, ad_scale=None):
    n = len(df)
    t = np.arange(t0, t0 + n, dtype=float)
    return dict(
        t=t,
        s1=np.sin(2 * np.pi * t / 52), c1=np.cos(2 * np.pi * t / 52),
        disc=df.discount_pct.values.astype(float),
        promo=df.promo.values.astype(float),
        ad_lag=lag_matrix(df.ad_spend.values.astype(float), L_MAX) / ad_scale,
        hol=df.holiday.values.astype(float),
        temp=df.temp.values - temp_mean,
        stock=df.stockout_days.values.astype(float),
    )


def build_model(X, y):
    with pm.Model() as m:
        intercept = pm.Normal("intercept", 900, 300)
        trend = pm.Normal("trend", 0, 5)
        s1 = pm.Normal("s1", 0, 150)
        c1 = pm.Normal("c1", 0, 150)
        b_disc = pm.HalfNormal("b_disc", 30)
        b_promo = pm.HalfNormal("b_promo", 400)
        b_ad = pm.HalfNormal("b_ad", 400)
        decay = pm.Beta("decay", 2, 2)
        lam = pm.Gamma("lam", 2, 1)
        b_hol = pm.HalfNormal("b_hol", 300)
        b_temp = pm.Normal("b_temp", 0, 20)
        b_stock = pm.HalfNormal("b_stock_abs", 200)
        sigma = pm.HalfNormal("sigma", 100)

        Xd = {k: pm.Data("x_" + k, v) for k, v in X.items()}
        w = decay ** pt.arange(L_MAX)
        adstock = pt.dot(Xd["ad_lag"], w)
        sat = (1 - pt.exp(-lam * adstock)) / (1 + pt.exp(-lam * adstock))

        comps = {
            "기본 수요": intercept + trend * Xd["t"],
            "계절성": s1 * Xd["s1"] + c1 * Xd["c1"],
            "할인": b_disc * Xd["disc"],
            "프로모션": b_promo * Xd["promo"],
            "광고": b_ad * sat,
            "공휴일": b_hol * Xd["hol"],
            "기온": b_temp * Xd["temp"],
            "결품": -b_stock * Xd["stock"],
        }
        for k, v in comps.items():
            pm.Deterministic(k, v)
        mu = pm.Deterministic("mu", sum(comps.values()))
        pm.Normal("y", mu, sigma, observed=y, shape=Xd["t"].shape[0])
    return m


def sample(m):
    with m:
        return pm.sample(1000, tune=1000, chains=4, target_accept=0.92,
                         random_seed=SEED, progressbar=False)


def ols_holdout(df):
    """기존 OLS 모델로 같은 홀드아웃 성능 측정"""
    from sellout_model import build_X
    tr, te = df.iloc[:-HOLDOUT], df.iloc[-HOLDOUT:]
    best = None
    for decay in np.arange(0, 0.91, 0.1):
        X = build_X(df, decay)                     # 전체 기간으로 adstock 계산(정보 누수 없음: 과거만 사용)
        M = np.column_stack([np.ones(len(df))] + list(X.values()))
        beta, *_ = np.linalg.lstsq(M[:-HOLDOUT], tr.sellout.values, rcond=None)
        p = M[:-HOLDOUT] @ beta
        r2 = 1 - ((tr.sellout - p) ** 2).sum() / ((tr.sellout - tr.sellout.mean()) ** 2).sum()
        if best is None or r2 > best[0]:
            best = (r2, M[-HOLDOUT:] @ beta)
    y = te.sellout.values
    return float(np.mean(np.abs(y - best[1]) / y) * 100)


def q(a, axis=0):
    return np.percentile(a, 5, axis=axis), np.percentile(a, 95, axis=axis)


def main():
    df = load_data()
    y = df.sellout.values.astype(float)
    n = len(df)
    temp_mean, ad_scale = df.temp.mean(), df.ad_spend.max()

    # ---------- 1) 홀드아웃 검증: 앞 92주로 학습, 뒤 12주 예측 ----------
    tr = df.iloc[:-HOLDOUT]
    X_tr = design(tr, 0, temp_mean, ad_scale)
    m_tr = build_model(X_tr, tr.sellout.values.astype(float))
    idata_tr = sample(m_tr)
    X_all = design(df, 0, temp_mean, ad_scale)
    with m_tr:
        pm.set_data({"x_" + k: v for k, v in X_all.items()})
        pp = pm.sample_posterior_predictive(idata_tr, var_names=["mu"], random_seed=SEED, progressbar=False)
    mu_te = pp.posterior_predictive["mu"].stack(s=("chain", "draw")).values[-HOLDOUT:].mean(axis=1)
    y_te = y[-HOLDOUT:]
    mmm_holdout = float(np.mean(np.abs(y_te - mu_te) / y_te) * 100)
    ols_hold = ols_holdout(df)

    # ---------- 2) 전체 기간으로 최종 학습 ----------
    m = build_model(X_all, y)
    idata = sample(m)
    posterior = idata.posterior
    posterior = posterior.to_dataset() if hasattr(posterior, "to_dataset") else posterior
    post = posterior.stack(s=("chain", "draw"))
    summ = az.summary(idata, var_names=["intercept", "trend", "b_disc", "b_promo", "b_ad", "decay", "lam",
                                         "b_hol", "b_temp", "b_stock_abs", "sigma"])
    rhat_max = float(summ.r_hat.max())
    divergences = int(idata.sample_stats.diverging.sum())

    factors = ["기본 수요", "계절성", "할인", "프로모션", "광고", "공휴일", "기온", "결품"]
    contrib, lo, hi, period = {}, {}, {}, {}
    for k in factors:
        a = post[k].values               # (n, draws)
        contrib[k] = a.mean(axis=1).round(1).tolist()
        l, h = q(a, axis=1)
        lo[k], hi[k] = l.round(1).tolist(), h.round(1).tolist()
        tot = a.sum(axis=0)
        period[k] = [round(float(tot.mean())), round(float(np.percentile(tot, 5))), round(float(np.percentile(tot, 95)))]
    mu = post["mu"].values
    pred = mu.mean(axis=1)
    pl, ph = q(mu, axis=1)
    contrib["설명 안 된 부분"] = (y - pred).round(1).tolist()
    r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    mape = float(np.mean(np.abs(y - pred) / y) * 100)

    def p(name, scale=1.0):
        a = post[name].values * scale
        return [round(float(a.mean()), 2), round(float(np.percentile(a, 5)), 2), round(float(np.percentile(a, 95)), 2)]

    s1, c1 = post["s1"].values, post["c1"].values
    amp = np.hypot(s1, c1)
    params = {
        "계절성": [round(float(amp.mean()), 1), round(float(np.percentile(amp, 5)), 1), round(float(np.percentile(amp, 95)), 1)],
        "할인": p("b_disc"), "프로모션": p("b_promo"), "광고_decay": p("decay"),
        "공휴일": p("b_hol"), "기온": p("b_temp"), "결품": p("b_stock_abs", -1),
    }

    # ---------- 3) 광고 반응곡선 (매주 같은 금액을 꾸준히 쓸 때의 주간 효과) ----------
    spend = np.arange(0, 61, 2.0)
    dec, lam, bad = post["decay"].values, post["lam"].values, post["b_ad"].values
    geo = np.array([(dec ** l) for l in range(L_MAX)]).sum(axis=0)          # (draws,)
    x = spend[:, None] * geo[None, :] / ad_scale
    resp = bad[None, :] * (1 - np.exp(-lam * x)) / (1 + np.exp(-lam * x))
    rl, rh = q(resp, axis=1)
    avg_spend = float(df.ad_spend.mean())
    # 평균 지출 수준에서 1백만원 추가 시 효과 (한계효과)
    def resp_at(s):
        xx = s * geo / ad_scale
        return bad * (1 - np.exp(-lam * xx)) / (1 + np.exp(-lam * xx))
    marg = resp_at(avg_spend + 1) - resp_at(avg_spend)
    avg_eff = resp_at(avg_spend) / avg_spend

    out = {
        "weeks": df.week.dt.strftime("%Y-%m-%d").tolist(),
        "actual": y.tolist(),
        "pred": pred.round(1).tolist(), "pred_lo": pl.round(1).tolist(), "pred_hi": ph.round(1).tolist(),
        "contrib": contrib, "lo": lo, "hi": hi, "period": period,
        "inputs": df.drop(columns=["week", "sellout"]).to_dict(orient="list"),
        "params": params,
        "metrics": {"r2": round(float(r2), 3), "mape": round(mape, 1),
                    "holdout_mmm": round(mmm_holdout, 1), "holdout_ols": round(ols_hold, 1),
                    "rhat_max": round(rhat_max, 3), "divergences": divergences,
                    "draws": int(post.sizes["s"])},
        "ad_curve": {"spend": spend.tolist(), "mean": resp.mean(axis=1).round(1).tolist(),
                     "lo": rl.round(1).tolist(), "hi": rh.round(1).tolist(),
                     "avg_spend": round(avg_spend, 1),
                     "marginal": [round(float(marg.mean()), 1), round(float(np.percentile(marg, 5)), 1), round(float(np.percentile(marg, 95)), 1)],
                     "average": [round(float(avg_eff.mean()), 1), round(float(np.percentile(avg_eff, 5)), 1), round(float(np.percentile(avg_eff, 95)), 1)]},
    }
    with open(OUT_DIR / "mmm_data.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    print(json.dumps(out["metrics"], ensure_ascii=False))
    print(json.dumps(out["params"], ensure_ascii=False))
    print(json.dumps(out["ad_curve"]["marginal"]), json.dumps(out["ad_curve"]["average"]), out["ad_curve"]["avg_spend"])
    print(json.dumps(period, ensure_ascii=False))


if __name__ == "__main__":
    main()
