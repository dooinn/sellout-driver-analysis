"""
셀아웃 기여도 분해 모델 — 선형 회귀(OLS, MMM 방식의 가법 구조)

    셀아웃 = 기본수요(추세) + 계절성 + 할인 + 프로모션 + 광고(adstock) + 공휴일 + 기온 + 결품 + 오차

- 광고 이월률(decay)은 0.0~0.9를 모두 시도해 가장 잘 맞는 값을 선택
- 주별 요인 기여도 = 계수 × 그 주의 값 → outputs/decomposition_data.json
실행: python src/sellout_model.py
"""
import json
import numpy as np

from common import load_data, adstock, mape, r2, OUT_DIR

DECAYS = np.arange(0, 0.91, 0.1)


def build_X(df, decay):
    n = len(df)
    t = np.arange(n)
    return {
        "trend": t.astype(float),
        "s1": np.sin(2 * np.pi * t / 52), "c1": np.cos(2 * np.pi * t / 52),
        "discount": df.discount_pct.values.astype(float),
        "promo": df.promo.values.astype(float),
        "ad": np.log1p(adstock(df.ad_spend.values, decay)),
        "holiday": df.holiday.values.astype(float),
        "temp": df.temp.values - df.temp.mean(),   # 평균 기온 대비
        "stockout": df.stockout_days.values.astype(float),
    }


def fit(df):
    y = df.sellout.values
    best = None
    for decay in DECAYS:                       # 광고 이월효과 탐색
        X = build_X(df, decay)
        M = np.column_stack([np.ones(len(y))] + list(X.values()))
        beta, *_ = np.linalg.lstsq(M, y, rcond=None)
        pred = M @ beta
        score = r2(y, pred)
        if best is None or score > best[0]:
            best = (score, decay, X, beta, pred)
    return best


def decompose(df):
    """반환: 예측치, 요인별 기여도 dict, 계수 dict, 지표 dict"""
    score, decay, X, beta, pred = fit(df)
    coef = dict(zip(["intercept"] + list(X.keys()), beta))
    y = df.sellout.values
    contrib = {
        "기본 수요": coef["intercept"] + coef["trend"] * X["trend"],
        "계절성": coef["s1"] * X["s1"] + coef["c1"] * X["c1"],
        "할인": coef["discount"] * X["discount"],
        "프로모션": coef["promo"] * X["promo"],
        "광고": coef["ad"] * X["ad"],
        "공휴일": coef["holiday"] * X["holiday"],
        "기온": coef["temp"] * X["temp"],
        "결품": coef["stockout"] * X["stockout"],
    }
    metrics = {"r2": round(score, 3), "mape": round(mape(y, pred), 1), "ad_decay": round(float(decay), 1)}
    return pred, contrib, coef, metrics


def main():
    df = load_data()
    pred, contrib, coef, metrics = decompose(df)
    y = df.sellout.values
    contrib = dict(contrib, **{"설명 안 된 부분": y - pred})
    out = {
        "weeks": df.week.dt.strftime("%Y-%m-%d").tolist(),
        "actual": y.tolist(),
        "pred": np.round(pred, 1).tolist(),
        "contrib": {k: np.round(v, 1).tolist() for k, v in contrib.items()},
        "inputs": df.drop(columns=["week", "sellout"]).to_dict(orient="list"),
        "metrics": metrics,
        "coef": {k: round(float(v), 2) for k, v in coef.items()},
    }
    with open(OUT_DIR / "decomposition_data.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    print(metrics)
    print({k: round(v, 2) for k, v in out["coef"].items()})


if __name__ == "__main__":
    main()
