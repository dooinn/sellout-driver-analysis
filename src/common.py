"""
공통 유틸리티: 데이터 로드/생성, adstock, 정답 효과, 효과 크기 계산, 차트 헬퍼
모든 스크립트와 노트북이 이 모듈을 공유합니다.
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "sample_sellout_data.csv"
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)

FACTORS = ["기본 수요", "계절성", "할인", "프로모션", "광고", "공휴일", "기온", "결품"]
FACTOR_COLORS = {
    "기본 수요": "#aab4c2", "계절성": "#23998b", "할인": "#df7616", "프로모션": "#c2336a",
    "광고": "#2f6bd4", "공휴일": "#7a50c4", "기온": "#86972a", "결품": "#8e3a2a",
}
REQUIRED_COLUMNS = ["week", "sellout", "discount_pct", "promo", "ad_spend", "holiday", "temp", "stockout_days"]

# 가상 데이터를 만들 때 사용한 '정답' 효과
TRUE_PARAMS = dict(base=900, trend=1.5, season_amp=120, discount=14, promo=260,
                   ad_coef=45, ad_decay=0.5, holiday=180, temp=-6, temp_ref=14, stockout=-95)


def make_synthetic(n=104, seed=7):
    """정답 효과를 알고 있는 가상 주간 셀아웃 데이터 생성"""
    rng = np.random.default_rng(seed)
    weeks = pd.date_range("2024-10-07", periods=n, freq="W-MON")
    t = np.arange(n)
    P = TRUE_PARAMS
    season = P["season_amp"] * np.sin(2 * np.pi * (t - 8) / 52)
    discount = np.where(rng.random(n) < 0.3, rng.choice([5, 10, 15, 20], n), 0)
    promo = (rng.random(n) < 0.18).astype(int)
    ad = np.where(rng.random(n) < 0.45, rng.uniform(5, 30, n), 0).round(1)   # 백만원
    holiday = np.isin(t % 52, [11, 12, 16, 17, 33, 38, 39]).astype(int)
    temp = (14 + 11 * np.sin(2 * np.pi * (t - 20) / 52) + rng.normal(0, 2.5, n)).round(1)
    stockout = np.where(rng.random(n) < 0.08, rng.integers(1, 5, n), 0)
    sales = (P["base"] + P["trend"] * t + season + P["discount"] * discount + P["promo"] * promo
             + P["ad_coef"] * np.log1p(adstock(ad, P["ad_decay"])) + P["holiday"] * holiday
             + P["temp"] * (temp - P["temp_ref"]) + P["stockout"] * stockout + rng.normal(0, 45, n))
    return pd.DataFrame(dict(week=weeks, sellout=sales.round(), discount_pct=discount, promo=promo,
                             ad_spend=ad, holiday=holiday, temp=temp, stockout_days=stockout))


def load_data(path=DATA_PATH):
    """실제 데이터로 바꿀 때는 이 함수만 수정하세요. 필요한 컬럼: REQUIRED_COLUMNS"""
    df = pd.read_csv(path, parse_dates=["week"])
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"데이터에 필요한 컬럼이 없습니다: {sorted(missing)}")
    return df.sort_values("week").reset_index(drop=True)


def adstock(x, decay):
    """광고 이월 효과: 이번 주 광고 + 지난주 효과 × decay"""
    out = np.zeros(len(x))
    for i in range(len(x)):
        out[i] = x[i] + (decay * out[i - 1] if i else 0)
    return out


def mape(y, p):
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(p)) / np.asarray(y)) * 100)


def r2(y, p):
    y, p = np.asarray(y), np.asarray(p)
    return float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def true_contrib(df):
    """가상 데이터의 정답 기여도 (실제 데이터에서는 알 수 없음)"""
    t = np.arange(len(df), dtype=float)
    P = TRUE_PARAMS
    return {
        "기본 수요": P["base"] + P["trend"] * t,
        "계절성": P["season_amp"] * np.sin(2 * np.pi * (t - 8) / 52),
        "할인": P["discount"] * df.discount_pct.values,
        "프로모션": P["promo"] * df.promo.values,
        "광고": P["ad_coef"] * np.log1p(adstock(df.ad_spend.values, P["ad_decay"])),
        "공휴일": P["holiday"] * df.holiday.values,
        "기온": P["temp"] * (df.temp.values - P["temp_ref"]),
        "결품": P["stockout"] * df.stockout_days.values,
    }


def effects(contrib, df):
    """모든 모델에 같은 정의로 효과 크기 계산 → 모델끼리, 그리고 정답과 비교 가능"""
    def gap(k, mask):
        return float(contrib[k][mask].mean() - contrib[k][~mask].mean())

    def slope(k, x):
        return float(np.polyfit(np.asarray(x, float), contrib[k], 1)[0])

    s = np.asarray(contrib["계절성"])
    return {
        "할인": slope("할인", df.discount_pct),                 # 할인 1%p당
        "프로모션": gap("프로모션", df.promo.values == 1),       # 프로모션 주 vs 아닌 주
        "광고": gap("광고", df.ad_spend.values > 0),            # 광고한 주 vs 안 한 주
        "공휴일": gap("공휴일", df.holiday.values == 1),
        "기온": slope("기온", df.temp),                         # 1°C당
        "결품": slope("결품", df.stockout_days),                # 결품 1일당
        "계절성": float((s.max() - s.min()) / 2),               # 성수기~비수기 폭의 절반
    }


def effects_table(est_by_model, df):
    """여러 모델의 효과 추정치를 정답과 나란히 놓은 표"""
    truth = effects(true_contrib(df), df)
    tab = pd.DataFrame({"정답": truth, **est_by_model}).round(1)
    err = {m: np.mean([abs(v[k] - truth[k]) / abs(truth[k]) * 100 for k in truth]) for m, v in est_by_model.items()}
    tab.loc["평균 오차율(%)"] = [np.nan] + [round(err[m], 1) for m in est_by_model]
    return tab


# ------------------------------------------------------------------ 차트 헬퍼 (노트북용)
def plot_stacked(df, contrib, title="주별 요인 기여도", ax=None):
    import matplotlib.pyplot as plt
    ax = ax or plt.subplots(figsize=(12, 4.5))[1]
    x = np.arange(len(df))
    pos, neg = np.zeros(len(df)), np.zeros(len(df))
    for k in FACTORS:
        v = np.asarray(contrib[k])
        p, q = np.clip(v, 0, None), np.clip(v, None, 0)
        ax.bar(x, p, bottom=pos, color=FACTOR_COLORS[k], width=0.8, label=k)
        ax.bar(x, q, bottom=neg, color=FACTOR_COLORS[k], width=0.8)
        pos += p
        neg += q
    ax.plot(x, df.sellout.values, color="black", lw=1.3, label="실제 판매")
    ticks = x[::13]
    ax.set_xticks(ticks, [df.week.dt.strftime("%y.%m")[i] for i in ticks])
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_title(title)
    ax.set_ylabel("판매량 (개)")
    ax.legend(ncol=5, fontsize=8, loc="upper left", frameon=False)
    return ax


def plot_waterfall(contrib, i, actual=None, title=None, ax=None):
    """i번째 주: 기본 수요에서 출발해 요인별로 더하고 빼서 예측치에 도달하는 과정"""
    import matplotlib.pyplot as plt
    ax = ax or plt.subplots(figsize=(8, 4.5))[1]
    cum, rows = 0.0, []
    for k in FACTORS:
        v = float(contrib[k][i])
        rows.append((k, cum, v))
        cum += v
    labels = [r[0] for r in rows] + ["모델 예측"] + (["실제 판매"] if actual is not None else [])
    for j, (k, start, v) in enumerate(rows):
        left = start if v >= 0 else start + v
        ax.barh(j, abs(v), left=left if j else 0, color=FACTOR_COLORS[k])
        ax.text(max(start, start + v) + 8, j, f"{v:+.0f}" if j else f"{v:.0f}", va="center", fontsize=9)
    ax.barh(len(rows), cum, color="#1d5bb8")
    ax.text(cum + 8, len(rows), f"{cum:.0f}", va="center", fontsize=9, weight="bold")
    if actual is not None:
        ax.barh(len(rows) + 1, actual, color="#555")
        ax.text(actual + 8, len(rows) + 1, f"{actual:.0f}", va="center", fontsize=9, weight="bold")
    ax.set_yticks(range(len(labels)), labels)
    ax.invert_yaxis()
    ax.set_xlabel("판매량 (개)")
    ax.set_title(title or f"{i}번째 주 기여도 분해")
    return ax
