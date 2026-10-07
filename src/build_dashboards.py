"""
outputs/*.json 결과를 templates/*.html에 넣어 독립 실행 HTML 대시보드를 만듭니다.
먼저 모델 스크립트를 실행해야 합니다:
    python src/sellout_model.py
    python src/sellout_mmm.py
    python src/compare_models.py
    python src/build_dashboards.py
결과: dashboards/*.html (브라우저로 바로 열 수 있음, 인터넷은 글꼴 로딩에만 사용)
"""
from common import ROOT, OUT_DIR

TPL = ROOT / "templates"
DASH = ROOT / "dashboards"
DASH.mkdir(exist_ok=True)

PAGES = [
    ("decomposition_template.html", "decomposition_data.json", "01_contribution_decomposition.html"),
    ("mmm_template.html", "mmm_data.json", "02_bayesian_mmm_roi.html"),
    ("compare_template.html", "compare_data.json", "03_model_comparison.html"),
]


def main():
    css = (TPL / "base.css").read_text(encoding="utf-8")
    for tpl, data, out in PAGES:
        src = OUT_DIR / data
        if not src.exists():
            print(f"건너뜀: {data} 없음 (해당 모델 스크립트를 먼저 실행하세요)")
            continue
        html = (TPL / tpl).read_text(encoding="utf-8")
        html = html.replace("__CSS__", css).replace("__DATA__", src.read_text(encoding="utf-8"))
        # 제목·글꼴·스타일은 <head>로, 본문은 <body>로
        cut = html.index('<div class="wrap">')
        head, body = html[:cut], html[cut:]
        page = ("<!doctype html>\n<html lang=\"ko\">\n<head>\n<meta charset=\"utf-8\">\n"
                "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
                "<style>body{margin:0}</style>\n" + head + "</head>\n<body>\n" + body + "\n</body>\n</html>\n")
        (DASH / out).write_text(page, encoding="utf-8")
        print("생성:", DASH / out)


if __name__ == "__main__":
    main()
