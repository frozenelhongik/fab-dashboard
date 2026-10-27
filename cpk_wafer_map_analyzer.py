"""Auditable statistics for the wafer analyzer; no UI or automatic acceptance.

References: NIST pmc322 (I-MR), eda352/prc231 (normal-IID intervals),
SciPy stats.shapiro/bootstrap, Minitab special-cause tests 1/2/3.
"""
import csv
import io
import warnings

import numpy as np
import pandas as pd
from scipy import stats


def parse_csv(raw: bytes, header: bool = True, sep=",", encoding="utf-8-sig"):
    """Read exact cells, without inferred headers, NA tokens, or numeric types."""
    if not isinstance(raw, (bytes, bytearray)) or not raw:
        raise ValueError("CSV 파일이 비어 있습니다.")
    if len(raw) > 20*1024*1024:
        raise ValueError("CSV는 20 MiB 이하만 지원합니다.")
    if not isinstance(header, bool) or not isinstance(sep, str) or len(sep) != 1 or sep in "\r\n\0":
        raise ValueError("헤더 여부와 한 글자 구분자를 확인하세요.")
    try:
        text = raw.decode(encoding)
    except (UnicodeError, LookupError) as exc:
        raise ValueError("선택한 인코딩으로 읽을 수 없습니다.") from exc
    if "\0" in text:
        raise ValueError("CSV에 NUL 문자가 있습니다. 텍스트 CSV를 선택하세요.")
    rows, lines, skipped = [], [], 0
    try:
        reader = csv.reader(io.StringIO(text, newline=""), delimiter=sep, strict=True)
        for row in reader:
            if not row:  # A blank physical line is recorded, not a sample.
                skipped += 1
                continue
            rows.append(row)
            lines.append(reader.line_num)
            if len(rows) > 50000 + int(header):
                raise ValueError("CSV 데이터는 50,000행 이하만 지원합니다.")
    except csv.Error as exc:
        raise ValueError(f"CSV 구문 오류: {exc}") from exc
    if not rows:
        raise ValueError("CSV에 데이터가 없습니다.")
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise ValueError("행마다 열 수가 다릅니다. 구분자·따옴표·누락 셀을 확인하세요.")
    names = [v.strip() for v in rows[0]] if header else [f"column_{i+1}" for i in range(width)]
    if any(not name for name in names) or len(set(names)) != len(names):
        raise ValueError("빈 열 이름 또는 중복된 열 이름이 있습니다.")
    if header:
        rows, lines = rows[1:], lines[1:]
    if not rows:
        raise ValueError("헤더만 있고 측정 행이 없습니다.")
    frame = pd.DataFrame(rows, columns=names)
    frame.attrs.update(source_lines=lines, blank_lines_skipped=skipped,
                       header=header, separator=sep, encoding=encoding)
    return frame


def clean_numeric(series):
    """Return value/status, retaining every original row and its index."""
    series = pd.Series(series, copy=False)
    values, statuses = [], []
    for item in series:
        if pd.isna(item) or (isinstance(item, str) and not item.strip()):
            value, status = np.nan, "missing"
        else:
            try:
                if isinstance(item, (bool, np.bool_)) or (isinstance(item, str) and "_" in item):
                    raise ValueError
                value = float(item)
                status = "valid" if np.isfinite(value) else "non_finite"
            except (TypeError, ValueError, OverflowError):
                value, status = np.nan, "non_numeric"
        values.append(value if status == "valid" else np.nan)
        statuses.append(status)
    return pd.DataFrame({"value": values, "status": statuses}, index=series.index)


def _values(values, minimum=2):
    try:
        a = np.asarray(values, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("유효한 숫자만 전달하세요.") from exc
    if a.ndim != 1 or a.size < minimum or not np.isfinite(a).all():
        raise ValueError(f"유한한 1차원 측정값이 {minimum}개 이상 필요합니다.")
    return a


def summarize(values, lsl, usl, target=1.33):
    """Overall sample-s Pp/Ppk; cp/cpk are compatibility aliases only."""
    a = _values(values)
    try:
        lsl, usl, target = float(lsl), float(usl), float(target)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("규격과 비교 기준은 유한한 숫자여야 합니다.") from exc
    if not np.isfinite([lsl, usl, target]).all() or lsl >= usl or target <= 0:
        raise ValueError("유한한 LSL < USL과 양수 비교 기준이 필요합니다.")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        mean, sd = float(np.mean(a)), float(np.std(a, ddof=1))
        if not np.isfinite([mean, sd]).all() or sd <= 0:
            raise ValueError("표준편차가 0이거나 계산 범위를 벗어났습니다. 능력지수를 계산할 수 없습니다.")
        pp, ppl, ppu = (usl-lsl)/(6*sd), (mean-lsl)/(3*sd), (usl-mean)/(3*sd)
    if not np.isfinite([pp, ppl, ppu]).all():
        raise ValueError("계산 범위를 벗어났습니다. 단위와 데이터 크기를 확인하세요.")
    ppk = min(ppl, ppu)
    below, above = int(np.count_nonzero(a < lsl)), int(np.count_nonzero(a > usl))
    return dict(n=len(a), mean=mean, std=sd, sample_sd=sd, lsl=lsl, usl=usl,
                pp=pp, ppk=ppk, ppl=ppl, ppu=ppu, cp=pp, cpk=ppk,
                minimum=float(a.min()), maximum=float(a.max()),
                below_count=below, above_count=above, oos_count=below+above,
                oos_rate=(below+above)/len(a), within_count=len(a)-below-above,
                target=target, meets_target_point_estimate=bool(ppk >= target),
                method="overall_sample_sd_ddof1", ddof=1)


def within_sigma(values, groups):
    """Pooled within-subgroup s, weighted by df; no c4 bias correction."""
    a = _values(values)
    g = pd.Series(groups, copy=False).reset_index(drop=True)
    if len(g) != len(a) or g.isna().any() or g.astype(str).str.strip().eq("").any():
        raise ValueError("모든 측정값에 누락 없는 합리적 부분군 ID가 필요합니다.")
    frame = pd.DataFrame({"group": g, "value": a})
    table = frame.groupby("group", sort=False)["value"].agg(["count", "mean", "std"])
    used = table[table["count"] >= 2]
    dof = int((used["count"]-1).sum())
    if dof <= 0:
        raise ValueError("측정값이 2개 이상인 부분군이 필요합니다.")
    if not np.isfinite(used[["mean", "std"]]).all().all():
        raise ValueError("부분군 통계가 계산 범위를 벗어났습니다. 단위를 확인하세요.")
    with np.errstate(over="ignore", invalid="ignore"):
        sigma = float(np.sqrt(((used["count"]-1)*used["std"]**2).sum()/dof))
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("부분군 내 변동이 0이거나 계산할 수 없습니다.")
    return dict(sigma=sigma, sigma_within=sigma, dof=dof, n=len(a),
                used_n=int(used["count"].sum()), used_groups=len(used),
                total_groups=len(table), singletons=int((table["count"] == 1).sum()),
                per_group=table.reset_index(), method="pooled_s_no_c4")


def stability(values):
    """Exploratory Phase-I I-MR. Input must be consecutive individual readings.

    Signal indices are zero-based; run/trend signals mark window endpoints.
    Absence of these selected signals does not establish process stability.
    """
    a = _values(values, 3)
    with np.errstate(over="ignore", invalid="ignore"):
        mean = float(a.mean())
        mr = np.abs(np.diff(a))
        mr_bar = float(mr.mean())
        sigma = mr_bar / 1.128
        lower, upper, mr_upper = mean-3*sigma, mean+3*sigma, 3.267*mr_bar
    if not np.isfinite([mean, mr_bar, sigma, lower, upper, mr_upper]).all() or sigma <= 0:
        raise ValueError("이동범위가 0이거나 계산 범위를 벗어나 I-MR을 계산할 수 없습니다.")
    side = np.sign(a-mean)
    run9 = [i for i in range(8, len(a)) if abs(side[i-8:i+1].sum()) == 9]
    differences = np.diff(a)
    trend6 = [i for i in range(5, len(a)) if
              np.all(differences[i-5:i] > 0) or np.all(differences[i-5:i] < 0)]
    signals = dict(beyond_3sigma=np.flatnonzero((a < lower) | (a > upper)).tolist(),
                   mr_beyond_3sigma=(np.flatnonzero(mr > mr_upper)+1).tolist(),
                   same_side_9=run9, monotonic_6=trend6)
    indices = sorted(set(index for group in signals.values() for index in group))
    return dict(n=len(a), mean=mean, mr_bar=mr_bar, sigma_within=sigma,
                i_lcl=lower, i_ucl=upper, mr_lcl=0.0, mr_ucl=mr_upper,
                moving_ranges=mr, signals=signals, signal_indices=indices,
                has_signals=bool(indices), method="phase1_imr_mr2",
                assumptions="같은 공정의 누락 없는 시간순 개별 측정·독립성. 현재 자료로 한계를 추정한 탐색 결과입니다.")


def normality(values, alpha=0.05, seed=20261027):
    """Shapiro-Wilk, capped at a reproducible random subsample of 5,000."""
    a = _values(values, 3)
    if not 0 < alpha < 1 or np.all(a == a[0]):
        raise ValueError("정규성 검정에는 서로 다른 측정값과 0 < alpha < 1이 필요합니다.")
    indices = np.arange(len(a)) if len(a) <= 5000 else np.sort(
        np.random.default_rng(seed).choice(len(a), 5000, replace=False))
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        try:
            result = stats.shapiro(a[indices])
        except (UserWarning, ValueError) as exc:
            raise ValueError("Shapiro-Wilk 검정을 계산할 수 없습니다.") from exc
    if not np.isfinite([result.statistic, result.pvalue]).all():
        raise ValueError("정규성 검정의 계산 범위를 벗어났습니다.")
    return dict(method="shapiro_wilk", statistic=float(result.statistic),
                pvalue=float(result.pvalue), alpha=alpha,
                reject_normality=bool(result.pvalue < alpha), n=len(a), used_n=len(indices),
                subsampled=len(a) > 5000, seed=seed, sample_indices=indices,
                note="p≥alpha는 정규성의 증명이 아닙니다. Q-Q 그림·샘플링 구조와 함께 해석하세요.")


def capability_ci(values, lsl, usl, confidence=0.95, resamples=400, seed=20261027):
    """Optional normal-IID mean/s intervals and exploratory percentile Pp/Ppk.

    Bootstrap resamples whole rows IID, not correlated time/wafer clusters.
    Degenerate resamples cause unavailable capability intervals, not deletion.
    """
    a = _values(values, 3)
    result = summarize(a, lsl, usl)
    if len(a) > 50000 or not 0 < confidence < 1:
        raise ValueError("구간 계산은 50,000행 이하 및 0 < confidence < 1에서 지원합니다.")
    if isinstance(resamples, bool) or int(resamples) != resamples or not 100 <= resamples <= 5000:
        raise ValueError("재표집 횟수는 100~5,000의 정수여야 합니다.")
    n, sd, mean, tail = len(a), result["std"], result["mean"], (1-confidence)/2
    half = float(stats.t.ppf(1-tail, n-1)*sd/np.sqrt(n))
    sd_ci = sd*np.sqrt((n-1)/stats.chi2.ppf([1-tail, tail], n-1))
    if not np.isfinite([mean-half, mean+half, *sd_ci]).all() or sd_ci[0] <= 0:
        raise ValueError("신뢰구간이 계산 범위를 벗어났습니다. 신뢰수준과 단위를 확인하세요.")
    rng, pp, ppk, degenerate = np.random.default_rng(seed), [], [], 0
    for _ in range(int(resamples)):  # O(n) memory, never an n x B matrix.
        sample = a[rng.integers(0, n, size=n)]
        s, m = float(sample.std(ddof=1)), float(sample.mean())
        if s <= 0 or not np.isfinite([s, m]).all():
            degenerate += 1
            continue
        pp.append((float(usl)-float(lsl))/(6*s))
        ppk.append(min(float(usl)-m, m-float(lsl))/(3*s))
    available = degenerate == 0 and np.isfinite(pp).all() and np.isfinite(ppk).all()
    return dict(confidence=confidence, resamples=int(resamples), seed=seed, n=n,
                mean_ci=[mean-half, mean+half], std_ci=sd_ci.tolist(),
                pp_ci=np.quantile(pp, [tail, 1-tail]).tolist() if available else None,
                ppk_ci=np.quantile(ppk, [tail, 1-tail]).tolist() if available else None,
                capability_available=bool(available), degenerate_resamples=degenerate,
                method="normal_iid_t_chi2_and_iid_percentile_bootstrap",
                note="평균·표준편차 구간은 독립 정규 표본 가정. Pp/Ppk는 IID 백분위 재표집 탐색 구간으로 승인 기준이 아닙니다.")


def xy_metrics(x, y, values, radius, center=(0, 0), center_frac=0.3,
               edge_frac=0.8, duplicate="mean"):
    """Aggregate measured XY sites; never interpolate or invent measurements.

    Region means weight each unique measured site equally after aggregation.
    center: r/R <= center_frac; edge: edge_frac <= r/R <= 1.
    """
    radius = float(radius)
    origin = np.asarray(center, dtype=float)
    if origin.shape != (2,) or not np.isfinite(origin).all() or not np.isfinite(radius) or radius <= 0:
        raise ValueError("유한한 중심 좌표 2개와 양수 반경이 필요합니다.")
    if not 0 <= center_frac < edge_frac <= 1 or duplicate not in {"mean", "median", "reject"}:
        raise ValueError("영역은 0≤중심비율<가장자리비율≤1, 중복 처리법은 mean/median/reject여야 합니다.")
    columns = [clean_numeric(v).reset_index(drop=True) for v in (x, y, values)]
    if len({len(c) for c in columns}) != 1 or not len(columns[0]):
        raise ValueError("X·Y·측정값의 행 수가 같고 비어 있지 않아야 합니다.")
    frame = pd.DataFrame({name: c["value"] for name, c in zip(("x", "y", "value"), columns)})
    valid = np.logical_and.reduce([c["status"].eq("valid").to_numpy() for c in columns])
    invalid_indices = np.flatnonzero(~valid).tolist()
    clean = frame.loc[valid].copy()
    with np.errstate(over="ignore", invalid="ignore"):
        clean["r"] = np.hypot(clean["x"]-origin[0], clean["y"]-origin[1])
    outside = clean["r"] > radius
    outside_indices = clean.index[outside].tolist()
    clean = clean.loc[~outside]
    if clean.empty:
        raise ValueError("반경 안에 유효한 실측 좌표가 없습니다.")
    duplicate_count = int(clean.duplicated(["x", "y"]).sum())
    if duplicate == "reject" and duplicate_count:
        raise ValueError(f"중복 좌표가 {duplicate_count}행 있습니다. 처리법을 명시하세요.")
    points = clean.groupby(["x", "y"], sort=False)["value"].agg(
        value="mean" if duplicate == "reject" else duplicate, count="size").reset_index()
    points["r"] = np.hypot(points["x"]-origin[0], points["y"]-origin[1])
    points["r_norm"] = points["r"]/radius
    if not np.isfinite(points[["x", "y", "value", "r", "r_norm"]]).all().all():
        raise ValueError("좌표 집계가 계산 범위를 벗어났습니다.")
    c = points.loc[points["r_norm"] <= center_frac]
    e = points.loc[points["r_norm"] >= edge_frac]
    cm = float(c["value"].mean()) if len(c) else None
    em = float(e["value"].mean()) if len(e) else None
    delta = em-cm if cm is not None and em is not None else None
    return dict(points=points, input_count=len(frame), invalid_count=len(invalid_indices),
                invalid_indices=invalid_indices, outside_count=len(outside_indices),
                outside_indices=outside_indices, duplicate_count=duplicate_count,
                duplicate_method=duplicate, accepted_count=len(clean), unique_count=len(points),
                center_count=len(c), edge_count=len(e), center_mean=cm, edge_mean=em, delta=delta,
                center_observation_count=int(c["count"].sum()), edge_observation_count=int(e["count"].sum()),
                radius=radius, center=origin.tolist(), center_frac=center_frac, edge_frac=edge_frac,
                method="measured_sites_equal_weight_after_explicit_aggregation")


# The functions above are pure analysis functions; Streamlit starts in main().
import hashlib
import html
import json
import re
import zipfile
from datetime import datetime, timezone
from io import BytesIO

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

APP_VERSION = "2.0.0"
NONE = "(선택 안 함)"
SOURCES = ["합성 예제", "CSV 파일", "CSV 붙여넣기"]


def plain(value):
    """Strict JSON: missing/nonfinite values become null, never NaN/Infinity."""
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [plain(v) for v in value]
    if isinstance(value, pd.DataFrame):
        return plain(value.to_dict("records"))
    if isinstance(value, np.generic):
        return plain(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    return value


def json_bytes(value):
    return json.dumps(plain(value), ensure_ascii=False, indent=2,
                      allow_nan=False).encode("utf-8")


def fmt(value, digits=3):
    return "계산 불가" if value is None or not np.isfinite(value) else f"{value:,.{digits}f}"


def id_like(name):
    if str(name).lower() in {"sampleid", "waferid", "lotid", "rowid", "measurementid"}:
        return True
    name = re.sub(r"([a-z])([A-Z])", r"\1_\2", str(name))
    return bool(re.search(
        r"(^|[_\W])(id|index|seq|no|time|date|timestamp|x|y|row|lot|wafer|chamber|recipe|subgroup)([_\W]|$)"
        r"|(^번호$|식별|시간|날짜|좌표)", str(name).lower()))


def default_measurement(df):
    candidates = [c for c in df if not id_like(c)
                  and clean_numeric(df[c])["status"].eq("valid").any()]
    return next((c for c in candidates
                 if re.search(r"measurement|thickness|value|두께|계측", c, re.I)),
                candidates[0] if candidates else NONE)


def choose(label, columns, key, preferred=None):
    options = [NONE] + list(columns)
    default = preferred if preferred in options else NONE
    return st.selectbox(label, options, index=options.index(default), key=key)


def demo_frame(n=200, mean=200.0, sd=4.0, seed=42):
    rng = np.random.default_rng(seed)
    values = rng.normal(mean, sd, n)
    angle, radius = rng.uniform(0, 2 * np.pi, n), 95 * np.sqrt(rng.random(n))
    return pd.DataFrame({
        "measurement": values,
        "timestamp": pd.date_range("2026-01-01", periods=n, freq="min", tz="UTC").astype(str),
        "lot_id": "DEMO_LOT", "wafer_id": np.where(np.arange(n) < n // 2, "W01", "W02"),
        "chamber": "DEMO_A", "recipe": "DEMO_R1",
        "subgroup": np.arange(n) // 5,
        "x_mm": radius * np.cos(angle), "y_mm": radius * np.sin(angle), "unit": "nm",
    })


def csv_input(prefix, allow_demo=True):
    mode = st.radio("데이터 입력", SOURCES if allow_demo else SOURCES[1:],
                    horizontal=True, key=f"{prefix}_source")
    parsing = {}
    if mode == SOURCES[0]:
        cols = st.columns(4)
        n = cols[0].number_input("샘플 수", 10, 50000, 200, 10, key=f"{prefix}_n")
        mean = cols[1].number_input("생성 평균", value=200.0, key=f"{prefix}_mean")
        sd = cols[2].number_input("생성 표준편차", min_value=0.0, value=4.0, key=f"{prefix}_sd")
        seed = cols[3].number_input("시드", 0, 2147483647, 42, key=f"{prefix}_seed")
        if not np.isfinite([mean, sd]).all():
            st.error("생성 평균과 표준편차는 유한한 숫자여야 합니다.")
            return None
        frame = demo_frame(int(n), mean, sd, int(seed))
        raw, name = frame.to_csv(index=False).encode("utf-8"), "synthetic_demo.csv"
        parsing = {"n": n, "mean": mean, "sd": sd, "seed": seed}
        st.info("합성 예제입니다. 계측값·좌표·시간·그룹은 교육용으로 생성되었습니다.")
    else:
        cols = st.columns(3)
        header = cols[0].checkbox("첫 행이 열 이름", value=True, key=f"{prefix}_header")
        separator = cols[1].selectbox("구분자", [",", ";", "\t"], key=f"{prefix}_sep")
        encoding = cols[2].selectbox("인코딩", ["utf-8-sig", "cp949"], key=f"{prefix}_encoding")
        if mode == SOURCES[1]:
            upload = st.file_uploader("CSV 업로드 · 최대 20 MiB / 50,000행",
                                     type=["csv"], key=f"{prefix}_upload")
            if upload is None:
                st.info("파일을 업로드하세요.")
                return None
            raw, name = upload.getvalue(), upload.name
        else:
            content = st.text_area("CSV 내용", height=150, key=f"{prefix}_paste",
                                   placeholder="measurement\n198\n200\n202")
            if not content.strip():
                return None
            try:
                raw = content.encode(encoding)
            except UnicodeError:
                st.error("선택한 인코딩으로 표현할 수 없는 문자가 있습니다. UTF-8을 선택하세요.")
                return None
            name = "pasted.csv"
        try:
            frame = parse_csv(raw, header=header, sep=separator, encoding=encoding)
        except ValueError as exc:
            st.error(str(exc))
            return None
        numeric_headers = pd.to_numeric(pd.Series(frame.columns), errors="coerce").notna().any()
        if header and numeric_headers:
            st.warning("숫자로 된 열 이름이 있습니다. 헤더 없는 파일이면 '첫 행이 열 이름'을 해제하세요.")
            header_key = f"{prefix}_numeric_header_ok_{hashlib.sha256(raw).hexdigest()[:16]}"
            if not st.checkbox("숫자가 실제 열 이름임을 확인", key=header_key):
                return None
        parsing = {"header": header, "separator": separator, "encoding": encoding}
    if any(str(c).startswith("__qa_") for c in frame):
        st.error("__qa_로 시작하는 열 이름은 내부 기록용입니다. 열 이름을 변경하세요.")
        return None
    skipped = int(frame.attrs.get("blank_lines_skipped", 0))
    if skipped:
        st.caption(f"내용 없는 물리적 빈 줄 {skipped}개를 건너뛰었습니다. 유효 데이터 행 수와 별도로 기록합니다.")
    meta = {"name": name, "sha256": hashlib.sha256(raw).hexdigest(),
            "source": mode, "parsing": parsing, "raw_rows": len(frame),
            "blank_lines_skipped": skipped}
    return frame, meta


def filter_frame(frame, prefix):
    with st.expander("제품·장비·로트 등 분석 대상 필터"):
        fields = st.multiselect("필터에 사용할 열", list(frame.columns), key=f"{prefix}_filter_fields")
        filters = {}
        for i, field in enumerate(fields):
            # Preserve original types; masks use stable string labels only.
            labels = frame[field].astype("string").fillna("(결측)")
            options = list(pd.unique(labels))
            if len(options) > 1000:
                st.warning(f"{field}: 고유값이 1,000개를 넘습니다. 다른 그룹 열을 선택하세요.")
                continue
            selected = st.multiselect(f"{field} 포함값", options, default=options,
                                      key=f"{prefix}_filter_{i}_{field}")
            filters[field] = selected
            frame = frame.loc[labels.isin(selected)].copy()
    return frame, filters


def prepare_data(frame, prefix, declared_unit):
    frame, filters = filter_frame(frame, prefix)
    st.caption(f"필터 후 {len(frame):,}행 · 미리보기 최대 50행")
    st.dataframe(frame.head(50), hide_index=True, width="stretch")
    column = choose("분석할 계측 열", frame.columns, f"{prefix}_value", default_measurement(frame))
    if column == NONE:
        st.info("계측값 열을 선택하세요. ID·좌표 열은 자동 선택하지 않습니다.")
        return None
    fingerprint = hashlib.sha256(
        pd.util.hash_pandas_object(frame, index=True).values.tobytes()
        + column.encode("utf-8") + declared_unit.encode("utf-8")).hexdigest()
    state_key = f"{prefix}_analysis_fingerprint"
    if st.session_state.get(state_key) != fingerprint:
        reset = [f"{prefix}_allow_drop", f"{prefix}_id_ok", "comparison_confirm"]
        if prefix == "main":
            reset += ["ordered_confirm", "subgroup_confirm", "map_single"]
        for key in reset:
            st.session_state[key] = False
        st.session_state[state_key] = fingerprint
    if id_like(column):
        st.warning("선택한 열 이름이 ID·시간·좌표·그룹 정보와 유사합니다.")
        if not st.checkbox("이 열이 연속형 계측값임을 확인", key=f"{prefix}_id_ok"):
            return None
    unit_column = choose("단위 열 (없으면 입력한 단위 사용)", frame.columns,
                         f"{prefix}_unit_column", "unit")
    if unit_column != NONE:
        units = frame[unit_column].astype("string").fillna("").str.strip()
        if units.eq("").any() or set(units.unique()) != {declared_unit.strip()}:
            st.error("단위 열에 결측·혼합 단위가 있거나 입력 단위와 다릅니다. 필터 또는 원본을 확인하세요.")
            return None
    quality = clean_numeric(frame[column])
    audit = frame.copy()
    audit["__qa_record"] = frame.index.to_numpy() + 1
    source_lines = frame.attrs.get("source_lines")
    if source_lines is not None:
        audit["__qa_source_line_end"] = [source_lines[i] for i in frame.index]
    audit["__qa_status"] = quality["status"]
    audit["__qa_value"] = quality["value"]
    counts = {k: int(quality.status.eq(k).sum())
              for k in ["valid", "missing", "non_numeric", "non_finite"]}
    st.write(f"유효 {counts['valid']} / 결측 {counts['missing']} / "
             f"문자 {counts['non_numeric']} / 비유한값(NaN·Inf) {counts['non_finite']}")
    excluded = quality.status.ne("valid")
    if excluded.any():
        st.dataframe(audit.loc[excluded].head(100), hide_index=True, width="stretch")
        if not st.checkbox("위의 부적합 행을 제외하고 분석", key=f"{prefix}_allow_drop"):
            st.warning("부적합 행 처리에 동의하기 전에는 지수를 계산하지 않습니다.")
            return None
    clean = audit.loc[~excluded].copy()
    mixed = [c for c in ["product", "chamber", "recipe"] if c in clean
             and clean[c].astype(str).nunique() > 1]
    if mixed:
        st.warning("여러 조건이 섞여 있습니다: " + ", ".join(mixed) +
                   ". 전체 산포와 조건별 차이를 구분하려면 필터·그룹 비교를 사용하세요.")
    return {"frame": clean, "audit": audit, "counts": counts, "filters": filters,
            "column": column, "unit_column": unit_column, "unit": declared_unit,
            "excluded": int(excluded.sum()), "filtered_rows": len(frame)}


def histogram(values, lsl, usl, unit, name="분포"):
    fig = go.Figure(go.Histogram(x=values, nbinsx=35, name=name))
    for x, label in [(lsl, "LSL"), (usl, "USL")]:
        fig.add_vline(x=x, line_dash="dash", line_color="#c2410c", annotation_text=label)
    fig.update_layout(title=name, xaxis_title=f"계측값 ({unit})", yaxis_title="관측 수",
                      template="plotly_white", height=370, bargap=.04)
    return fig


def draw(fig, key):
    st.plotly_chart(fig, width="stretch", key=key,
                    config={"displaylogo": False, "toImageButtonOptions": {"format": "svg"}})


def time_analysis(data, config):
    frame = data["frame"]
    options = [c for c in frame if not str(c).startswith("__qa_")]
    time_col = choose("측정 시각 열", options, "time_column", "timestamp")
    subgroup = choose("합리적 부분군 열", options, "subgroup_column")
    ordered = frame.copy()
    time_ok = True
    naive_timezone = st.selectbox("오프셋 없는 시각의 시간대", ["Asia/Seoul", "UTC"], key="time_zone")
    if time_col != NONE:
        def parse_time(value):
            try:
                stamp = pd.Timestamp(value)
                if pd.isna(stamp):
                    return pd.NaT
                if stamp.tzinfo is None:
                    stamp = stamp.tz_localize(naive_timezone)
                return stamp.tz_convert("UTC")
            except (ValueError, TypeError, OverflowError):
                return pd.NaT
        parsed = pd.to_datetime(ordered[time_col].map(parse_time), errors="coerce", utc=True)
        time_ok = bool(parsed.notna().all())
        if not time_ok:
            st.warning(f"시각 해석 실패 {int(parsed.isna().sum())}행. 파일 순서를 표시하며 I-MR은 비활성화합니다.")
        else:
            ordered = ordered.assign(__qa_time=parsed).sort_values("__qa_time", kind="stable")
            if ordered["__qa_time"].duplicated().any():
                st.warning("같은 시각의 관측이 있습니다. 해당 관측 간 순서는 파일 순서를 따릅니다.")
    else:
        st.caption("시각 열이 없어 파일의 행 순서를 표시합니다. 실제 측정 시간 순서인지 별도로 확인하세요.")
    x = ordered["__qa_time"] if time_col != NONE and time_ok else ordered["__qa_record"]
    y = ordered["__qa_value"].to_numpy(float)
    line = go.Figure(go.Scatter(x=x, y=y, mode="lines+markers", name="계측값"))
    line.update_layout(title="측정 순서 · 시간 변화", xaxis_title="UTC 시각" if time_col != NONE and time_ok else "원본 데이터 행 번호",
                       yaxis_title=f"계측값 ({config['unit']})", template="plotly_white", height=380)
    for limit, label in [(config["lsl"], "LSL"), (config["usl"], "USL")]:
        line.add_hline(y=limit, line_dash="dash", line_color="#c2410c", annotation_text=label)
    mode = st.radio("단기 변동 추정", ["추정 안 함", "연속 개별값 I-MR", "합리적 부분군 합동 s"],
                    horizontal=True, key="within_mode")
    result = {"time_column": time_col, "subgroup_column": subgroup, "mode": mode,
              "time_parse_valid": time_ok, "naive_timezone": naive_timezone,
              "stability": None, "within": None}
    extra_fig = None
    if mode == "연속 개별값 I-MR":
        confirmed = st.checkbox("같은 공정에서 시간 순서대로 얻은 연속 개별 측정임을 확인", key="ordered_confirm")
        if data["excluded"] or not time_ok:
            st.warning("행 제외 또는 잘못된 시각이 있으면 인접 관측 관계가 불명확해 I-MR을 계산하지 않습니다.")
        elif confirmed:
            try:
                diag = stability(y)
                result["stability"] = diag
                result["within"] = {"sigma": diag["sigma_within"], "method": "MRbar/1.128"}
                for k, label in [("i_lcl", "I-LCL"), ("i_ucl", "I-UCL")]:
                    line.add_hline(y=diag[k], line_dash="dot", line_color="#2563eb", annotation_text=label)
                if diag["has_signals"]:
                    st.warning("선택한 규칙에서 공정 변화 신호가 검출되었습니다.")
                else:
                    st.info("선택한 규칙에서 신호가 검출되지 않았습니다. 공정 안정성을 입증하는 결과는 아닙니다.")
                st.json({k: len(v) for k, v in diag["signals"].items()})
                extra_fig = go.Figure(go.Scatter(x=x.iloc[1:], y=diag["moving_ranges"], mode="lines+markers"))
                extra_fig.add_hline(y=diag["mr_ucl"], line_dash="dot", annotation_text="MR-UCL")
                extra_fig.update_layout(title="Moving Range · 연속 두 관측의 절대 차이",
                                        template="plotly_white", yaxis_title=config["unit"], height=300)
            except ValueError as exc:
                st.warning(str(exc))
        else:
            st.info("연속 개별 측정 조건을 확인하면 I-MR과 Cp/Cpk를 계산합니다.")
    elif mode == "합리적 부분군 합동 s":
        confirmed = st.checkbox("부분군 내 측정들이 동일 조건의 반복 관측임을 확인", key="subgroup_confirm")
        if subgroup == NONE:
            st.info("부분군 열을 선택하세요. 장비 ID나 로트 번호를 부분군으로 간주할지는 샘플링 설계로 결정해야 합니다.")
        elif confirmed:
            try:
                w = within_sigma(y, ordered[subgroup].to_numpy())
                result["within"] = w
                st.dataframe(w["per_group"], hide_index=True, width="stretch")
                st.caption(f"사용 부분군 {w['used_groups']}개 · 제외 단일 관측 부분군 {w['singletons']}개 · 자유도 {w['dof']}")
            except ValueError as exc:
                st.warning(str(exc))
    draw(line, "time_chart")
    if extra_fig is not None:
        draw(extra_fig, "mr_chart")
    if result["within"] is not None:
        sigma = result["within"]["sigma"]
        mu = float(np.mean(y))
        cp = (config["usl"] - config["lsl"]) / (6 * sigma)
        cpk = min(config["usl"] - mu, mu - config["lsl"]) / (3 * sigma)
        if not np.isfinite([cp, cpk]).all():
            st.warning("단기 지수가 수치 계산 범위를 벗어나 표시할 수 없습니다. 단위를 확인하세요.")
            cp, cpk = None, None
        result["within"].update({"cp": cp, "cpk": cpk})
        cols = st.columns(3)
        for col, label, value in zip(cols, ["단기 σ", "Cp", "Cpk"], [sigma, cp, cpk]):
            col.metric(label, fmt(value))
        st.caption("부분군 방식: sqrt(Σ(nᵢ−1)sᵢ² / Σ(nᵢ−1)), c4 보정 없음. I-MR 방식: MR 평균/1.128.")
    st.caption("주황선은 규격, 파란선은 관리한계입니다. I-MR 규칙: 3σ 초과·같은 쪽 9점·단조 증가/감소 6점·MR 한계 초과. 한계는 현재 표본으로 추정합니다.")
    return result, [line] + ([extra_fig] if extra_fig is not None else [])


def group_analysis(data, config):
    frame = data["frame"]
    columns = [c for c in frame if not c.startswith("__qa_")]
    group = choose("비교할 그룹 열", columns, "group_column", "wafer_id")
    if group == NONE:
        return None, None
    labels = frame[group].astype("string").fillna("(결측)")
    if labels.nunique() > 100:
        st.warning("그룹이 100개를 넘습니다. 필터로 범위를 좁히세요.")
        return None, None
    rows, fig = [], go.Figure()
    for name, chunk in frame.groupby(labels, sort=False):
        vals = chunk["__qa_value"].to_numpy(float)
        row = {"group": name, "n": len(vals)}
        try:
            s = summarize(vals, config["lsl"], config["usl"])
            row.update({k: s[k] for k in ["mean", "std", "pp", "ppk", "oos_count"]})
            row["status"] = "계산됨"
        except ValueError as exc:
            row["status"] = str(exc)
        rows.append(row)
        fig.add_trace(go.Box(y=vals, name=str(name), boxpoints="outliers"))
    table = pd.DataFrame(rows)
    st.dataframe(table, hide_index=True, width="stretch")
    fig.update_layout(title=f"{group}별 분포", yaxis_title=config["unit"], template="plotly_white", height=380)
    draw(fig, "group_chart")
    return {"column": group, "results": plain(table)}, fig


def wafer_plot(points, config, title):
    fig = go.Figure(go.Scatter(
        x=points["x"], y=points["y"], mode="markers",
        customdata=points[["value", "count"]].to_numpy(),
        marker={"size": 9, "color": points["value"], "colorscale": "RdYlBu_r",
                "cmin": config["color_min"], "cmax": config["color_max"],
                "colorbar": {"title": config["unit"]}, "line": {"width": .3, "color": "#374151"}},
        hovertemplate="X=%{x:.3f}<br>Y=%{y:.3f}<br>value=%{customdata[0]:.4f}<br>repeats=%{customdata[1]}<extra></extra>"))
    cx, cy, r = config["center_x"], config["center_y"], config["radius"]
    fig.add_shape(type="circle", x0=cx-r, y0=cy-r, x1=cx+r, y1=cy+r,
                  line={"color": "#374151"})
    fig.update_layout(title=title, template="plotly_white", height=480,
                      xaxis={"title": "X (mm)", "range": [cx-r*1.05, cx+r*1.05]},
                      yaxis={"title": "Y (mm)", "range": [cy-r*1.05, cy+r*1.05],
                             "scaleanchor": "x", "scaleratio": 1})
    return fig


def map_analysis(data, config, prefix="map", source_is_synthetic=False):
    frame = data["frame"]
    columns = [c for c in frame if not c.startswith("__qa_")]
    cols = st.columns(3)
    with cols[0]:
        xcol = choose("X 좌표 열 · mm", columns, f"{prefix}_x", "x_mm")
    with cols[1]:
        ycol = choose("Y 좌표 열 · mm", columns, f"{prefix}_y", "y_mm")
    with cols[2]:
        wafer = choose("웨이퍼 ID 열", columns, f"{prefix}_wafer", "wafer_id")
    selected_wafer = None
    if wafer != NONE:
        options = frame[wafer].astype("string").fillna("(결측)")
        selected_wafer = st.selectbox("표시할 웨이퍼", list(pd.unique(options)), key=f"{prefix}_wafer_value")
        frame = frame.loc[options.eq(selected_wafer)]
    if xcol == NONE or ycol == NONE:
        st.info("실제 X·Y 좌표 열을 선택하면 측정 위치만 표시합니다. 좌표가 없으면 맵을 생성하지 않습니다.")
        return None, None
    if xcol == ycol or data["column"] in [xcol, ycol]:
        st.error("X·Y·계측값은 서로 다른 열이어야 합니다.")
        return None, None
    if wafer == NONE and not st.checkbox("한 장의 웨이퍼 좌표임을 확인", key=f"{prefix}_single"):
        st.info("웨이퍼 ID를 지정하거나 한 장의 데이터임을 확인하세요.")
        return None, None
    dup_label = st.selectbox("같은 좌표의 반복 측정", ["오류로 처리", "평균 집계", "중앙값 집계"], key=f"{prefix}_duplicate")
    duplicate = {"오류로 처리": "reject", "평균 집계": "mean", "중앙값 집계": "median"}[dup_label]
    try:
        m = xy_metrics(frame[xcol], frame[ycol], frame["__qa_value"],
                       radius=config["radius"], center=(config["center_x"], config["center_y"]),
                       center_frac=config["center_frac"], edge_frac=config["edge_frac"], duplicate=duplicate)
    except ValueError as exc:
        st.error(str(exc))
        return None, None
    if m["invalid_count"] or m["outside_count"]:
        st.warning(f"좌표 부적합 {m['invalid_count']}행 · 원 밖 {m['outside_count']}행은 맵에서 제외됩니다. 공정능력 분석 표본과 다를 수 있습니다.")
    saturated = int(((m["points"]["value"] < config["color_min"]) |
                     (m["points"]["value"] > config["color_max"])).sum())
    st.caption(f"입력 {m['input_count']}행 → 원내 유효 {m['accepted_count']}행 → 고유 위치 {m['unique_count']}개. "
               f"공통 색 범위 [{config['color_min']:g}, {config['color_max']:g}] 밖 위치 {saturated}개는 끝 색으로 포화됩니다.")
    if wafer != NONE:
        st.caption("위의 공정능력 지수는 필터 후 전체 표본, 이 맵은 선택한 웨이퍼의 좌표를 사용합니다.")
    st.caption("좌표를 보간하지 않고 관측 위치만 표시합니다. 색 범위는 모든 비교 맵에 공통 적용됩니다.")
    title = "합성 예제의 좌표 맵" if source_is_synthetic else "업로드 좌표의 계측 맵"
    fig = wafer_plot(m["points"], config, title)
    draw(fig, f"{prefix}_chart")
    cols = st.columns(3)
    cols[0].metric("중심 관측 평균", fmt(m["center_mean"]))
    cols[1].metric("가장자리 관측 평균", fmt(m["edge_mean"]))
    cols[2].metric("가장자리 − 중심", fmt(m["delta"]))
    st.caption(f"중심 r/R ≤ {config['center_frac']:.2f}: {m['center_count']}위치 · "
               f"가장자리 r/R ≥ {config['edge_frac']:.2f}: {m['edge_count']}위치. "
               "중복 집계 후 각 측정 위치에 같은 가중치를 줍니다. 빈 영역의 차이는 계산하지 않습니다.")
    result = dict(m, x_column=xcol, y_column=ycol, wafer_column=wafer,
                  wafer_value=selected_wafer, duplicate_method=duplicate, synthetic=source_is_synthetic)
    return result, fig


def analyze(source, config, config_valid=True):
    frame, meta = source
    data = prepare_data(frame, "main", config["unit"])
    if data is None:
        return None
    if not config_valid:
        st.info("입력 파일과 필터를 유지합니다. 공통 설정을 수정하면 같은 데이터로 분석을 재개합니다.")
        return None
    values = data["frame"]["__qa_value"].to_numpy(float)
    try:
        summary = summarize(values, config["lsl"], config["usl"], config["target"])
    except ValueError as exc:
        if len(values):
            oos = int(((values < config["lsl"]) | (values > config["usl"])).sum())
            with np.errstate(over="ignore", invalid="ignore"):
                descriptive_mean = float(np.mean(values))
            st.write(f"관측 요약: 유효 {len(values)}개 · 평균 {fmt(descriptive_mean)} · 규격 이탈 {oos}/{len(values)}")
        st.error(str(exc))
        return None
    cols = st.columns(6)
    for col, label, value in zip(cols, ["유효 n", "평균", "전체 표본 s", "Pp", "Ppk", "규격 이탈"],
                                [str(summary["n"]), fmt(summary["mean"]), fmt(summary["std"]),
                                 fmt(summary["pp"]), fmt(summary["ppk"]), str(summary["oos_count"])]):
        col.metric(label, value)
    st.info(f"Ppk {summary['ppk']:.3f}: 설정 목표 {config['target']:.2f} "
            f"{'이상' if summary['ppk'] >= config['target'] else '미만'}. "
            "전체 표본 s(ddof=1)를 사용한 기술통계입니다. 자동 출하·신뢰성 판정으로 사용하지 않습니다.")
    if summary["n"] < 100:
        st.warning("표본이 100개 미만입니다. 지수의 불확실성과 샘플링 대표성을 함께 검토하세요. 100개도 충분성을 보장하지는 않습니다.")
    st.caption("규격 경계값은 규격 내입니다. 관측 이탈 0건은 미래 불량률 0%를 뜻하지 않습니다.")
    tabs = st.tabs(["분포·불확실성", "시간·부분군", "그룹 비교", "계측 좌표 맵"])
    figures = []
    with tabs[0]:
        fig = histogram(values, config["lsl"], config["usl"], config["unit"])
        draw(fig, "histogram")
        figures.append(fig)
        try:
            norm = normality(values)
        except ValueError as exc:
            norm = {"pvalue": None, "note": str(exc), "n": len(values),
                    "method": "shapiro_wilk", "available": False}
        if norm["pvalue"] is None:
            st.warning(norm["note"])
        else:
            st.write(f"Shapiro–Wilk p={norm['pvalue']:.4g} · 사용 n={norm['used_n']}")
            if norm["reject_normality"]:
                st.warning("정규분포 가정이 기각됩니다. 정규 모형의 지수를 합격 판정이나 불량률 예측에 사용하지 마세요.")
            else:
                st.info("정규성 가설이 기각되지 않았습니다. 정규분포 또는 독립성을 입증한 결과는 아닙니다.")
            if norm["subsampled"]:
                st.caption(f"전체 {norm['n']}개에서 고정 시드 {norm['seed']}로 무복원 5,000개를 추출했습니다. " + norm["note"])
        sample = np.sort(values)[np.linspace(0, len(values)-1, min(5000, len(values))).astype(int)]
        (theoretical, actual), (slope, intercept, _) = stats.probplot(sample)
        qq = go.Figure(go.Scatter(x=theoretical, y=actual, mode="markers", name="관측"))
        qq.add_trace(go.Scatter(x=theoretical, y=slope*theoretical+intercept, name="참고 직선"))
        qq.update_layout(title="정규 Q–Q · 최대 5,000개 순서통계량", template="plotly_white", height=320)
        draw(qq, "qq_chart")
        figures.append(qq)
        q1, q3 = np.quantile(values, [.25, .75])
        outliers = int(np.count_nonzero((values < q1-1.5*(q3-q1)) | (values > q3+1.5*(q3-q1))))
        st.caption(f"1.5×IQR 범위 밖 관측 {outliers}개. 이상치 후보는 자동 제거하지 않았습니다.")
        ci = None
        if st.checkbox("평균·s 및 Pp·Ppk의 95% 참고 구간 계산", key="show_ci"):
            try:
                ci = capability_ci(values, config["lsl"], config["usl"])
                st.json({k: ci[k] for k in ["mean_ci", "std_ci", "pp_ci", "ppk_ci"]})
                st.caption(ci["note"])
                if not ci["capability_available"]:
                    st.warning(f"분산 0 등 퇴화 재표집 {ci['degenerate_resamples']}회로 지수 구간을 제공하지 않습니다.")
            except ValueError as exc:
                st.warning(str(exc))
    with tabs[1]:
        temporal, figs = time_analysis(data, config)
        figures.extend(figs)
    with tabs[2]:
        groups, fig = group_analysis(data, config)
        if fig is not None:
            figures.append(fig)
    with tabs[3]:
        wafer, fig = map_analysis(data, config, source_is_synthetic=meta["source"] == SOURCES[0])
        if fig is not None:
            figures.append(fig)
    report = {"app_version": APP_VERSION, "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "source": meta, "config": config, "measurement_column": data["column"],
              "unit_column": data["unit_column"], "filters": data["filters"],
              "filtered_rows": data["filtered_rows"], "quality_counts": data["counts"],
              "summary": {k: v for k, v in summary.items() if k not in ["cp", "cpk"]},
              "normality": norm, "iid_bootstrap": ci,
              "iqr_flagged_count": outliers, "temporal": temporal, "group_comparison": groups,
              "wafer_map": wafer,
              "interpretation": "Descriptive screening; stability, independence, normality and measurement adequacy require review."}
    return {"data": data, "report": report, "figures": figures}


def compare(primary, config):
    bunit = st.text_input("B 계측 단위", value=config["unit"], key="b_unit")
    source = csv_input("b", allow_demo=False)
    if primary is None:
        st.info("먼저 왼쪽 분석 탭에서 유효한 A 데이터를 준비하세요.")
        return None, []
    st.write("현재 분석을 A로 사용합니다. B에도 같은 규격과 색상 범위를 적용합니다.")
    if source is None:
        return None, []
    if bunit.strip() != config["unit"]:
        st.error("A와 B의 단위가 다릅니다. 같은 단위로 변환한 파일을 사용하세요.")
        return None, []
    data = prepare_data(source[0], "b", bunit)
    if data is None:
        return None, []
    if not st.checkbox("A와 B가 같은 계측 항목이며 샘플링 조건이 비교 가능함을 확인", key="comparison_confirm"):
        return None, []
    vals = data["frame"]["__qa_value"].to_numpy(float)
    try:
        bsum = summarize(vals, config["lsl"], config["usl"], config["target"])
    except ValueError as exc:
        st.error(str(exc))
        return None, []
    asum = primary["report"]["summary"]
    fields = ["n", "mean", "std", "pp", "ppk", "oos_count", "oos_rate"]
    st.dataframe(pd.DataFrame({"A": {k: asum[k] for k in fields},
                               "B": {k: bsum[k] for k in fields},
                               "B − A": {k: bsum[k] - asum[k] for k in fields}}), width="stretch")
    fig = go.Figure()
    both = [primary["data"]["frame"]["__qa_value"].to_numpy(float), vals]
    low, high = min(v.min() for v in both), max(v.max() for v in both)
    bin_size = (high-low)/35
    for label, value in zip(["A", "B"], both):
        fig.add_trace(go.Histogram(x=value, name=label, histnorm="probability",
                                  xbins={"start": low, "end": high+bin_size, "size": bin_size}, opacity=.55))
    fig.update_layout(title="A/B 공통 구간 상대도수", barmode="overlay", template="plotly_white",
                      xaxis_title=config["unit"], yaxis_title="비율", height=380)
    draw(fig, "comparison_histogram")
    st.caption("같은 규격·단위의 기술통계 비교입니다. 차이의 원인이나 개선의 인과효과를 입증하지 않습니다.")
    left, right = st.columns(2)
    with left:
        amap = primary["report"]["wafer_map"]
        if amap is not None:
            draw(wafer_plot(amap["points"], config, "A · 선택된 웨이퍼"), "comparison_amap")
        else:
            st.info("A의 계측 좌표 맵을 설정하면 이곳에 함께 표시합니다.")
    with right:
        bmap, mapfig = map_analysis(data, config, prefix="bmap")
    figures = [fig] + ([mapfig] if mapfig is not None else [])
    return {"source": source[1], "measurement_column": data["column"], "unit": bunit,
            "unit_column": data["unit_column"],
            "filters": data["filters"], "quality_counts": data["counts"],
            "filtered_rows": data["filtered_rows"],
            "summary": {k: v for k, v in bsum.items() if k not in ["cp", "cpk"]}, "wafer_map": bmap,
            "confirmed_comparable": True, "_audit": data["audit"]}, figures


def simulate(config, config_valid=True):
    st.info("독립적인 교육용 시뮬레이션입니다. 업로드한 CSV의 산포·좌표·공정 물리를 추정하지 않습니다.")
    cols = st.columns(4)
    base = cols[0].number_input("시뮬레이션 기준값", value=200.0, key="sim_base")
    edge = cols[1].slider("가장자리 효과 (%)", -50, 50, 12, key="sim_edge")
    noise = cols[2].slider("노이즈 크기 (%)", 0, 30, 5, key="sim_noise")
    seed = cols[3].number_input("시뮬레이션 시드", 0, 2147483647, 7, key="sim_seed")
    if not config_valid:
        st.info("공통 좌표·색상 설정을 확인하면 시뮬레이션을 표시합니다.")
        return None
    if not np.isfinite(base):
        st.error("시뮬레이션 기준값은 유한해야 합니다.")
        return None
    r, cx, cy = config["radius"], config["center_x"], config["center_y"]
    x, y = np.meshgrid(np.linspace(-r, r, 81), np.linspace(-r, r, 81))
    ratio = np.hypot(x, y) / r
    with np.errstate(over="ignore", invalid="ignore"):
        values = base + base * (edge / 100) * ratio + np.random.default_rng(seed).normal(
            0, abs(base) * (noise / 100), size=x.shape)
    mask = ratio <= 1
    if not np.isfinite(values[mask]).all():
        st.error("시뮬레이션이 수치 범위를 벗어났습니다. 기준값과 노이즈를 줄이세요.")
        return None
    points = pd.DataFrame({"x": x[mask]+cx, "y": y[mask]+cy,
                           "value": values[mask], "count": 1})
    draw(wafer_plot(points, config, "합성 방사형 패턴 · 실제 계측 맵 아님"), "simulation_chart")
    st.metric("설정된 이론적 가장자리 − 중심", fmt(base * (edge / 100)))
    st.caption("노이즈 표준편차는 |기준값|×노이즈%입니다. 0·음수 기준값도 허용합니다. "
               "반경은 좌표 크기를 정하며 정규화된 패턴 자체를 바꾸지 않습니다.")
    settings = {"base": base, "edge_percent": edge, "noise_percent": noise, "seed": seed,
                "grid": 81, "radius": r, "center": [cx, cy], "synthetic": True}
    st.download_button("합성 좌표 CSV",
                       points.to_csv(index=False).encode("utf-8-sig"),
                       "synthetic_wafer_points.csv", "text/csv", key="download_simulation")
    return settings


def export_bundle(report, audit, figures, comparison_audit=None):
    clean_report = plain(report)
    parts = ["<!doctype html><html lang='ko'><meta charset='utf-8'><title>계측 분석 보고서</title>",
             "<style>body{font-family:Arial,sans-serif;max-width:1100px;margin:30px auto;padding:20px}"
             "pre{white-space:pre-wrap;overflow-wrap:anywhere}table{border-collapse:collapse}"
             "td,th{padding:6px;border:1px solid #ddd}</style><h1>계측 분석 보고서</h1>",
             "<p>자동 품질 승인 문서가 아닙니다. 입력·제외 행·규격·추정법과 통계 가정을 함께 확인하세요.</p>",
             "<h2>입력 및 분석 조건</h2><pre>" + html.escape(json_bytes(clean_report).decode()) + "</pre>"]
    for i, fig in enumerate(figures):
        parts.append(pio.to_html(fig, full_html=False, include_plotlyjs=True if i == 0 else False))
    parts.append("</html>")
    html_report = "\n".join(parts).encode("utf-8")
    stream = BytesIO()
    with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("analysis.json", json_bytes(clean_report))
        archive.writestr("rows_with_quality.csv", audit.to_csv(index=False).encode("utf-8-sig"))
        archive.writestr("analysis_values.csv", audit.loc[audit["__qa_status"].eq("valid")].to_csv(index=False).encode("utf-8-sig"))
        summary_rows = []
        for label, item in [("A", report), ("B", report.get("comparison"))]:
            if item is not None:
                row = {"dataset": label, "source_file": item["source"]["name"],
                       "source_sha256": item["source"]["sha256"],
                       "measurement_column": item["measurement_column"],
                       "unit": report["config"]["unit"], "spec_version": report["config"]["spec_version"],
                       "created_at_utc": report["created_at_utc"], "app_version": APP_VERSION}
                row.update({k: v for k, v in item["summary"].items() if k not in ["cp", "cpk"]})
                summary_rows.append(row)
                if item.get("wafer_map") is not None:
                    archive.writestr(f"{label}_measured_wafer_sites.csv",
                                     item["wafer_map"]["points"].to_csv(index=False).encode("utf-8-sig"))
        archive.writestr("summary.csv", pd.DataFrame(summary_rows).to_csv(index=False).encode("utf-8-sig"))
        if report.get("group_comparison") is not None:
            archive.writestr("group_summary.csv",
                             pd.DataFrame(report["group_comparison"]["results"]).to_csv(index=False).encode("utf-8-sig"))
        if comparison_audit is not None:
            archive.writestr("comparison_rows_with_quality.csv", comparison_audit.to_csv(index=False).encode("utf-8-sig"))
        archive.writestr("report.html", html_report)
    return stream.getvalue(), html_report


def exports(primary, comparison, comparison_figures, simulation):
    history = st.session_state.setdefault("analysis_history", [])
    if primary is not None:
        comparison_report = {k: v for k, v in comparison.items() if k != "_audit"} if comparison else None
        report = dict(primary["report"], comparison=comparison_report, independent_simulation=simulation)
        if st.button("현재 분석을 세션 이력에 저장", key="save_history"):
            history.append(plain(report))
            del history[:-20]
            st.success("분석 조건과 결과를 저장했습니다.")
        if st.button("현재 결과 다운로드 파일 만들기", key="build_export"):
            package, html_report = export_bundle(
                report, primary["data"]["audit"], primary["figures"]+comparison_figures,
                comparison["_audit"] if comparison else None)
            st.session_state["export_ready"] = {
                "zip": package, "html": html_report, "report": json_bytes(report),
                "signature": hashlib.sha256(json_bytes({k: v for k, v in report.items() if k != "created_at_utc"})).hexdigest()}
        signature = hashlib.sha256(json_bytes({k: v for k, v in report.items() if k != "created_at_utc"})).hexdigest()
        saved = st.session_state.get("export_ready")
        if saved and saved["signature"] == signature:
            st.download_button("분석 묶음 ZIP", saved["zip"], "measurement_analysis.zip",
                               "application/zip", key="download_zip")
            st.download_button("그래프 포함 HTML 보고서", saved["html"], "measurement_report.html",
                               "text/html", key="download_html")
            st.download_button("설정·결과 JSON", saved["report"], "analysis.json",
                               "application/json", key="download_json")
        elif saved:
            st.info("분석 조건이 변경되었습니다. 현재 결과 파일을 다시 만드세요.")
    st.caption("이력은 현재 브라우저 세션에서 최근 20개만 유지됩니다. 재접속 전 JSON을 내려받으세요. "
               "파일명·SHA-256·선택 열·단위·규격 버전·제외 건수·추정법·설정이 기록됩니다.")
    if history:
        rows = [{"time": r["created_at_utc"], "file": r["source"]["name"],
                 "column": r["measurement_column"], "unit": r["config"]["unit"],
                 "spec_version": r["config"]["spec_version"], "n": r["summary"]["n"],
                 "Ppk": r["summary"]["ppk"]} for r in history]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.download_button("세션 이력 JSON", json_bytes(history), "analysis_history.json",
                           "application/json", key="download_history")


def main():
    st.set_page_config(page_title="Cpk & Wafer Map Analyzer", layout="wide")
    st.title("계측 데이터 · 공정능력 · 웨이퍼맵")
    st.caption(f"v{APP_VERSION} · 연속형 계측값용 · 전체 표본 Pp/Ppk와 단기 추정 Cp/Cpk를 구분합니다.")
    with st.sidebar:
        st.header("규격과 공통 설정")
        st.caption("초기값은 예시입니다. 실제 적용 규격과 단위를 입력하세요.")
        unit = st.text_input("계측 단위", value="nm", key="unit").strip()
        spec_version = st.text_input("규격 이름 / 버전", value="SPEC-001", key="spec_version").strip()
        lsl = st.number_input("LSL · 하한", value=180.0, key="lsl")
        usl = st.number_input("USL · 상한", value=220.0, key="usl")
        target = st.number_input("지수 목표값", min_value=.01, value=1.33, key="target")
        st.subheader("웨이퍼 좌표 · 공통 색상")
        radius = st.number_input("반경 (mm)", min_value=.001, value=100.0, key="radius")
        cx = st.number_input("중심 X (mm)", value=0.0, key="center_x")
        cy = st.number_input("중심 Y (mm)", value=0.0, key="center_y")
        center_frac = st.slider("중심 영역 최대 r/R", .0, .49, .3, key="center_frac")
        edge_frac = st.slider("가장자리 영역 최소 r/R", .5, 1., .8, key="edge_frac")
        scale_mode = st.radio("공통 색 범위", ["규격 범위", "직접 지정"], key="color_mode")
        cmin, cmax = lsl, usl
        if scale_mode == "직접 지정":
            cmin = st.number_input("색 최솟값", value=180.0, key="color_min")
            cmax = st.number_input("색 최댓값", value=220.0, key="color_max")
    numbers = [lsl, usl, target, radius, cx, cy, cmin, cmax]
    config_valid = bool(unit and spec_version and np.isfinite(numbers).all() and usl > lsl and cmax > cmin)
    if not config_valid:
        st.error("단위·규격 버전을 입력하고 유한한 숫자, USL > LSL, 색 최댓값 > 최솟값을 확인하세요.")
    config = {"unit": unit, "spec_version": spec_version, "lsl": lsl, "usl": usl, "target": target,
              "radius": radius, "center_x": cx, "center_y": cy, "center_frac": center_frac,
              "edge_frac": edge_frac, "color_min": cmin, "color_max": cmax, "color_mode": scale_mode}
    tabs = st.tabs(["계측 분석", "A/B 파일 비교", "패턴 시뮬레이션", "이력·내보내기"])
    primary = None
    with tabs[0]:
        source = csv_input("main")
        if source is not None:
            primary = analyze(source, config, config_valid)
    with tabs[1]:
        comparison, comparison_figures = compare(primary, config)
    with tabs[2]:
        simulation = simulate(config, config_valid)
    with tabs[3]:
        exports(primary, comparison, comparison_figures, simulation)
    with st.expander("계산식과 해석 범위"):
        st.markdown("""
**Pp/Ppk:** 전체 표본 표준편차 s(ddof=1)를 사용합니다.
Pp=(USL−LSL)/(6s), Ppk=min(USL−평균, 평균−LSL)/(3s).
**Cp/Cpk:** 사용자가 확인한 연속 개별 측정의 MR 또는 합리적 부분군의 합동 s를 사용합니다.
높은 지수만으로 시간 안정성·독립성·정규성·측정 적합성·제품 신뢰성이 입증되지 않습니다.
정규성 검정과 관리도의 신호 유무는 참고 진단이며, 전체 규칙이나 현장 승인 절차를 대체하지 않습니다.

근거: [NIST 공정능력](https://www.itl.nist.gov/div898/handbook/pmc/section1/pmc16.htm),
[NIST I 관리도](https://www.itl.nist.gov/div898/handbook/pmc/section3/pmc322.htm),
[SciPy Shapiro](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.shapiro.html).
""")


if __name__ == "__main__":
    main()
