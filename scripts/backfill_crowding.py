from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipeline import BREADTH_PATH, CROWDING_PATH, STATUS_PATH  # noqa: E402
from src.utils import (  # noqa: E402
    date_key,
    ensure_dirs,
    numeric,
    pick_column,
    read_csv_safe,
    write_csv_atomic,
    write_json,
)

A_SHARE_PREFIXES = (
    "000", "001", "002", "003", "300", "301",
    "600", "601", "603", "605", "688", "689",
)


def now_iso() -> str:
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")


def validate_date(value: str, name: str) -> str:
    try:
        return datetime.strptime(value, "%Y%m%d").strftime("%Y%m%d")
    except ValueError as exc:
        raise ValueError(f"{name} 必须是 YYYYMMDD，例如 20250101") from exc


def _codes_from_frame(frame: pd.DataFrame, candidates: list[str]) -> list[str]:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return []
    code_col = pick_column(frame, candidates)
    if code_col is None:
        return []
    codes = (
        frame[code_col]
        .astype(str)
        .str.extract(r"(\d{6})", expand=False)
        .dropna()
        .str.zfill(6)
    )
    codes = codes[codes.str.startswith(A_SHARE_PREFIXES)]
    return sorted(codes.drop_duplicates().tolist())


def _call_with_retry(label: str, fn, attempts: int = 4) -> pd.DataFrame:
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            frame = fn()
            if isinstance(frame, pd.DataFrame) and not frame.empty:
                return frame
            raise RuntimeError("返回空表")
        except Exception as exc:
            last_error = exc
            if attempt < attempts:
                wait_seconds = min(30, 2 ** attempt)
                print(
                    f"{label} 第 {attempt}/{attempts} 次失败，"
                    f"{wait_seconds} 秒后重试：{exc!r}",
                    flush=True,
                )
                time.sleep(wait_seconds)
    raise RuntimeError(f"{label} 连续失败：{last_error!r}")


def _universe_cache_path() -> Path:
    return ROOT / "data" / "a_share_universe.csv"


def _load_cached_universe() -> list[str]:
    path = _universe_cache_path()
    frame = read_csv_safe(path)
    return _codes_from_frame(frame, ["code", "证券代码", "A股代码"])


def _save_universe(codes: list[str], source: str) -> None:
    path = _universe_cache_path()
    rows = pd.DataFrame({
        "code": sorted(set(codes)),
    })
    rows["exchange"] = np.where(
        rows["code"].str.startswith("6"), "SSE", "SZSE"
    )
    rows["source"] = source
    rows["updated_at"] = now_iso()
    write_csv_atomic(rows, path)


def load_a_share_codes() -> tuple[list[str], str]:
    """获取沪深 A 股代码。

    优先读取上交所、深交所官方股票列表；其次尝试 AKShare 的沪深京
    汇总接口；最后使用仓库内上次成功保存的代码快照。这样即使东方财富
    对 GitHub Runner 临时断开连接，历史回填仍能继续。
    """
    import akshare as ak

    minimum_expected = 4000
    errors: list[str] = []
    cached = _load_cached_universe()

    official_specs = [
        (
            "上交所主板A股",
            lambda: ak.stock_info_sh_name_code(symbol="主板A股"),
            ["证券代码", "code"],
        ),
        (
            "上交所科创板",
            lambda: ak.stock_info_sh_name_code(symbol="科创板"),
            ["证券代码", "code"],
        ),
        (
            "深交所A股",
            lambda: ak.stock_info_sz_name_code(symbol="A股列表"),
            ["A股代码", "证券代码", "code"],
        ),
    ]

    official_codes: list[str] = []
    official_success = 0
    for label, fn, candidates in official_specs:
        try:
            frame = _call_with_retry(label, fn)
            codes = _codes_from_frame(frame, candidates)
            if not codes:
                raise RuntimeError("未识别出证券代码字段")
            official_codes.extend(codes)
            official_success += 1
            print(f"{label}：{len(codes)} 只", flush=True)
        except Exception as exc:
            errors.append(f"{label}: {exc!r}")

    official_codes = sorted(set(official_codes))
    if official_success == len(official_specs) and len(official_codes) >= minimum_expected:
        source = "上交所及深交所官方上市股票列表 / AKShare"
        _save_universe(official_codes, source)
        return official_codes, source

    # 如果官方接口只成功一部分，用上次代码快照补齐；避免丢失整个市场。
    official_plus_cache = sorted(set(official_codes).union(cached))
    if len(official_plus_cache) >= minimum_expected:
        source = "交易所官方列表（部分）+ 仓库内A股代码快照"
        _save_universe(official_plus_cache, source)
        return official_plus_cache, source

    combined_specs = [
        (
            "沪深京A股实时列表",
            getattr(ak, "stock_zh_a_spot_em", None),
            ["代码", "code", "symbol", "股票代码"],
        ),
        (
            "A股代码名称表",
            getattr(ak, "stock_info_a_code_name", None),
            ["code", "代码", "symbol", "股票代码"],
        ),
    ]
    for label, fn, candidates in combined_specs:
        if not callable(fn):
            continue
        try:
            frame = _call_with_retry(label, fn)
            codes = _codes_from_frame(frame, candidates)
            if len(codes) < minimum_expected:
                raise RuntimeError(
                    f"仅返回 {len(codes)} 只，低于完整沪深A股列表的安全阈值"
                )
            source = f"{label} / AKShare"
            _save_universe(codes, source)
            return codes, source
        except Exception as exc:
            errors.append(f"{label}: {exc!r}")

    if len(cached) >= minimum_expected:
        return cached, "仓库内上次成功保存的A股代码快照"

    raise RuntimeError(
        "无法取得完整沪深A股代码列表；已尝试交易所官方列表、"
        "AKShare汇总列表和本地快照。详细错误：" + " | ".join(errors)
    )


def normalize_history(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame(columns=["trade_date", "amount", "pct_change"])
    date_col = pick_column(frame, ["日期", "date", "Date", "trade_date"])
    amount_col = pick_column(frame, ["成交额", "amount", "Amount", "成交金额"])
    pct_col = pick_column(frame, ["涨跌幅", "pct_change", "pct_chg", "涨跌幅%"])
    if date_col is None or amount_col is None:
        return pd.DataFrame(columns=["trade_date", "amount", "pct_change"])
    out = pd.DataFrame({
        "trade_date": frame[date_col].map(date_key),
        "amount": numeric(frame[amount_col]),
        "pct_change": numeric(frame[pct_col]) if pct_col else np.nan,
    })
    return out.dropna(subset=["trade_date", "amount"]).query("amount > 0")


def fetch_batch(codes: list[str], start_date: str, end_date: str, attempts: int = 3) -> dict[str, pd.DataFrame]:
    import efinance as ef

    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            result = ef.stock.get_quote_history(
                codes,
                beg=start_date,
                end=end_date,
                klt=101,
                fqt=0,
                suppress_error=True,
            )
            if isinstance(result, pd.DataFrame):
                return {codes[0]: result}
            if isinstance(result, dict):
                return result
            return {}
        except Exception as exc:
            last_error = exc
            if attempt < attempts:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"批量历史行情连续失败：{last_error!r}")


def calculate_history(
    start_date: str,
    end_date: str,
    top_fraction: float,
    batch_size: int,
    pause_seconds: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if not 0 < top_fraction <= 1:
        raise ValueError("top_fraction 必须在 (0, 1] 范围内")
    codes, universe_source = load_a_share_codes()
    amounts_by_date: dict[str, list[float]] = defaultdict(list)
    pct_by_date: dict[str, list[float]] = defaultdict(list)
    successful_symbols = 0
    failed_batches: list[dict[str, Any]] = []
    total_batches = math.ceil(len(codes) / batch_size)
    print(f"A股代码数：{len(codes)}；批次数：{total_batches}")

    for batch_no, start in enumerate(range(0, len(codes), batch_size), start=1):
        batch = codes[start : start + batch_size]
        print(f"[{batch_no}/{total_batches}] 获取 {batch[0]} - {batch[-1]}", flush=True)
        try:
            result = fetch_batch(batch, start_date, end_date)
        except Exception as exc:
            failed_batches.append({"batch": batch_no, "first": batch[0], "last": batch[-1], "error": repr(exc)})
            print(f"  批次失败：{exc!r}", flush=True)
            continue
        batch_success = 0
        for _, frame in result.items():
            clean = normalize_history(frame)
            if clean.empty:
                continue
            batch_success += 1
            for trade_date, group in clean.groupby("trade_date"):
                amounts_by_date[str(trade_date)].extend(group["amount"].astype(float).tolist())
                pct_by_date[str(trade_date)].extend(group["pct_change"].dropna().astype(float).tolist())
        successful_symbols += batch_success
        print(f"  本批有效股票：{batch_success}", flush=True)
        if pause_seconds > 0:
            time.sleep(pause_seconds)

    crowding_rows: list[dict[str, Any]] = []
    breadth_rows: list[dict[str, Any]] = []
    for trade_date in sorted(amounts_by_date):
        amounts = np.asarray(amounts_by_date[trade_date], dtype="float64")
        amounts = amounts[np.isfinite(amounts) & (amounts > 0)]
        stock_count = int(amounts.size)
        if stock_count == 0:
            continue
        top_count = max(1, math.ceil(stock_count * top_fraction))
        split_at = stock_count - top_count
        top_amount = float(np.partition(amounts, split_at)[split_at:].sum())
        total_amount = float(amounts.sum())
        crowding_rows.append({
            "trade_date": trade_date,
            "top_fraction": top_fraction,
            "stock_count": stock_count,
            "top_count": top_count,
            "top_amount_trillion": top_amount / 1e12,
            "total_amount_trillion": total_amount / 1e12,
            "crowding_pct": top_amount / total_amount * 100,
            "source": "东方财富历史行情 / efinance（当前上市A股回溯口径）",
        })
        pct_values = np.asarray(pct_by_date.get(trade_date, []), dtype="float64")
        pct_values = pct_values[np.isfinite(pct_values)]
        breadth_rows.append({
            "trade_date": trade_date,
            "up_count": int((pct_values > 0).sum()),
            "down_count": int((pct_values < 0).sum()),
            "flat_count": int((pct_values == 0).sum()),
            "total_count": int(pct_values.size),
            "total_amount_trillion": total_amount / 1e12,
            "total_market_cap_trillion": np.nan,
            "broad_turnover_pct": np.nan,
            "source": "东方财富历史行情 / efinance（当前上市A股回溯口径）",
        })

    if not crowding_rows:
        raise RuntimeError("没有生成任何历史交易拥挤度记录")
    metadata = {
        "requested_symbols": len(codes),
        "universe_source": universe_source,
        "successful_symbols": successful_symbols,
        "failed_batches": failed_batches,
        "start_date": start_date,
        "end_date": end_date,
        "generated_rows": len(crowding_rows),
        "universe_note": "历史回填使用运行日仍上市的A股代码，退市股票可能未纳入；属于无Token公开数据条件下的近似点时股票池。",
    }
    return (
        pd.DataFrame(crowding_rows).sort_values("trade_date").reset_index(drop=True),
        pd.DataFrame(breadth_rows).sort_values("trade_date").reset_index(drop=True),
        metadata,
    )


def merge_file(path: Path, new: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    old = read_csv_safe(path)
    merged = new.copy() if old.empty else pd.concat([old, new], ignore_index=True)
    merged = merged.drop_duplicates(keys, keep="last").sort_values(keys).reset_index(drop=True)
    write_csv_atomic(merged, path)
    return merged


def save(crowding: pd.DataFrame, breadth: pd.DataFrame, metadata: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    crowding_all = merge_file(CROWDING_PATH, crowding, ["trade_date"])
    breadth_all = merge_file(BREADTH_PATH, breadth, ["trade_date"])
    status: dict[str, Any] = {}
    if STATUS_PATH.exists():
        try:
            loaded = json.loads(STATUS_PATH.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                status = loaded
        except Exception:
            status = {}
    status.setdefault("datasets", {})
    status["updated_at"] = now_iso()
    status["datasets"]["crowding"] = {
        "status": "success", "rows": len(crowding), "cached_rows": len(crowding_all),
        "latest_date": str(crowding_all["trade_date"].max()), "backfill": metadata,
    }
    status["datasets"]["breadth"] = {
        "status": "success", "rows": len(breadth), "cached_rows": len(breadth_all),
        "latest_date": str(breadth_all["trade_date"].max()), "backfill": metadata,
    }
    write_json(STATUS_PATH, status)
    return crowding_all, breadth_all


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="回填A股交易拥挤度与涨跌家数历史")
    parser.add_argument("--start", required=True, help="开始日期 YYYYMMDD")
    parser.add_argument("--end", default="", help="结束日期 YYYYMMDD；留空为今天")
    parser.add_argument("--top-fraction", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--pause-seconds", type=float, default=0.6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()
    start_date = validate_date(args.start, "start")
    end_date = validate_date(args.end or datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d"), "end")
    if start_date > end_date:
        raise ValueError("start 不能晚于 end")
    crowding, breadth, metadata = calculate_history(
        start_date, end_date, args.top_fraction, args.batch_size, args.pause_seconds
    )
    crowding_all, breadth_all = save(crowding, breadth, metadata)
    print(json.dumps({
        "status": "success",
        "new_crowding_rows": len(crowding),
        "new_breadth_rows": len(breadth),
        "total_crowding_rows": len(crowding_all),
        "total_breadth_rows": len(breadth_all),
        **metadata,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
