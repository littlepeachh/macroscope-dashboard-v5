from __future__ import annotations

import io
import math
import re
from datetime import datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from tenacity import retry, stop_after_attempt, wait_exponential

from src.providers import DataFetchError, _import_akshare
from src.utils import date_key, numeric, pick_column

A_SHARE_PREFIXES = (
    "000", "001", "002", "003", "300", "301",
    "600", "601", "603", "605", "688", "689",
)


def _latest_trade_date(ak: Any) -> str:
    today = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d")
    try:
        calendar = ak.tool_trade_date_hist_sina()
        date_col = pick_column(calendar, ["trade_date", "日期", "date"])
        if date_col:
            dates = calendar[date_col].map(date_key).dropna()
            eligible = dates[dates <= today]
            if not eligible.empty:
                return str(eligible.max())
    except Exception:
        pass
    return today


def _clean_a_spot(spot: pd.DataFrame) -> pd.DataFrame:
    code_col = pick_column(spot, ["代码", "symbol", "股票代码"])
    amount_col = pick_column(spot, ["成交额", "amount", "成交金额"])
    pct_col = pick_column(spot, ["涨跌幅", "pct_change", "涨跌幅%"])
    cap_col = pick_column(spot, ["总市值", "market_cap", "总市值(元)"])
    if code_col is None or amount_col is None:
        raise DataFetchError(f"A股实时行情缺少代码或成交额字段: {list(spot.columns)}")
    clean = pd.DataFrame({
        "code": spot[code_col].astype(str).str.extract(r"(\d{6})", expand=False).str.zfill(6),
        "amount": numeric(spot[amount_col]),
        "pct_change": numeric(spot[pct_col]) if pct_col else np.nan,
        "market_cap": numeric(spot[cap_col]) if cap_col else np.nan,
    })
    clean = clean[clean["code"].str.startswith(A_SHARE_PREFIXES, na=False)]
    clean = clean.drop_duplicates("code", keep="last")
    return clean


class GlobalMarketProvider:
    """Public end-of-day market data via Yahoo Finance/yfinance."""

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10), reraise=True)
    def fetch(self, items: list[dict[str, Any]], start_date: str) -> pd.DataFrame:
        try:
            import yfinance as yf
        except Exception as exc:  # pragma: no cover
            raise DataFetchError(f"yfinance 无法加载: {exc}") from exc

        tickers = [str(item["ticker"]) for item in items]
        metadata = {str(item["ticker"]): item for item in items}
        raw = yf.download(
            tickers=tickers,
            start=pd.to_datetime(start_date).strftime("%Y-%m-%d"),
            auto_adjust=True,
            group_by="ticker",
            progress=False,
            threads=False,
            timeout=45,
        )
        if raw is None or raw.empty:
            raise DataFetchError("Yahoo Finance 返回空行情")

        rows: list[pd.DataFrame] = []
        if isinstance(raw.columns, pd.MultiIndex):
            available = set(str(x) for x in raw.columns.get_level_values(0))
            for ticker in tickers:
                if ticker not in available:
                    continue
                frame = raw[ticker].reset_index().copy()
                frame["symbol"] = ticker
                rows.append(frame)
        else:
            frame = raw.reset_index().copy()
            frame["symbol"] = tickers[0]
            rows.append(frame)

        if not rows:
            raise DataFetchError("Yahoo Finance 没有返回任何指定资产")

        out = pd.concat(rows, ignore_index=True).rename(
            columns={"Date": "trade_date", "Close": "close", "Volume": "volume"}
        )
        out["trade_date"] = out["trade_date"].map(date_key)
        out["close"] = numeric(out.get("close", pd.Series(index=out.index, dtype=float)))
        out["volume"] = numeric(out.get("volume", pd.Series(index=out.index, dtype=float)))
        out["name"] = out["symbol"].map(lambda x: metadata.get(str(x), {}).get("chinese_name") or metadata.get(str(x), {}).get("name"))
        out["market"] = out["symbol"].map(lambda x: metadata.get(str(x), {}).get("market", "GLOBAL"))
        out["currency"] = out["symbol"].map(lambda x: metadata.get(str(x), {}).get("currency", ""))
        out["asset_group"] = out["symbol"].map(lambda x: metadata.get(str(x), {}).get("group", "全球市场"))
        out = out.dropna(subset=["trade_date", "close"]).sort_values(["symbol", "trade_date"])
        out["pct_change"] = out.groupby("symbol")["close"].pct_change() * 100
        out["amount"] = np.nan
        out["source"] = "Yahoo Finance / yfinance"
        return out[[
            "trade_date", "symbol", "name", "market", "currency", "asset_group",
            "close", "pct_change", "volume", "amount", "source",
        ]].reset_index(drop=True)


class FredTreasuryProvider:
    """Federal Reserve H.15 Treasury yields exposed through FRED CSV."""

    CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS2,DGS10"

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10), reraise=True)
    def fetch(self, start_date: str) -> pd.DataFrame:
        response = requests.get(
            self.CSV_URL,
            timeout=40,
            headers={"User-Agent": "Mozilla/5.0 (compatible; MacroScopePublic/5.0; research dashboard)"},
        )
        response.raise_for_status()
        raw = pd.read_csv(io.StringIO(response.text))
        date_col = pick_column(raw, ["DATE", "observation_date", "date"])
        if date_col is None:
            raise DataFetchError(f"FRED CSV 缺少日期字段: {list(raw.columns)}")
        frames: list[pd.DataFrame] = []
        names = {"DGS2": "美国2年期国债收益率", "DGS10": "美国10年期国债收益率"}
        for series, name in names.items():
            if series not in raw.columns:
                continue
            frame = pd.DataFrame({
                "trade_date": raw[date_col].map(date_key),
                "series": series,
                "name": name,
                "value_pct": numeric(raw[series]),
                "unit": "%",
                "source": "美联储H.15 / FRED",
            })
            frame = frame.dropna(subset=["trade_date", "value_pct"])
            frame = frame[frame["trade_date"] >= start_date]
            frames.append(frame)
        if not frames:
            raise DataFetchError("FRED 未返回DGS2/DGS10")
        return pd.concat(frames, ignore_index=True).sort_values(["series", "trade_date"])


class ChinaLiquidityProvider:
    """Official/official-adapter money-market rates. DR is never replaced with FDR."""

    CHINAMONEY_URL = "https://www.chinamoney.com.cn/chinese"

    def __init__(self) -> None:
        self.ak = _import_akshare()

    def fetch_shibor_overnight(self) -> pd.DataFrame:
        frame = self.ak.rate_interbank(
            market="上海银行同业拆借市场",
            symbol="Shibor人民币",
            indicator="隔夜",
        )
        if frame is None or frame.empty:
            raise DataFetchError("隔夜Shibor返回空表")
        date_col = pick_column(frame, ["报告日", "日期", "date"])
        value_col = pick_column(frame, ["利率", "今值", "value"])
        if date_col is None or value_col is None:
            raise DataFetchError(f"隔夜Shibor缺少字段: {list(frame.columns)}")
        out = pd.DataFrame({
            "trade_date": frame[date_col].map(date_key),
            "shibor_on_pct": numeric(frame[value_col]),
        })
        out = out.dropna(subset=["trade_date", "shibor_on_pct"]).drop_duplicates("trade_date", keep="last")
        return out.sort_values("trade_date")

    @staticmethod
    def _extract_rate(text: str, code: str) -> float | None:
        patterns = [
            rf"\b{re.escape(code)}\b\s*[:：]?\s*([0-9]+(?:\.[0-9]+)?)",
            rf"\b{re.escape(code)}\b.{{0,80}}?([0-9]+(?:\.[0-9]+)?)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.I | re.S)
            if match:
                value = float(match.group(1))
                if 0 <= value <= 30:
                    return value
        return None

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8), reraise=True)
    def fetch_dr_current(self) -> pd.DataFrame:
        response = requests.get(
            self.CHINAMONEY_URL,
            timeout=35,
            headers={"User-Agent": "Mozilla/5.0 (compatible; MacroScopePublic/5.0; research dashboard)"},
        )
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        text = " ".join(soup.stripped_strings)
        dr001 = self._extract_rate(text, "DR001")
        dr007 = self._extract_rate(text, "DR007")
        if dr001 is None and dr007 is None:
            raise DataFetchError("中国货币网页面未解析到DR001/DR007；将保留历史缓存，不用FDR替代")
        trade_date = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d")
        return pd.DataFrame([{
            "trade_date": trade_date,
            "dr001_pct": dr001,
            "dr007_pct": dr007,
            "source": "中国货币网 / 全国银行间同业拆借中心",
        }])

    def fetch(self) -> tuple[pd.DataFrame, dict[str, Any]]:
        details: dict[str, Any] = {}
        shibor = pd.DataFrame()
        dr = pd.DataFrame()
        try:
            shibor = self.fetch_shibor_overnight()
            details["shibor_on"] = {"status": "success", "rows": len(shibor), "source": "中国货币网 / AKShare"}
        except Exception as exc:
            details["shibor_on"] = {"status": "failed", "error": repr(exc)}
        try:
            dr = self.fetch_dr_current()
            details["dr"] = {"status": "success", "rows": len(dr), "source": "中国货币网"}
        except Exception as exc:
            details["dr"] = {"status": "failed", "error": repr(exc), "note": "严格不使用FDR替代DR"}
        if shibor.empty and dr.empty:
            raise DataFetchError("DR与Shibor数据源均失败")
        if shibor.empty:
            out = dr.copy()
            out["shibor_on_pct"] = np.nan
        elif dr.empty:
            out = shibor.copy()
            out["dr001_pct"] = np.nan
            out["dr007_pct"] = np.nan
            out["source"] = "中国货币网 / AKShare"
        else:
            out = shibor.merge(dr, on="trade_date", how="outer", suffixes=("_shibor", "_dr"))
            out["source"] = out.get("source_dr").fillna(out.get("source_shibor"))
            out = out.drop(columns=[x for x in ["source_dr", "source_shibor"] if x in out.columns])
        return out.sort_values("trade_date"), details


class ChinaSentimentProvider:
    """A-share breadth, crowding, broad turnover, and margin leverage."""

    def __init__(self) -> None:
        self.ak = _import_akshare()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=12), reraise=True)
    def _call_frame(self, fn: Any, **kwargs: Any) -> pd.DataFrame:
        frame = fn(**kwargs)
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            raise DataFetchError("数据源返回空表")
        return frame

    def _split_spot_em(self) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        for fn_name in ["stock_sh_a_spot_em", "stock_sz_a_spot_em"]:
            fn = getattr(self.ak, fn_name, None)
            if not callable(fn):
                continue
            try:
                frame = self._call_frame(fn)
                frames.append(frame)
            except Exception:
                continue
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def _spot(self) -> tuple[pd.DataFrame, str]:
        """Return one all-A-share snapshot with Eastmoney, Sina and split-market fallbacks."""
        errors: list[str] = []
        candidates: list[tuple[str, Any]] = [
            ("东方财富沪深京A股实时行情", getattr(self.ak, "stock_zh_a_spot_em", None)),
            ("新浪沪深京A股实时行情", getattr(self.ak, "stock_zh_a_spot", None)),
            ("东方财富沪深A股分市场行情", self._split_spot_em),
        ]
        for source, fn in candidates:
            if not callable(fn):
                errors.append(f"{source}: 当前AKShare版本无此接口")
                continue
            try:
                frame = self._call_frame(fn)
                clean = _clean_a_spot(frame)
                valid = clean[clean["amount"].notna() & (clean["amount"] > 0)]
                if len(valid) < 1000:
                    errors.append(f"{source}: 有效A股仅{len(valid)}只")
                    continue
                return clean, source + " / AKShare"
            except Exception as exc:
                errors.append(f"{source}: {exc!r}")
        raise DataFetchError("A股实时行情失败: " + " | ".join(errors))

    @staticmethod
    def _to_yuan(value: float | int | None, *, default_unit: str) -> float | None:
        if value is None or not np.isfinite(float(value)):
            return None
        number = float(value)
        if default_unit == "100m":
            return number * 1e8
        return number

    def _official_market_cap(self, trade_date: str) -> tuple[float | None, str]:
        """Get沪深股票总市值 from exchange market-summary tables as fallback."""
        errors: list[str] = []
        sse_cap_yuan: float | None = None
        szse_cap_yuan: float | None = None

        sse_fn = getattr(self.ak, "stock_sse_summary", None)
        if callable(sse_fn):
            try:
                sse = self._call_frame(sse_fn)
                item_col = pick_column(sse, ["项目", "item"])
                stock_col = pick_column(sse, ["股票", "stock"])
                if item_col and stock_col:
                    row = sse[sse[item_col].astype(str).str.replace(r"\s+", "", regex=True) == "总市值"]
                    if not row.empty:
                        raw = numeric(row[stock_col]).dropna()
                        if not raw.empty:
                            # 上交所股票数据总貌的总市值字段为亿元。
                            sse_cap_yuan = float(raw.iloc[-1]) * 1e8
            except Exception as exc:
                errors.append(f"上交所市场总貌: {exc!r}")

        sz_fn = getattr(self.ak, "stock_szse_summary", None)
        if callable(sz_fn):
            candidate_dates = pd.bdate_range(
                end=pd.to_datetime(trade_date), periods=8
            ).strftime("%Y%m%d").tolist()[::-1]
            for candidate in candidate_dates:
                try:
                    sz = self._call_frame(sz_fn, date=candidate)
                    category_col = pick_column(sz, ["证券类别", "类别"])
                    cap_col = pick_column(sz, ["总市值", "market_cap"])
                    if category_col and cap_col:
                        categories = sz[category_col].astype(str)
                        a_rows = sz[categories.str.contains("A股", na=False)]
                        target = a_rows if not a_rows.empty else sz[categories == "股票"]
                        values = numeric(target[cap_col]).dropna()
                        if not values.empty:
                            # 深交所市场总貌总市值字段为元。
                            szse_cap_yuan = float(values.sum())
                            break
                except Exception as exc:
                    errors.append(f"深交所市场总貌{candidate}: {exc!r}")

        parts = [x for x in [sse_cap_yuan, szse_cap_yuan] if x is not None and np.isfinite(x)]
        if not parts:
            return None, "；".join(errors[-4:])
        return float(sum(parts)), "上交所、深交所市场总貌 / AKShare"

    def fetch_snapshot(self, top_fraction: float) -> tuple[dict[str, Any], dict[str, Any]]:
        if not 0 < top_fraction <= 1:
            raise ValueError("top_fraction必须在(0,1]内")
        clean, source = self._spot()
        amount_clean = clean[clean["amount"].notna() & (clean["amount"] > 0)].copy()
        if amount_clean.empty:
            raise DataFetchError("没有有效A股成交额")
        total_amount = float(amount_clean["amount"].sum())
        top_count = max(1, math.ceil(len(amount_clean) * top_fraction))
        top_amount = float(amount_clean.nlargest(top_count, "amount")["amount"].sum())
        valid_pct = clean[clean["pct_change"].notna()].copy()
        up_count = int((valid_pct["pct_change"] > 0).sum())
        down_count = int((valid_pct["pct_change"] < 0).sum())
        flat_count = int((valid_pct["pct_change"] == 0).sum())
        trade_date = _latest_trade_date(self.ak)

        if clean["market_cap"].notna().any():
            total_market_cap = float(clean["market_cap"].dropna().sum())
            cap_source = source
        else:
            total_market_cap, cap_source = self._official_market_cap(trade_date)
            total_market_cap = float(total_market_cap) if total_market_cap is not None else np.nan

        crowding = {
            "trade_date": trade_date,
            "top_fraction": top_fraction,
            "stock_count": int(len(amount_clean)),
            "top_count": int(top_count),
            "top_amount_trillion": top_amount / 1e12,
            "total_amount_trillion": total_amount / 1e12,
            "crowding_pct": top_amount / total_amount * 100 if total_amount else np.nan,
            "source": source,
        }
        breadth = {
            "trade_date": trade_date,
            "up_count": up_count,
            "down_count": down_count,
            "flat_count": flat_count,
            "total_count": int(len(valid_pct)),
            "total_amount_trillion": total_amount / 1e12,
            "total_market_cap_trillion": total_market_cap / 1e12 if np.isfinite(total_market_cap) else np.nan,
            "broad_turnover_pct": total_amount / total_market_cap * 100 if np.isfinite(total_market_cap) and total_market_cap > 0 else np.nan,
            "source": source,
            "market_cap_source": cap_source,
        }
        return crowding, breadth

    @staticmethod
    def _extract_margin_total(frame: pd.DataFrame) -> float | None:
        if frame is None or frame.empty:
            return None
        col = pick_column(frame, ["融资融券余额", "融资余额合计", "融资融券余额(元)"], contains=["融资", "融券", "余额"])
        if col is None:
            return None
        values = numeric(frame[col]).dropna()
        if values.empty:
            return None
        text = frame.astype(str).agg(" ".join, axis=1)
        total_mask = text.str.contains("合计|总计", regex=True, na=False)
        if total_mask.any():
            totals = numeric(frame.loc[total_mask, col]).dropna()
            if not totals.empty:
                return float(totals.iloc[-1])
        if len(values) == 1:
            return float(values.iloc[0])
        return float(values.sum())

    @staticmethod
    def _latest_margin_from_history(frame: pd.DataFrame) -> tuple[str, float] | None:
        if frame is None or frame.empty:
            return None
        date_col = pick_column(frame, ["日期", "信用交易日期", "trade_date"])
        value_col = pick_column(frame, ["融资融券余额", "融资融券余额(元)"], contains=["融资", "融券", "余额"])
        if date_col is None or value_col is None:
            return None
        clean = pd.DataFrame({
            "trade_date": frame[date_col].map(date_key),
            "value": numeric(frame[value_col]),
        }).dropna(subset=["trade_date", "value"]).sort_values("trade_date")
        if clean.empty:
            return None
        row = clean.iloc[-1]
        return str(row["trade_date"]), float(row["value"])

    def fetch_margin(self, total_market_cap_trillion: float | None) -> dict[str, Any]:
        """Fetch沪深两融余额 with correct exchange-specific units and historical fallbacks."""
        target_date = _latest_trade_date(self.ak)
        start = (datetime.strptime(target_date, "%Y%m%d") - timedelta(days=25)).strftime("%Y%m%d")
        errors: list[str] = []
        parts_yuan: list[tuple[str, float, str]] = []

        # SSE official endpoint: RMB yuan.
        try:
            sse = self.ak.stock_margin_sse(start_date=start, end_date=target_date)
            date_col = pick_column(sse, ["信用交易日期", "日期", "trade_date"])
            if date_col:
                sse = sse.assign(_date=sse[date_col].map(date_key)).dropna(subset=["_date"]).sort_values("_date")
                latest_date = str(sse["_date"].max())
                sse = sse[sse["_date"] == latest_date]
            else:
                latest_date = target_date
            value = self._extract_margin_total(sse)
            if value is not None:
                parts_yuan.append((latest_date, float(value), "上交所融资融券汇总"))
        except Exception as exc:
            errors.append(f"上交所官方: {exc!r}")

        # SZSE official endpoint: 100 million RMB.
        candidate_dates = pd.bdate_range(end=pd.to_datetime(target_date), periods=10).strftime("%Y%m%d").tolist()[::-1]
        for candidate in candidate_dates:
            try:
                szse = self.ak.stock_margin_szse(date=candidate)
                value = self._extract_margin_total(szse)
                if value is not None:
                    parts_yuan.append((candidate, float(value) * 1e8, "深交所融资融券汇总"))
                    break
            except Exception as exc:
                errors.append(f"深交所官方{candidate}: {exc!r}")

        # Historical public adapters are used only when an exchange part is missing.
        have_sse = any("上交所" in x[2] for x in parts_yuan)
        have_szse = any("深交所" in x[2] for x in parts_yuan)
        if not have_sse:
            try:
                result = self._latest_margin_from_history(self.ak.macro_china_market_margin_sh())
                if result:
                    date_, value = result
                    parts_yuan.append((date_, value, "上海两融历史公开表"))
            except Exception as exc:
                errors.append(f"上海两融备用: {exc!r}")
        if not have_szse:
            try:
                result = self._latest_margin_from_history(self.ak.macro_china_market_margin_sz())
                if result:
                    date_, value = result
                    parts_yuan.append((date_, value, "深圳两融历史公开表"))
            except Exception as exc:
                errors.append(f"深圳两融备用: {exc!r}")

        if not parts_yuan:
            raise DataFetchError("沪深两融余额失败: " + " | ".join(errors[-8:]))

        total_yuan = float(sum(x[1] for x in parts_yuan))
        trade_date = min(x[0] for x in parts_yuan)
        cap = float(total_market_cap_trillion) if total_market_cap_trillion is not None else np.nan
        return {
            "trade_date": trade_date,
            "margin_balance_trillion": total_yuan / 1e12,
            "total_market_cap_trillion": cap,
            "margin_to_market_cap_pct": (total_yuan / 1e12) / cap * 100 if np.isfinite(cap) and cap > 0 else np.nan,
            "source": " + ".join(x[2] for x in parts_yuan),
            "note": "上交所按元；深交所汇总接口按亿元转换；备用历史表按元。",
        }


class FundSubscriptionProvider:
    """Newly established public-fund raised shares; no fake daily net subscription amount."""

    def __init__(self) -> None:
        self.ak = _import_akshare()

    def fetch(self, max_rows: int = 300) -> pd.DataFrame:
        frame = self.ak.fund_new_found_em()
        if frame is None or frame.empty:
            raise DataFetchError("新成立基金返回空表")
        code_col = pick_column(frame, ["基金代码"])
        name_col = pick_column(frame, ["基金简称"])
        type_col = pick_column(frame, ["基金类型"])
        shares_col = pick_column(frame, ["募集份额"])
        date_col = pick_column(frame, ["成立日期"])
        company_col = pick_column(frame, ["发行公司"])
        if code_col is None or name_col is None or shares_col is None or date_col is None:
            raise DataFetchError(f"新成立基金缺少字段: {list(frame.columns)}")
        out = pd.DataFrame({
            "founded_date": frame[date_col].map(date_key),
            "fund_code": frame[code_col].astype(str),
            "fund_name": frame[name_col].astype(str),
            "fund_type": frame[type_col].astype(str) if type_col else "",
            "fund_company": frame[company_col].astype(str) if company_col else "",
            "raised_shares_100m": numeric(frame[shares_col]),
        })
        out["estimated_raised_amount_100m"] = out["raised_shares_100m"]
        out["source"] = "天天基金新成立基金 / AKShare"
        out["method_note"] = "募集份额单位为亿份；按常见初始面值1元/份近似估算募集规模，非存续期每日净申购额"
        return out.dropna(subset=["founded_date", "raised_shares_100m"]).sort_values("founded_date").tail(max_rows).reset_index(drop=True)
