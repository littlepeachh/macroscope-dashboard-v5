from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from jinja2 import Template

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils import DATA_DIR, PUBLIC_DIR, dataframe_to_records, ensure_dirs, load_settings, read_csv_safe  # noqa: E402


def read_status() -> dict[str, Any]:
    path = DATA_DIR / "status.json"
    if not path.exists():
        return {"overall_status": "empty", "datasets": {}}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {"overall_status": "empty", "datasets": {}}
    except Exception:
        return {"overall_status": "empty", "datasets": {}}


def latest_market_cards(market: pd.DataFrame) -> list[dict[str, Any]]:
    cards: list[dict[str, Any]] = []
    if market.empty:
        return cards
    for symbol, group in market.groupby("symbol"):
        group = group.sort_values("trade_date").copy()
        group["close"] = pd.to_numeric(group["close"], errors="coerce")
        group = group.dropna(subset=["close"])
        if group.empty:
            continue
        last = group.iloc[-1]
        current_year = str(last["trade_date"])[:4]
        year_group = group[group["trade_date"].astype(str).str.startswith(current_year)]
        ytd_base = float(year_group.iloc[0]["close"]) if not year_group.empty else np.nan
        cutoff = pd.to_datetime(str(last["trade_date"]), format="%Y%m%d", errors="coerce") - pd.Timedelta(days=365)
        dates = pd.to_datetime(group["trade_date"].astype(str), format="%Y%m%d", errors="coerce")
        year_ago_group = group[dates >= cutoff] if pd.notna(cutoff) else group
        one_year_base = float(year_ago_group.iloc[0]["close"]) if not year_ago_group.empty else np.nan
        daily_pct = float(last.get("pct_change")) if pd.notna(last.get("pct_change")) else (
            (float(last["close"]) / float(group.iloc[-2]["close"]) - 1) * 100 if len(group) > 1 else np.nan
        )
        cards.append({
            "symbol": str(symbol),
            "name": str(last.get("name", symbol)),
            "market": str(last.get("market", "")),
            "currency": str(last.get("currency", "")) if pd.notna(last.get("currency")) else "",
            "asset_group": str(last.get("asset_group", "")) if pd.notna(last.get("asset_group")) else "",
            "trade_date": str(last["trade_date"]),
            "close": float(last["close"]),
            "daily_pct": daily_pct,
            "ytd_pct": (float(last["close"]) / ytd_base - 1) * 100 if np.isfinite(ytd_base) and ytd_base else np.nan,
            "one_year_pct": (float(last["close"]) / one_year_base - 1) * 100 if np.isfinite(one_year_base) and one_year_base else np.nan,
            "source": str(last.get("source", "")),
        })
    order = {"US_INDEX": 0, "FX_INDEX": 1, "CN_INDEX": 2, "US_EQUITY": 3, "KR_EQUITY": 4}
    return sorted(cards, key=lambda x: (order.get(x["market"], 9), x["name"]))


def valuation_summary(frame: pd.DataFrame) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    if frame.empty:
        return output
    for code, group in frame.groupby("index_code"):
        group = group.sort_values("trade_date").copy()
        values = pd.to_numeric(group["pe_ttm"], errors="coerce").dropna()
        if values.empty:
            continue
        current = float(values.iloc[-1])
        output.append({
            "index_code": str(code),
            "index_name": str(group.iloc[-1]["index_name"]),
            "trade_date": str(group.iloc[-1]["trade_date"]),
            "current_pe": current,
            "pe_mean": float(values.mean()),
            "pe_median": float(values.median()),
            "pe_q25": float(values.quantile(.25)),
            "pe_q75": float(values.quantile(.75)),
            "pe_percentile": float((values <= current).mean() * 100),
            "source": str(group.iloc[-1].get("source", "")),
        })
    return output


HTML = r'''<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{{ title }}</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
:root{--bg:#f3f6fb;--panel:#fff;--ink:#172033;--muted:#6d7890;--line:#e4e9f2;--blue:#3167e3;--navy:#172c58;--cyan:#1e9ca5;--purple:#7456d8;--amber:#c98a18;--up:#d64242;--down:#159567;--shadow:0 14px 36px rgba(25,42,80,.08)}
*{box-sizing:border-box}body{margin:0;background:linear-gradient(180deg,#eef3fb 0,#f7f9fc 320px);font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif;color:var(--ink)}
.wrap{max-width:1500px;margin:0 auto;padding:24px}.hero{background:radial-gradient(circle at 80% 0,#345ea8 0,transparent 30%),linear-gradient(135deg,#102243,#1c3b72 62%,#235b83);color:white;border-radius:24px;padding:28px 30px;box-shadow:var(--shadow);position:relative;overflow:hidden}.hero h1{margin:0;font-size:30px;letter-spacing:.5px}.hero p{margin:8px 0 0;color:#cbd8ee}.hero-meta{display:flex;gap:10px;flex-wrap:wrap;margin-top:18px}.badge{padding:7px 11px;border:1px solid rgba(255,255,255,.18);background:rgba(255,255,255,.09);border-radius:999px;font-size:12px;color:#e8eef9}
.tabs{display:flex;gap:8px;overflow:auto;padding:18px 0 12px}.tab{border:0;background:#e8edf6;color:#55627a;padding:10px 15px;border-radius:11px;font-weight:700;cursor:pointer;white-space:nowrap}.tab.active{background:var(--navy);color:#fff}.panel{display:none}.panel.active{display:block}.grid{display:grid;gap:16px}.g2{grid-template-columns:repeat(2,minmax(0,1fr))}.g3{grid-template-columns:repeat(3,minmax(0,1fr))}.g4{grid-template-columns:repeat(4,minmax(0,1fr))}.kpis{grid-template-columns:repeat(6,minmax(0,1fr));margin-bottom:16px}.card{background:var(--panel);border:1px solid var(--line);border-radius:18px;box-shadow:var(--shadow);padding:17px}.card h3{margin:0 0 12px;font-size:16px}.card h3 small{font-weight:400;color:var(--muted);margin-left:6px}.chart{height:380px}.chart.tall{height:470px}.kpi{background:var(--panel);border:1px solid var(--line);border-radius:16px;padding:15px;box-shadow:var(--shadow);min-height:112px}.kpi-label{font-size:12px;color:var(--muted);font-weight:700}.kpi-value{font-size:25px;font-weight:800;margin-top:8px;line-height:1}.kpi-note{font-size:11px;color:#8a94a8;margin-top:10px}.positive,.up{color:var(--up)!important}.negative,.down{color:var(--down)!important}.neutral{color:var(--muted)!important}.asset-grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}.asset{border:1px solid var(--line);border-radius:15px;padding:14px;background:linear-gradient(180deg,#fff,#fbfcff)}.asset-top{display:flex;justify-content:space-between;gap:8px}.asset-name{font-weight:800;font-size:14px}.asset-symbol{font-size:11px;color:var(--muted);margin-top:3px}.asset-price{font-size:24px;font-weight:800;margin:12px 0}.asset-metrics{display:grid;grid-template-columns:repeat(3,1fr);gap:6px}.asset-metrics div{background:#f4f6fa;border-radius:9px;padding:7px}.asset-metrics span{display:block;font-size:10px;color:var(--muted)}.asset-metrics b{font-size:12px}.hint{background:#f6f8fc;border:1px solid var(--line);border-radius:12px;padding:11px 13px;color:#66738a;font-size:12px;line-height:1.6;margin:12px 0}.valuation-grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}.valuation-card{border:1px solid var(--line);border-radius:14px;padding:13px;cursor:pointer}.valuation-card.selected{border-color:var(--blue);box-shadow:0 0 0 2px rgba(49,103,227,.10)}.multiple{font-size:25px;font-weight:800;margin:9px 0}.stat-row{display:grid;grid-template-columns:repeat(3,1fr);gap:5px}.stat{background:#f5f7fb;border-radius:7px;padding:6px;font-size:10px;color:var(--muted)}.stat b{display:block;color:var(--ink);font-size:11px;margin-top:2px}.bar{height:5px;background:#edf0f5;border-radius:5px;margin-top:10px;overflow:hidden}.bar i{display:block;height:100%;background:linear-gradient(90deg,var(--down),var(--amber),var(--up))}.table-wrap{overflow:auto;border:1px solid var(--line);border-radius:13px}table{width:100%;border-collapse:collapse;font-size:12px;min-width:760px}th{background:#f3f6fb;color:#59667d;text-align:left;padding:10px;position:sticky;top:0}td{border-top:1px solid var(--line);padding:9px 10px}tr:hover td{background:#fafcff}.source-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}.source-card{border:1px solid var(--line);border-radius:14px;padding:13px;background:#fff}.source-status{font-weight:800}.source-status.success,.source-status.cached{color:var(--down)}.source-status.failed{color:var(--up)}.source-status.partial{color:var(--amber)}.empty{display:none;padding:28px;text-align:center;color:var(--muted)}.footer{padding:25px 0;color:var(--muted);font-size:11px;line-height:1.7}
@media(max-width:1100px){.kpis{grid-template-columns:repeat(3,1fr)}.asset-grid,.valuation-grid{grid-template-columns:repeat(2,1fr)}.g2,.g3,.g4{grid-template-columns:1fr}.source-grid{grid-template-columns:1fr 1fr}}
@media(max-width:650px){.wrap{padding:12px}.hero{padding:22px 18px;border-radius:18px}.hero h1{font-size:23px}.kpis{grid-template-columns:repeat(2,1fr)}.asset-grid,.valuation-grid,.source-grid{grid-template-columns:1fr}.chart{height:330px}}
</style>
</head>
<body><div class="wrap">
<section class="hero"><h1>{{ title }}</h1><p>真实公开数据 · GitHub Pages公开访问 · 无需用户Token</p><div class="hero-meta"><span class="badge">版本 {{ version }}</span><span class="badge" id="statusBadge">状态读取中</span><span class="badge" id="updatedBadge">更新时间读取中</span><span class="badge">统一口径：红涨绿跌</span></div></section>
<nav class="tabs">
<button class="tab active" data-tab="overview">总览</button><button class="tab" data-tab="macro">宏观与资金</button><button class="tab" data-tab="global">全球市场</button><button class="tab" data-tab="ashare">A股情绪与杠杆</button><button class="tab" data-tab="valuation">估值与偏离度</button><button class="tab" data-tab="fund">基金募集</button><button class="tab" data-tab="health">数据状态与口径</button>
</nav>

<section class="panel active" id="overview"><div class="grid kpis" id="overviewKpis"></div><div class="grid g2"><div class="card"><h3>全球主要资产标准化走势 <small>起点=100</small></h3><div id="overviewMarket" class="chart"></div></div><div class="card"><h3>A股市场温度</h3><div id="overviewBreadth" class="chart"></div></div></div></section>

<section class="panel" id="macro"><div class="grid kpis" id="macroKpis"></div><div class="grid g2"><div class="card"><h3>M1、M2与剪刀差</h3><div id="moneyChart" class="chart"></div></div><div class="card"><h3>银行间资金利率</h3><div id="liquidityChart" class="chart"></div></div><div class="card"><h3>社会融资规模</h3><div id="socialChart" class="chart"></div></div><div class="card"><h3>PMI与CPI</h3><div id="pmiChart" class="chart"></div></div></div></section>

<section class="panel" id="global"><div class="card"><h3>美国主要指数、美元与科技龙头</h3><div id="globalCards" class="asset-grid"></div></div><div class="grid g2" style="margin-top:16px"><div class="card"><h3>美国主要指数与美元指数 <small>起点=100</small></h3><div id="usIndexChart" class="chart"></div></div><div class="card"><h3>美国国债收益率</h3><div id="treasuryChart" class="chart"></div></div></div><div class="card" style="margin-top:16px"><h3>中美韩科技龙头 <small>包含美光、三星电子、SK海力士</small></h3><div id="techCards" class="asset-grid"></div><div id="techChart" class="chart tall"></div></div></section>

<section class="panel" id="ashare"><div class="grid kpis" id="ashareKpis"></div><div class="hint">交易拥挤度 = 当日A股成交额排名前5%的股票成交额合计 ÷ 沪深A股全部股票成交额。历史回填使用当前仍上市股票回溯，可能存在退市样本缺失；日常收盘更新使用当日完整股票池。</div><div class="grid g2"><div class="card"><h3>交易拥挤度历史</h3><div id="crowdingChart" class="chart"></div></div><div class="card"><h3>上涨、下跌和平盘家数</h3><div id="breadthChart" class="chart"></div></div><div class="card"><h3>A股成交额与广义换手率</h3><div id="turnoverChart" class="chart"></div></div><div class="card"><h3>两融余额 / 市场总市值</h3><div id="leverageChart" class="chart"></div></div></div></section>

<section class="panel" id="valuation"><div class="card"><h3>指数历史估值</h3><div id="valuationCards" class="valuation-grid"></div><div id="valuationChart" class="chart tall"></div></div><div class="card" style="margin-top:16px"><h3>指数偏离度 <small>(收盘价/10周或20周均线−1)×100%</small></h3><div id="deviationCards" class="asset-grid"></div><div id="deviationChart" class="chart tall"></div></div></section>

<section class="panel" id="fund"><div class="grid g2"><div class="card"><h3>单只新成立基金募集规模 <small>估算口径</small></h3><div id="fundChart" class="chart"></div></div><div class="card"><h3>口径说明</h3><div class="hint">免费公开源可稳定取得“募集份额（亿份）”，但不存在统一、连续、免费的“单只基金每日净申购金额”字段。网站按常见初始面值1元/份，将募集份额近似展示为募集规模（亿元），并明确标注，不把它冒充存续期净申购额。</div><div id="fundKpis" class="grid g2"></div></div></div><div class="card" style="margin-top:16px"><h3>最新新成立基金</h3><div class="table-wrap"><table><thead><tr><th>成立日期</th><th>基金代码</th><th>基金名称</th><th>类型</th><th>基金公司</th><th>募集份额（亿份）</th><th>估算规模（亿元）</th></tr></thead><tbody id="fundTable"></tbody></table></div></div></section>

<section class="panel" id="health"><div class="card"><h3>数据模块状态</h3><div id="sourceCards" class="source-grid"></div></div><div class="card" style="margin-top:16px"><h3>主要数据来源与口径</h3><div class="table-wrap"><table><thead><tr><th>模块</th><th>数据源</th><th>更新口径</th></tr></thead><tbody>
<tr><td>M1/M2、社融、PMI、CPI</td><td>人民银行、国家统计局及公开适配器</td><td>月度，发布后更新</td></tr><tr><td>DR001/DR007</td><td>中国货币网</td><td>存款类机构质押式回购加权利率；不以FDR替代</td></tr><tr><td>隔夜Shibor</td><td>中国货币网 / AKShare</td><td>每个工作日官方发布</td></tr><tr><td>美债收益率</td><td>美联储H.15 / FRED</td><td>DGS2、DGS10，单位%</td></tr><tr><td>全球指数与股票</td><td>Yahoo Finance / yfinance</td><td>日线复权收盘；红涨绿跌</td></tr><tr><td>A股情绪、成交额、市值</td><td>东方财富公开行情 / AKShare</td><td>收盘后统计沪深A股</td></tr><tr><td>两融余额</td><td>上交所、深交所 / AKShare</td><td>沪深两市融资融券余额合计</td></tr><tr><td>基金募集</td><td>天天基金 / AKShare</td><td>募集份额（亿份），1元/份近似规模</td></tr></tbody></table></div></div></section>

<div class="footer">{{ disclaimer }}<br>该网站只展示公开数据与计算结果。公开网页接口可能调整，系统会保留上一次成功缓存并在“数据状态”中披露失败。红色统一表示上涨/正收益，绿色统一表示下跌/负收益。</div>
</div>
<script>
const DATA={{ payload|safe }};
const C={blue:'#3167e3',navy:'#172c58',cyan:'#1e9ca5',purple:'#7456d8',amber:'#c98a18',up:'#d64242',down:'#159567',muted:'#78849a',grid:'#e8ecf3'};
const CONFIG={responsive:true,displaylogo:false,modeBarButtonsToRemove:['lasso2d','select2d']};
const baseLayout={paper_bgcolor:'rgba(0,0,0,0)',plot_bgcolor:'rgba(0,0,0,0)',font:{family:'-apple-system,BlinkMacSystemFont,Segoe UI,PingFang SC,Microsoft YaHei',color:'#536078',size:11},margin:{l:48,r:30,t:18,b:42},legend:{orientation:'h',y:1.13},hovermode:'x unified'};
const layout=x=>Object.assign({},baseLayout,x||{});
const fmt=(v,d=2)=>v===null||v===undefined||Number.isNaN(Number(v))?'—':Number(v).toLocaleString('zh-CN',{minimumFractionDigits:d,maximumFractionDigits:d});
const signed=(v,d=2)=>v===null||v===undefined||Number.isNaN(Number(v))?'—':`${Number(v)>=0?'+':''}${fmt(v,d)}%`;
const cnDate=v=>{const s=String(v||'');return s.length===8?`${s.slice(0,4)}-${s.slice(4,6)}-${s.slice(6,8)}`:s.length===6?`${s.slice(0,4)}-${s.slice(4,6)}`:s||'—'};
const cls=v=>Number(v)>0?'positive':Number(v)<0?'negative':'neutral';
function kpi(label,value,unit,note,klass=''){return `<div class="kpi"><div class="kpi-label">${label}</div><div class="kpi-value ${klass}">${value}${value==='—'?'':` <small style="font-size:12px;color:#7d899f">${unit}</small>`}</div><div class="kpi-note">${note||''}</div></div>`}
function latest(rows,key){for(let i=rows.length-1;i>=0;i--){const v=rows[i][key];if(v!==null&&v!==undefined&&!Number.isNaN(Number(v)))return {v:Number(v),row:rows[i]}}return {v:null,row:rows[rows.length-1]||{}}}
function normTraces(rows,symbols,max=520){const traces=[];symbols.forEach(sym=>{let g=rows.filter(r=>r.symbol===sym).sort((a,b)=>String(a.trade_date).localeCompare(String(b.trade_date))).slice(-max);if(!g.length)return;const base=g.find(x=>Number(x.close)>0)?.close;if(!base)return;traces.push({x:g.map(x=>cnDate(x.trade_date)),y:g.map(x=>Number(x.close)/Number(base)*100),name:g[0].name,mode:'lines',line:{width:2}})});return traces}
function assetCard(c){return `<div class="asset"><div class="asset-top"><div><div class="asset-name">${c.name}</div><div class="asset-symbol">${c.symbol} · ${cnDate(c.trade_date)}</div></div><b class="${cls(c.daily_pct)}">${signed(c.daily_pct)}</b></div><div class="asset-price">${fmt(c.close,2)} <small style="font-size:11px;color:#7d899f">${c.currency||''}</small></div><div class="asset-metrics"><div><span>当日</span><b class="${cls(c.daily_pct)}">${signed(c.daily_pct)}</b></div><div><span>年初至今</span><b class="${cls(c.ytd_pct)}">${signed(c.ytd_pct)}</b></div><div><span>近一年</span><b class="${cls(c.one_year_pct)}">${signed(c.one_year_pct)}</b></div></div></div>`}

document.getElementById('statusBadge').textContent=`数据状态：${DATA.status.overall_status||'empty'}`;document.getElementById('updatedBadge').textContent=`更新：${DATA.status.updated_at?DATA.status.updated_at.replace('T',' '):'尚未更新'}`;
document.querySelectorAll('.tab').forEach(btn=>btn.onclick=()=>{document.querySelectorAll('.tab').forEach(x=>x.classList.remove('active'));document.querySelectorAll('.panel').forEach(x=>x.classList.remove('active'));btn.classList.add('active');document.getElementById(btn.dataset.tab).classList.add('active');setTimeout(()=>window.dispatchEvent(new Event('resize')),80)});

function renderOverview(){const m=DATA.macro,b=DATA.breadth,c=DATA.crowding,l=DATA.liquidity,gm=DATA.global_macro,cards=DATA.market_cards;const gap=latest(m,'m1_m2_gap_pp'),dr7=latest(l,'dr007_pct'),crowd=latest(c,'crowding_pct'),up=latest(b,'up_count'),turn=latest(b,'broad_turnover_pct'),t10=latest(gm.filter(x=>x.series==='DGS10'),'value_pct');document.getElementById('overviewKpis').innerHTML=[kpi('M1−M2剪刀差',fmt(gap.v),'个百分点',cnDate(gap.row.month),cls(gap.v)),kpi('DR007',fmt(dr7.v),'%',cnDate(dr7.row.trade_date)),kpi('A股交易拥挤度',fmt(crowd.v),'%',cnDate(crowd.row.trade_date)),kpi('A股上涨家数',fmt(up.v,0),'只',cnDate(up.row.trade_date),'positive'),kpi('成交额/总市值',fmt(turn.v),'%',cnDate(turn.row.trade_date)),kpi('美国10年期国债',fmt(t10.v),'%',cnDate(t10.row.trade_date))].join('');const main=['^IXIC','^GSPC','^DJI','DX-Y.NYB','000688.SH'];Plotly.newPlot('overviewMarket',normTraces(DATA.market,main,300),layout({yaxis:{title:'起点=100',gridcolor:C.grid}}),CONFIG);if(b.length){const recent=b.slice(-80);Plotly.newPlot('overviewBreadth',[{x:recent.map(x=>cnDate(x.trade_date)),y:recent.map(x=>x.up_count),name:'上涨',type:'bar',marker:{color:C.up}},{x:recent.map(x=>cnDate(x.trade_date)),y:recent.map(x=>-Number(x.down_count||0)),name:'下跌（负轴）',type:'bar',marker:{color:C.down}}],layout({barmode:'relative',yaxis:{title:'家数',gridcolor:C.grid}}),CONFIG)}}

function renderMacro(){const m=DATA.macro,l=DATA.liquidity;if(!m.length)return;const a=latest(m,'m1_yoy_pct'),b=latest(m,'m2_yoy_pct'),gap=latest(m,'m1_m2_gap_pp'),sf=latest(m,'sf_stock_yoy_pct'),pmi=latest(m,'pmi_manufacturing'),cpi=latest(m,'cpi_yoy_pct');document.getElementById('macroKpis').innerHTML=[kpi('M1同比',fmt(a.v),'%',cnDate(a.row.month),cls(a.v)),kpi('M2同比',fmt(b.v),'%',cnDate(b.row.month),cls(b.v)),kpi('剪刀差',fmt(gap.v),'个百分点',cnDate(gap.row.month),cls(gap.v)),kpi('社融存量同比',fmt(sf.v),'%',cnDate(sf.row.month),cls(sf.v)),kpi('制造业PMI',fmt(pmi.v),'点',cnDate(pmi.row.month),Number(pmi.v)>=50?'positive':'negative'),kpi('CPI同比',fmt(cpi.v),'%',cnDate(cpi.row.month),cls(cpi.v))].join('');const x=m.map(r=>cnDate(r.month));Plotly.newPlot('moneyChart',[{x,y:m.map(r=>r.m1_yoy_pct),name:'M1同比',mode:'lines',line:{color:C.blue,width:2.5}},{x,y:m.map(r=>r.m2_yoy_pct),name:'M2同比',mode:'lines',line:{color:C.cyan,width:2.5}},{x,y:m.map(r=>r.m1_m2_gap_pp),name:'剪刀差',type:'bar',marker:{color:'rgba(201,138,24,.35)'}}],layout({yaxis:{title:'% / 百分点',gridcolor:C.grid},barmode:'relative'}),CONFIG);if(l.length)Plotly.newPlot('liquidityChart',[{x:l.map(r=>cnDate(r.trade_date)),y:l.map(r=>r.dr001_pct),name:'DR001',mode:'lines',line:{color:C.blue,width:2.2}},{x:l.map(r=>cnDate(r.trade_date)),y:l.map(r=>r.dr007_pct),name:'DR007',mode:'lines',line:{color:C.purple,width:2.2}},{x:l.map(r=>cnDate(r.trade_date)),y:l.map(r=>r.shibor_on_pct),name:'隔夜Shibor',mode:'lines',line:{color:C.amber,width:2}}],layout({yaxis:{title:'%',gridcolor:C.grid}}),CONFIG);Plotly.newPlot('socialChart',[{x,y:m.map(r=>r.sf_increment_trillion),name:'当月社融增量',type:'bar',marker:{color:'rgba(49,103,227,.35)'}},{x,y:m.map(r=>r.sf_stock_trillion),name:'社融存量',mode:'lines',line:{color:C.navy,width:2.3}},{x,y:m.map(r=>r.sf_stock_yoy_pct),name:'存量同比',mode:'lines',line:{color:C.up,width:2},yaxis:'y2'}],layout({yaxis:{title:'万亿元',gridcolor:C.grid},yaxis2:{title:'%',overlaying:'y',side:'right',showgrid:false}}),CONFIG);Plotly.newPlot('pmiChart',[{x,y:m.map(r=>r.pmi_manufacturing),name:'制造业PMI',mode:'lines',line:{color:C.blue,width:2.3}},{x,y:m.map(r=>r.pmi_non_manufacturing),name:'非制造业PMI',mode:'lines',line:{color:C.cyan,width:2.1}},{x,y:m.map(r=>r.cpi_yoy_pct),name:'CPI同比',mode:'lines',line:{color:C.up,width:2},yaxis:'y2'}],layout({yaxis:{title:'PMI点',gridcolor:C.grid},yaxis2:{title:'CPI %',overlaying:'y',side:'right',showgrid:false},shapes:[{type:'line',x0:x[0],x1:x[x.length-1],y0:50,y1:50,line:{color:C.muted,dash:'dot'}}]}),CONFIG)}

function renderGlobal(){const cards=DATA.market_cards;const usIndex=cards.filter(c=>['US_INDEX','FX_INDEX'].includes(c.market));const tech=cards.filter(c=>['US_EQUITY','KR_EQUITY','CN_INDEX'].includes(c.market));document.getElementById('globalCards').innerHTML=usIndex.map(assetCard).join('');document.getElementById('techCards').innerHTML=tech.map(assetCard).join('');Plotly.newPlot('usIndexChart',normTraces(DATA.market,usIndex.map(x=>x.symbol),420),layout({yaxis:{title:'起点=100',gridcolor:C.grid}}),CONFIG);Plotly.newPlot('techChart',normTraces(DATA.market,tech.map(x=>x.symbol),420),layout({yaxis:{title:'起点=100',gridcolor:C.grid}}),CONFIG);const g=DATA.global_macro;Plotly.newPlot('treasuryChart',[{x:g.filter(x=>x.series==='DGS2').map(x=>cnDate(x.trade_date)),y:g.filter(x=>x.series==='DGS2').map(x=>x.value_pct),name:'美国2年期',mode:'lines',line:{color:C.blue,width:2.1}},{x:g.filter(x=>x.series==='DGS10').map(x=>cnDate(x.trade_date)),y:g.filter(x=>x.series==='DGS10').map(x=>x.value_pct),name:'美国10年期',mode:'lines',line:{color:C.purple,width:2.1}}],layout({yaxis:{title:'%',gridcolor:C.grid}}),CONFIG)}

function renderAshare(){const c=DATA.crowding,b=DATA.breadth,l=DATA.leverage;const crowd=latest(c,'crowding_pct'),up=latest(b,'up_count'),down=latest(b,'down_count'),amount=latest(b,'total_amount_trillion'),turn=latest(b,'broad_turnover_pct'),margin=latest(l,'margin_balance_trillion'),ratio=latest(l,'margin_to_market_cap_pct');document.getElementById('ashareKpis').innerHTML=[kpi('交易拥挤度',fmt(crowd.v),'%',cnDate(crowd.row.trade_date)),kpi('上涨家数',fmt(up.v,0),'只',cnDate(up.row.trade_date),'positive'),kpi('下跌家数',fmt(down.v,0),'只',cnDate(down.row.trade_date),'negative'),kpi('两市成交额',fmt(amount.v),'万亿元',cnDate(amount.row.trade_date)),kpi('成交额/总市值',fmt(turn.v),'%',cnDate(turn.row.trade_date)),kpi('两融余额/总市值',fmt(ratio.v),'%',`${cnDate(ratio.row.trade_date)} · 两融${fmt(margin.v)}万亿元`)].join('');if(c.length)Plotly.newPlot('crowdingChart',[{x:c.map(r=>cnDate(r.trade_date)),y:c.map(r=>r.crowding_pct),name:'前5%成交额占比',mode:'lines',fill:'tozeroy',line:{color:C.purple,width:2.3},fillcolor:'rgba(116,86,216,.12)'}],layout({yaxis:{title:'%',gridcolor:C.grid}}),CONFIG);if(b.length){Plotly.newPlot('breadthChart',[{x:b.map(r=>cnDate(r.trade_date)),y:b.map(r=>r.up_count),name:'上涨',type:'bar',marker:{color:C.up}},{x:b.map(r=>cnDate(r.trade_date)),y:b.map(r=>r.down_count),name:'下跌',type:'bar',marker:{color:C.down}},{x:b.map(r=>cnDate(r.trade_date)),y:b.map(r=>r.flat_count),name:'平盘',type:'bar',marker:{color:C.muted}}],layout({barmode:'stack',yaxis:{title:'家数',gridcolor:C.grid}}),CONFIG);Plotly.newPlot('turnoverChart',[{x:b.map(r=>cnDate(r.trade_date)),y:b.map(r=>r.total_amount_trillion),name:'两市成交额',type:'bar',marker:{color:'rgba(49,103,227,.35)'}},{x:b.map(r=>cnDate(r.trade_date)),y:b.map(r=>r.broad_turnover_pct),name:'成交额/总市值',mode:'lines',line:{color:C.amber,width:2.2},yaxis:'y2'}],layout({yaxis:{title:'万亿元',gridcolor:C.grid},yaxis2:{title:'%',overlaying:'y',side:'right',showgrid:false}}),CONFIG)}if(l.length)Plotly.newPlot('leverageChart',[{x:l.map(r=>cnDate(r.trade_date)),y:l.map(r=>r.margin_balance_trillion),name:'两融余额',type:'bar',marker:{color:'rgba(30,156,165,.36)'}},{x:l.map(r=>cnDate(r.trade_date)),y:l.map(r=>r.margin_to_market_cap_pct),name:'两融/总市值',mode:'lines',line:{color:C.up,width:2.2},yaxis:'y2'}],layout({yaxis:{title:'万亿元',gridcolor:C.grid},yaxis2:{title:'%',overlaying:'y',side:'right',showgrid:false}}),CONFIG)}

let selectedVal=null;function renderValuation(){const s=DATA.valuation_summary;if(s.length){selectedVal=selectedVal||s[0].index_code;document.getElementById('valuationCards').innerHTML=s.map(v=>`<div class="valuation-card ${v.index_code===selectedVal?'selected':''}" data-code="${v.index_code}"><b>${v.index_name}</b><div class="asset-symbol">${cnDate(v.trade_date)}</div><div class="multiple">${fmt(v.current_pe)} <small style="font-size:11px;color:#7d899f">倍</small></div><div class="stat-row"><div class="stat">均值<b>${fmt(v.pe_mean)}</b></div><div class="stat">中位数<b>${fmt(v.pe_median)}</b></div><div class="stat">分位<b>${fmt(v.pe_percentile,0)}%</b></div></div><div class="bar"><i style="width:${Math.min(100,Math.max(0,v.pe_percentile||0))}%"></i></div></div>`).join('');document.querySelectorAll('.valuation-card').forEach(x=>x.onclick=()=>{selectedVal=x.dataset.code;renderValuation()});const rows=DATA.valuation.filter(x=>x.index_code===selectedVal);const meta=s.find(x=>x.index_code===selectedVal);Plotly.newPlot('valuationChart',[{x:rows.map(r=>cnDate(r.trade_date)),y:rows.map(r=>r.pe_ttm),name:`${meta.index_name} PE TTM`,mode:'lines',line:{color:C.blue,width:2.2}},{x:rows.map(r=>cnDate(r.trade_date)),y:rows.map(()=>meta.pe_mean),name:'历史均值',mode:'lines',line:{color:C.amber,dash:'dash'}},{x:rows.map(r=>cnDate(r.trade_date)),y:rows.map(()=>meta.pe_q25),name:'25%分位',mode:'lines',line:{color:C.down,dash:'dot'}},{x:rows.map(r=>cnDate(r.trade_date)),y:rows.map(()=>meta.pe_q75),name:'75%分位',mode:'lines',line:{color:C.up,dash:'dot'}}],layout({yaxis:{title:'倍',gridcolor:C.grid}}),CONFIG)}const d=DATA.deviation;const latestRows=[];[...new Set(d.map(x=>x.symbol))].forEach(sym=>{const g=d.filter(x=>x.symbol===sym).sort((a,b)=>String(a.trade_date).localeCompare(String(b.trade_date)));if(g.length)latestRows.push(g[g.length-1])});document.getElementById('deviationCards').innerHTML=latestRows.map(x=>`<div class="asset"><div class="asset-name">${x.name}</div><div class="asset-symbol">${cnDate(x.trade_date)}</div><div class="asset-metrics" style="margin-top:12px"><div><span>偏离10周</span><b class="${cls(x.dev10w_pct)}">${signed(x.dev10w_pct)}</b></div><div><span>偏离20周</span><b class="${cls(x.dev20w_pct)}">${signed(x.dev20w_pct)}</b></div><div><span>收盘</span><b>${fmt(x.close)}</b></div></div></div>`).join('');const traces=[];latestRows.forEach(x=>{const g=d.filter(r=>r.symbol===x.symbol).slice(-160);traces.push({x:g.map(r=>cnDate(r.trade_date)),y:g.map(r=>r.dev10w_pct),name:`${x.name} 10周`,mode:'lines',line:{width:1.8}})});Plotly.newPlot('deviationChart',traces,layout({yaxis:{title:'%',gridcolor:C.grid},shapes:[{type:'line',x0:0,x1:1,xref:'paper',y0:0,y1:0,line:{color:C.muted,dash:'dot'}}]}),CONFIG)}

function renderFund(){const f=DATA.fund;if(!f.length)return;const sorted=[...f].sort((a,b)=>String(a.founded_date).localeCompare(String(b.founded_date)));const recent=sorted.slice(-30).sort((a,b)=>Number(b.estimated_raised_amount_100m)-Number(a.estimated_raised_amount_100m)).slice(0,15);Plotly.newPlot('fundChart',[{x:recent.map(x=>x.fund_name),y:recent.map(x=>x.estimated_raised_amount_100m),type:'bar',marker:{color:'rgba(116,86,216,.65)'},name:'估算募集规模'}],layout({margin:{l:50,r:20,t:15,b:120},yaxis:{title:'亿元',gridcolor:C.grid},xaxis:{tickangle:-35}}),CONFIG);const last=sorted[sorted.length-1],largest=recent[0];document.getElementById('fundKpis').innerHTML=[kpi('最新成立基金',last.fund_name,'',`${cnDate(last.founded_date)} · ${fmt(last.raised_shares_100m)}亿份`),kpi('近30条最大募集',largest?.fund_name||'—','',largest?`${fmt(largest.estimated_raised_amount_100m)}亿元（估算）`:'' )].join('');document.getElementById('fundTable').innerHTML=sorted.slice(-80).reverse().map(x=>`<tr><td>${cnDate(x.founded_date)}</td><td>${x.fund_code}</td><td>${x.fund_name}</td><td>${x.fund_type||''}</td><td>${x.fund_company||''}</td><td>${fmt(x.raised_shares_100m)}</td><td>${fmt(x.estimated_raised_amount_100m)}</td></tr>`).join('')}

function renderHealth(){const ds=DATA.status.datasets||{};const labels={macro:'宏观数据',liquidity:'DR/Shibor',market:'全球行情',global_macro:'美债收益率',valuation:'历史估值',sentiment:'A股情绪',crowding:'交易拥挤度',breadth:'涨跌家数',leverage:'两融杠杆',deviation:'指数偏离度',fund_subscription:'基金募集'};document.getElementById('sourceCards').innerHTML=Object.entries(labels).map(([key,label])=>{const d=ds[key]||{status:'empty'};return `<div class="source-card"><div class="source-status ${d.status}">${label} · ${d.status||'empty'}</div><div class="asset-symbol" style="margin-top:7px">最新日期：${cnDate(d.latest_date)}</div><div class="asset-symbol">本次写入：${d.rows??0} 行；缓存：${d.cached_rows??d.total_cached_rows??0} 行</div>${d.error?`<div class="hint">${String(d.error).slice(0,420)}</div>`:''}</div>`}).join('')}

renderOverview();renderMacro();renderGlobal();renderAshare();renderValuation();renderFund();renderHealth();
</script></body></html>'''


def main() -> None:
    ensure_dirs()
    settings = load_settings()
    paths = {
        "macro": "macro.csv", "liquidity": "liquidity.csv", "market": "market.csv",
        "global_macro": "global_macro.csv", "valuation": "valuation.csv",
        "crowding": "crowding.csv", "breadth": "breadth.csv", "leverage": "leverage.csv",
        "deviation": "deviation.csv", "fund": "fund_subscription.csv",
    }
    data = {key: read_csv_safe(DATA_DIR / filename) for key, filename in paths.items()}
    for key in ["market", "valuation", "crowding", "breadth", "leverage", "liquidity", "global_macro", "deviation"]:
        if not data[key].empty:
            sort_cols = [x for x in ["symbol", "index_code", "series", "trade_date"] if x in data[key].columns]
            data[key] = data[key].sort_values(sort_cols) if sort_cols else data[key]
    payload = {
        "status": read_status(),
        "macro": dataframe_to_records(data["macro"]),
        "liquidity": dataframe_to_records(data["liquidity"], max_rows=5000),
        "market": dataframe_to_records(data["market"], max_rows=18 * 900),
        "market_cards": latest_market_cards(data["market"]),
        "global_macro": dataframe_to_records(data["global_macro"], max_rows=2 * 7000),
        "valuation": dataframe_to_records(data["valuation"], max_rows=7 * 4500),
        "valuation_summary": valuation_summary(data["valuation"]),
        "crowding": dataframe_to_records(data["crowding"], max_rows=3000),
        "breadth": dataframe_to_records(data["breadth"], max_rows=3000),
        "leverage": dataframe_to_records(data["leverage"], max_rows=3000),
        "deviation": dataframe_to_records(data["deviation"], max_rows=5 * 800),
        "fund": dataframe_to_records(data["fund"], max_rows=400),
    }
    html = Template(HTML).render(
        title=f"{settings['app']['name']} · {settings['app']['title_cn']}",
        version=settings["app"]["version"],
        disclaimer=settings["app"]["disclaimer"],
        payload=json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/"),
    )
    PUBLIC_DIR.mkdir(parents=True, exist_ok=True)
    (PUBLIC_DIR / "index.html").write_text(html, encoding="utf-8")
    (PUBLIC_DIR / ".nojekyll").write_text("", encoding="utf-8")
    print(f"Built {PUBLIC_DIR / 'index.html'} ({len(html):,} chars)")


if __name__ == "__main__":
    main()
