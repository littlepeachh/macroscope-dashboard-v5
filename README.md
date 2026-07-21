# MacroScope Public v5.0

一个无需用户 Token、可通过 GitHub Pages 公开访问的全球宏观与市场看板。

## v5 新增

- A股交易拥挤度历史：成交额排名前 5% 的A股成交额 / 沪深A股全部成交额。
- A股上涨、下跌、平盘家数及两市成交额。
- A股广义换手率：两市总成交额 / A股总市值。
- 两融杠杆：沪深两融余额 / A股总市值。
- DR001、DR007、隔夜 Shibor。
- 美国2年期、10年期国债收益率与美元指数。
- 纳斯达克综合指数、标普500、道琼斯工业指数。
- 指数相对10周、20周均线的偏离度。
- 单只新成立基金募集份额与估算募集规模。
- 科技龙头新增美光科技、三星电子、SK海力士。
- 所有中国、美国、韩国市场统一使用 **红色上涨、绿色下跌**。

## 页面模块

1. 总览
2. 宏观与资金
3. 全球市场
4. A股情绪与杠杆
5. 估值与偏离度
6. 基金募集
7. 数据状态与口径

## 数据文件

- `data/macro.csv`
- `data/liquidity.csv`
- `data/market.csv`
- `data/global_macro.csv`
- `data/valuation.csv`
- `data/crowding.csv`
- `data/breadth.csv`
- `data/leverage.csv`
- `data/deviation.csv`
- `data/fund_subscription.csv`

## 自动更新时间（北京时间）

- 06:20：全球市场、美股/韩股、美元指数、美债收益率。
- 10:10：中国宏观、DR、Shibor。
- 16:40（周一至周五）：A股收盘、估值、涨跌家数、拥挤度、换手率、两融杠杆、指数偏离度。
- 18:40：中国宏观、资金利率和基金募集再次检查。

GitHub Actions 可能因平台排队延迟几分钟。

## 历史交易拥挤度

首次部署后，在 GitHub Actions 中运行 `Backfill A-share crowding history`。建议先从最近一年开始：

- `start_date`: `20250101`
- `end_date`: 留空
- `batch_size`: `100`

回填会同时生成交易拥挤度与上涨/下跌家数历史。历史股票池使用回填时仍在上市的A股代码，可能遗漏历史退市股票，因此网站会明确披露该近似口径。

## 基金申购指标说明

公开免费数据源没有统一、稳定的“单只基金每日净申购金额”历史字段。本项目使用新成立基金的“募集份额（亿份）”，按常见初始面值1元/份近似展示募集规模（亿元）。它是募集期规模，不是基金存续期日常净申购额。

## 本地演示

```bash
python scripts/generate_demo_data.py
python scripts/build_site.py
python -m http.server 8000 --directory public
```

打开 `http://localhost:8000`。演示数据为合成数据，不可用于研究结论。

## GitHub Pages 部署

见：

- `docs/UPDATE_EXISTING_GITHUB_SITE.md`
- `docs/DATA_DICTIONARY.md`
- `docs/SOURCE_MAP.md`

## 许可与限制

代码使用 MIT License。底层公开数据的使用、展示与再分发受各数据源条款约束。本项目适合研究、内部展示和原型验证；正式收费或大规模商业发布前，应单独取得相应数据许可。
