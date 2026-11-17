# -*- coding: utf-8 -*-
"""
Day 1: 量化第一课 (v3 - 双数据源 + 本地缓存)
1) 拉 A 股日线: 优先东方财富, 被限流就自动切腾讯
2) 向量化 vs for 循环
3) 算一个动量因子的 IC
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
import akshare as ak

SYMBOL = "000001"
START, END = "20220101", "20241231"
CACHE = Path("data") / f"{SYMBOL}_{START}_{END}.csv"


def _tx_symbol(code: str) -> str:
    """000001 -> sz000001, 600519 -> sh600519"""
    if code.startswith(("6", "9")):
        return "sh" + code
    if code.startswith(("4", "8")):
        return "bj" + code
    return "sz" + code


def load_daily(symbol, start, end, adjust="qfq"):
    """返回统一列名 date / close / turnover(百分数) 的日线。"""
    try:
        d = ak.stock_zh_a_hist_em(
            symbol=symbol, period="daily",
            start_date=start, end_date=end, adjust=adjust,
        )
        d = d.rename(columns={"日期": "date", "收盘": "close", "换手率": "turnover"})
        d = d[["date", "close", "turnover"]]
        print("数据源: 东方财富")
    except Exception as e:
        # 东财限流时不要重试, 越试封得越久, 直接换源
        print("东方财富不可用, 改用腾讯")
        d = ak.stock_zh_a_hist_tx(
            symbol=_tx_symbol(symbol),
            start_date=start, end_date=end, adjust=adjust,
        )
        d = d[["date", "close", "turnover"]].copy()
        d["turnover"] = d["turnover"] * 100   # 腾讯是小数(0.006), 乘100对齐东财(0.6)
        print("数据源: 腾讯")
    d["date"] = pd.to_datetime(d["date"])
    return d.sort_values("date").reset_index(drop=True)


# ========== 1. 取数据 ==========
if CACHE.exists():
    print("读取本地缓存:", CACHE)
    df = pd.read_csv(CACHE, encoding="utf-8-sig", parse_dates=["date"])
else:
    print("正在下载...")
    df = load_daily(SYMBOL, START, END)
    CACHE.parent.mkdir(exist_ok=True)
    df.to_csv(CACHE, index=False, encoding="utf-8-sig")
    print("已缓存到:", CACHE)

print(df.head())
print("行数:", len(df))

# ========== 2. 向量化 vs for 循环 ==========
n = 1_000_000
rng = np.random.default_rng(0)
prices = np.cumsum(rng.normal(0, 1, n)) + 100

t0 = time.perf_counter()
ma_vec = pd.Series(prices).rolling(20).mean()
t_vec = time.perf_counter() - t0

t0 = time.perf_counter()
ma_loop = [np.nan] * 19
for i in range(19, n):
    ma_loop.append(prices[i - 19 : i + 1].mean())
t_loop = time.perf_counter() - t0

print(f"\n向量化 : {t_vec * 1000:8.2f} ms")
print(f"for循环: {t_loop * 1000:8.2f} ms")
print(f"快 {t_loop / t_vec:.0f} 倍")
print("两者结果一致:", np.allclose(ma_vec.to_numpy(), np.array(ma_loop), equal_nan=True))

# ========== 3. 因子与 IC ==========
df["ret_fwd"] = df["close"].pct_change().shift(-1)
df["mom20"] = df["close"].pct_change(1)

sub = df.dropna(subset=["mom20", "ret_fwd"])
ic = sub["mom20"].corr(sub["ret_fwd"], method="spearman")

print(f"\n1日动量因子 与 次日收益 的 IC (Spearman): {ic:.4f}")
print("样本数:", len(sub))
print("\n注: 这是单只股票的时间序列 IC, 只作演示。")
print("    真正的 IC 是横截面的: 同一天, 全市场所有股票 vs 未来收益。")
