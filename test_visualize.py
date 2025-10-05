import pandas as pd
from binance.client import Client
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from scipy.signal import find_peaks

def fetch_btc_daily_data(days="3000 days ago UTC"):
    client = Client()
    klines = client.get_historical_klines("BTCUSDT", Client.KLINE_INTERVAL_1DAY, days)
    all_columns = ['Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time', 'Quote asset volume', 'Number of trades', 'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore']
    df = pd.DataFrame(klines, columns=all_columns)
    df = df[['Open time', 'Open', 'High', 'Low', 'Close', 'Volume']].copy()
    df['Date'] = pd.to_datetime(df['Open time'], unit='ms')
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        df[col] = pd.to_numeric(df[col])
    return df.set_index('Date')

# --- 채널 계산 및 그리기 함수 ---
def plot_delayed_channel(ax, dates, prices, mode):
    peaks, _ = find_peaks(prices['High'], prominence=8000, width=10)
    troughs, _ = find_peaks(-prices['Low'], prominence=8000, width=10)
    peak_dates, trough_dates = prices.index[peaks], prices.index[troughs]
    all_dates_num = mdates.date2num(dates)
    
    if mode == 'bearish':
        # --- 로직 수정: 과거 하락장 타겟팅 ---
        historical_prices = prices.iloc[:-365] # 최근 1년 제외
        ath_date = historical_prices['High'].idxmax()
        next_peaks = peak_dates[peak_dates > ath_date]
        if len(next_peaks) == 0: return
        
        p1_date, p2_date = ath_date, next_peaks[0]
        relevant_troughs = trough_dates[(trough_dates > p1_date) & (trough_dates < p2_date)]
        if len(relevant_troughs) == 0: return
        p_low_date = relevant_troughs[0]
        
        p1_price, p2_price, p_low_price = prices.loc[p1_date]['High'], prices.loc[p2_date]['High'], prices.loc[p_low_date]['Low']
        
        p1_num, p2_num, p_low_num = mdates.date2num([p1_date, p2_date, p_low_date])
        log_p1, log_p2, log_p_low = np.log([p1_price, p2_price, p_low_price])
        
        slope = (log_p2 - log_p1) / (p2_num - p1_num)
        intercept_res = log_p1 - slope * p1_num
        intercept_sup = log_p_low - slope * p_low_num
        
        line_res = np.exp(slope * all_dates_num + intercept_res)
        line_sup = np.exp(slope * all_dates_num + intercept_sup)
        channel_width = intercept_res - intercept_sup
        line_delayed = np.exp(np.log(line_res) + channel_width)
        
        ax.fill_between(dates, line_res, line_sup, color='royalblue', alpha=0.1, label='Bearish Primary')
        ax.fill_between(dates, line_delayed, line_res, color='lightskyblue', alpha=0.1, label='Bearish Delayed')
        ax.plot([p1_date, p2_date, p_low_date], [p1_price, p2_price, p_low_price], 'x', color='blue', markersize=7)
        
    elif mode == 'bullish':
        if len(trough_dates) < 2: return
        p1_date, p2_date = trough_dates[-2], trough_dates[-1]
        relevant_peaks = peak_dates[(peak_dates > p1_date) & (peak_dates < p2_date)]
        if len(relevant_peaks) == 0: return
        
        p_high_date = relevant_peaks[0]
        p1_price, p2_price, p_high_price = prices.loc[p1_date]['Low'], prices.loc[p2_date]['Low'], prices.loc[p_high_date]['High']
        
        p1_num, p2_num, p_high_num = mdates.date2num([p1_date, p2_date, p_high_date])
        log_p1, log_p2, log_p_high = np.log([p1_price, p2_price, p_high_price])
        
        slope = (log_p2 - log_p1) / (p2_num - p1_num)
        intercept_sup = log_p1 - slope * p1_num
        intercept_res = log_p_high - slope * p_high_num
        
        line_sup = np.exp(slope * all_dates_num + intercept_sup)
        line_res = np.exp(slope * all_dates_num + intercept_res)
        channel_width = intercept_res - intercept_sup
        line_delayed = np.exp(np.log(line_sup) - channel_width)
        
        ax.fill_between(dates, line_res, line_sup, color='darkgreen', alpha=0.1, label='Bullish Primary')
        ax.fill_between(dates, line_sup, line_delayed, color='palegreen', alpha=0.1, label='Bullish Delayed')
        ax.plot([p1_date, p2_date, p_high_date], [p1_price, p2_price, p_high_price], 'x', color='green', markersize=7)
        
        # --- 채널 배수 확장 ---
        multiples = [0.5, 1.5, 2.0]
        for m in multiples:
            # 지연 채널(하단)을 기준으로 확장
            expanded_line = np.exp(np.log(line_delayed) - (channel_width * (m-1)))
            ax.plot(dates, expanded_line, ':', color='gray', linewidth=1)

# --- 시각화 실행 ---
btc_df = fetch_btc_daily_data()
future_dates = pd.date_range(start=btc_df.index[-1] + pd.Timedelta(days=1), periods=365)
all_dates = btc_df.index.append(future_dates)

# --- 가독성 개선: 흰색 배경 스타일 적용 ---
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(15, 8))
ax.plot(btc_df.index, btc_df['Close'], label='BTC Close Price', color='black', linewidth=1.5)
ax.set_yscale('log')

plot_delayed_channel(ax, all_dates, btc_df, 'bearish')
plot_delayed_channel(ax, all_dates, btc_df, 'bullish')

ax.set_ylim(bottom=btc_df['Low'].min() * 0.8, top=btc_df['High'].max() * 1.2)
ax.set_title('Delayed Trend Channel Strategy with Expansions', fontsize=16)
ax.set_ylabel('Price (USDT) - Log Scale', fontsize=12)
ax.legend(fontsize=10)
plt.tight_layout()
plt.show()