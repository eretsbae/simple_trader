import pandas as pd
import pandas_ta as ta
from binance.client import Client
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from scipy.signal import find_peaks
from datetime import datetime, timedelta

class TradingBot:
    def __init__(self, symbol='BTCUSDT', lt_tf='1d', st_tf='4h'):
        self.symbol = symbol
        self.lt_tf = lt_tf
        self.st_tf = st_tf
        self.client = Client()
        print(f"'{self.symbol}' 봇 초기화 (장기: {self.lt_tf}, 단기/기준: {self.st_tf})")

    def fetch_data(self, timeframe, start_str='2020-01-01', limit=None):
        """[수정] 데이터 분석 시작 시점을 2020년 1월 1일로 변경"""
        if start_str and not limit:
            all_klines = []
            start_ts = int(datetime.strptime(start_str, '%Y-%m-%d').timestamp() * 1000)
            while True:
                klines = self.client.get_historical_klines(self.symbol, timeframe, start_str=start_ts)
                if not klines: break
                all_klines.extend(klines)
                start_ts = klines[-1][0] + 1
            df = pd.DataFrame(all_klines)
        elif limit:
            klines = self.client.get_historical_klines(self.symbol, timeframe, limit=limit)
            df = pd.DataFrame(klines)
        else: raise ValueError("start_str 또는 limit 중 하나는 반드시 지정해야 합니다.")
        df.columns = ['Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time', 'Quote asset volume', 'Number of trades', 'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore']
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']: df[col] = pd.to_numeric(df[col])
        df['Date'] = pd.to_datetime(df['Open time'], unit='ms')
        df.drop_duplicates(subset='Date', inplace=True)
        df.set_index('Date', inplace=True)
        return df

    def determine_market_regime(self):
        # ... (이전과 동일)
        return 'BULLISH'

    def _analyze_channels(self, df, prominence_ratio, mode='recent', timeframe_label=''):
        # ... (이전과 동일, 버그 수정된 최종 로직)
        print(f"\n[{timeframe_label} 분석 시작] (Mode: {mode})")
        avg_price = df['Close'].mean()
        prominence = avg_price * prominence_ratio
        peaks_indices, _ = find_peaks(df['High'], prominence=prominence)
        troughs_indices, _ = find_peaks(-df['Low'], prominence=prominence)
        peak_dates = df.index[peaks_indices]
        channels = {}
        status_messages = []

        if len(troughs_indices) >= 2:
            p1_idx, p2_idx = (troughs_indices[0], troughs_indices[1]) if mode == 'all_time' else (troughs_indices[-2], troughs_indices[-1])
            if p1_idx < len(df) and p2_idx < len(df):
                p1_date, p2_date = df.index[p1_idx], df.index[p2_idx]
                relevant_peaks = peak_dates[(peak_dates > p1_date) & (peak_dates < p2_date)]
                if len(relevant_peaks) > 0:
                    p_high_date = relevant_peaks[0]
                    channels['bullish'] = self._calculate_channel_params(df, p1_date, p2_date, p_high_date, 'bullish')
                    status_messages.append(f'[{timeframe_label}] Bullish: OK')
                else: status_messages.append(f'[{timeframe_label}] Bullish: FAILED (No peak found)')
        else: status_messages.append(f'[{timeframe_label}] Bullish: FAILED (Not enough troughs)')
        
        if len(peaks_indices) >= 2:
            best_channel = None
            if mode == 'all_time':
                highest_score = 0
                for i in range(len(peaks_indices) - 1):
                    p1_idx, p2_idx = peaks_indices[i], peaks_indices[i+1]
                    p1_date, p2_date = df.index[p1_idx], df.index[p2_idx]
                    relevant_troughs_indices = troughs_indices[(df.index[troughs_indices] > p1_date) & (df.index[troughs_indices] < p2_date)]
                    if len(relevant_troughs_indices) > 0:
                        p_low_idx = relevant_troughs_indices[0]
                        score = (df.iloc[p1_idx]['High'] - df.iloc[p_low_idx]['Low']) / df.iloc[p1_idx]['High']
                        if score > highest_score:
                            highest_score = score
                            p_low_date = df.index[p_low_idx]
                            best_channel = self._calculate_channel_params(df, p1_date, p2_date, p_low_date, 'bearish')
            else: 
                p1_idx, p2_idx = peaks_indices[-2], peaks_indices[-1]
                p1_date, p2_date = df.index[p1_idx], df.index[p2_idx]
                relevant_troughs_indices = troughs_indices[(df.index[troughs_indices] > p1_date) & (df.index[troughs_indices] < p2_date)]
                if len(relevant_troughs_indices) > 0:
                    p_low_idx = relevant_troughs_indices[0]
                    p_low_date = df.index[p_low_idx]
                    best_channel = self._calculate_channel_params(df, p1_date, p2_date, p_low_date, 'bearish')
            if best_channel:
                channels['bearish'] = best_channel
                status_messages.append(f'[{timeframe_label}] Bearish: OK')
            else: status_messages.append(f'[{timeframe_label}] Bearish: FAILED (No valid channel)')
        else: status_messages.append(f'[{timeframe_label}] Bearish: FAILED (Not enough peaks)')

        print("  - " + "\n  - ".join(status_messages))
        return channels, status_messages
        
    def _calculate_channel_params(self, df, p1_date, p2_date, p3_date, channel_type):
        if channel_type == 'bullish': p1_price, p2_price, p3_price = df.loc[p1_date]['Low'], df.loc[p2_date]['Low'], df.loc[p3_date]['High']
        else: p1_price, p2_price, p3_price = df.loc[p1_date]['High'], df.loc[p2_date]['High'], df.loc[p3_date]['Low']
        p1_num, p2_num, p3_num = mdates.date2num([p1_date, p2_date, p3_date])
        log_p1, log_p2, log_p3 = np.log([p1_price, p2_price, p3_price])
        slope = (log_p2 - log_p1) / (p2_num - p1_num)
        intercept1, intercept2 = log_p1 - slope * p1_num, log_p3 - slope * p3_num
        return {'points': [(p1_date, p1_price), (p2_date, p2_price), (p3_date, p3_price)], 'slope': slope, 'intercept1': intercept1, 'intercept2': intercept2, 'width': abs(intercept1-intercept2)}

    def visualize_latest_strategy(self):
        market_regime = self.determine_market_regime()
        df_long_term = self.fetch_data(self.lt_tf, start_str='2020-01-01') # [수정] 분석 시작일 변경
        df_long_term.ta.ichimoku(append=True)
        df_short_term = df_long_term.resample(self.st_tf.upper()).last().ffill().dropna()
        
        lt_channels, lt_messages = self._analyze_channels(df_long_term, 0.3, 'all_time', self.lt_tf)
        st_channels, st_messages = self._analyze_channels(df_short_term, 0.05, 'recent', self.st_tf)

        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(15, 10), gridspec_kw={'height_ratios': [3, 1]})
        fig.suptitle(f'Macro/Micro Combined Strategy - Regime: {market_regime}', fontsize=16)

        ax1.plot(df_long_term.index, df_long_term['Close'], color='black', linewidth=1.0, label='Daily Close', zorder=10)
        ax1.set_yscale('log')
        ax1.set_ylabel('Price (USDT) - Log Scale')
        ax1.grid(which='major', linestyle='--')

        ax1.fill_between(df_long_term.index, df_long_term['ISA_9'], df_long_term['ISB_26'], where=df_long_term['ISA_9'] >= df_long_term['ISB_26'], color='lightgreen', alpha=0.3, label='Ichimoku Cloud (Bull)')
        ax1.fill_between(df_long_term.index, df_long_term['ISA_9'], df_long_term['ISB_26'], where=df_long_term['ISA_9'] < df_long_term['ISB_26'], color='lightcoral', alpha=0.3, label='Ichimoku Cloud (Bear)')

        future_dates = pd.date_range(start=df_long_term.index[-1], periods=90, freq='D')
        all_dates = df_long_term.index.union(future_dates)
        all_dates_num = mdates.date2num(all_dates)
        
        anchor_points = []
        
        channel_configs = {
            'lt_bull': (lt_channels, 'bullish', 'green', f'{self.lt_tf} Bullish'),
            'lt_bear': (lt_channels, 'bearish', 'red', f'{self.lt_tf} Bearish'),
            'st_bull': (st_channels, 'bullish', 'blue', f'{self.st_tf} Bullish')
        }

        for key, (channels, type, color, label) in channel_configs.items():
            if type in channels:
                ch = channels[type]
                line1, line2 = np.exp(ch['slope'] * all_dates_num + ch['intercept1']), np.exp(ch['slope'] * all_dates_num + ch['intercept2'])
                if 'lt' in key:
                    ax1.fill_between(all_dates, line1, line2, color=color, alpha=0.1, label=f'{label} Channel')
                    # [수정] 확장 채널 기능 복원
                    multiples = [0.5, 1.5, 2.0]
                    for m in multiples:
                        width_multiplier = -m if type == 'bullish' else m
                        exp_line1 = np.exp(np.log(line1) + ch['width'] * width_multiplier)
                        exp_line2 = np.exp(np.log(line2) + ch['width'] * width_multiplier)
                        ax1.plot(all_dates, exp_line1, ':', color=color, linewidth=0.7)
                else:
                    ax1.plot(all_dates, line1, '--', color=color, linewidth=1.2, label=f'{label} Channel')
                    ax1.plot(all_dates, line2, '--', color=color, linewidth=1.2)
                anchor_points.extend([(p[0], p[1], color) for p in ch['points']])
        
        ax2.bar(df_long_term.index, df_long_term['Volume'], color='gray', alpha=0.5, width=1.0)
        ax2.set_ylabel('Volume')
        ax2.set_yscale('log')

        anchor_dates_for_text = sorted(list(set((p[0], p[2]) for p in anchor_points)))
        y_pos, x_pos = 0.95, 0.01
        for i, (date, color) in enumerate(anchor_dates_for_text):
            if date in df_long_term.index:
                ax1.plot(date, df_long_term.loc[date]['Close'], 'x', color=color, markersize=6, markeredgewidth=2)
                volume_at_point = df_long_term.loc[date]['Volume']
                ax2.plot(date, volume_at_point, 'o', color=color, markersize=5)
                ax2.text(x_pos, y_pos, date.strftime("%Y/%m/%d"), transform=ax2.transAxes, fontsize=8, color=color, verticalalignment='top')
                y_pos -= 0.15
                if y_pos < 0.1: y_pos = 0.95; x_pos += 0.1
            
        handles, labels = ax1.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax1.legend(by_label.values(), by_label.keys(), loc='lower right', fontsize=8)
        
        status_text = "\n".join(lt_messages + st_messages)
        ax1.text(0.01, 0.98, status_text, transform=ax1.transAxes, fontsize=8, verticalalignment='top', horizontalalignment='left',
                 bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='black', alpha=0.7))

        # [수정] X, Y축 여백 추가
        ax1.set_xlim(left=df_long_term.index[0] - timedelta(days=30), right=all_dates[-1])
        ax1.set_ylim(bottom=df_long_term['Low'].min()*0.7, top=df_long_term['High'].max()*1.3)
        plt.show()

if __name__ == '__main__':
    btc_bot = TradingBot(symbol='BTCUSDT', lt_tf='1d', st_tf='4h')
    btc_bot.visualize_latest_strategy()