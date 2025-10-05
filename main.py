# main.py

from bot import TradingBot

# 새로운 파라미터인 lt_tf(장기), st_tf(단기)를 사용
btc_bot = TradingBot(symbol='BTCUSDT', lt_tf='1d', st_tf='4h')

btc_bot.visualize_latest_strategy()