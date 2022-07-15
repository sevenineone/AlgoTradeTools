import pandas as pd
import ta


class Torgash:
    def __init__(self):
        self.current_base_balance = 90000000
        self.base_symbol = 'usdt'
        self.current_trading_balance = 0
        self.trading_symbol = 'btc'
        self.trading_fee_multiplier = 1 - 0.0002
        self.min_bid_threshold = 0
        self.min_ask_threshold = 0
        ###################################
        self.date_time = []
        self.open = []
        self.high = []
        self.low = []
        self.close = []
        self.volume = []
        ###################################
        self.transaction_type_hist = []
        self.transaction_amount_hist = []
        self.transaction_datetime = []
        ###################################
        self.base_balance_hist = []
        self.trading_balance_hist = []
        ###################################
        self.data = None
        self.current_datetime = None
        self.current_price = 0
        ###################################

    def set_base_symbol(self, symbol):
        self.base_symbol = symbol


    def set_data(self, ohlcv_df: pd.DataFrame, date_time=0, open=1, high=2, low=3, close=4, volume=5):
        self.data = pd.DataFrame([
            ohlcv_df[ohlcv_df.columns[date_time]],
            ohlcv_df[ohlcv_df.columns[open]],
            ohlcv_df[ohlcv_df.columns[high]],
            ohlcv_df[ohlcv_df.columns[low]],
            ohlcv_df[ohlcv_df.columns[close]],
            ohlcv_df[ohlcv_df.columns[volume]]]).T
        self.data.columns = ['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']
        try:
            self.data['Datetime'] = pd.to_datetime(self.data['Datetime'], unit='ms')
        except:
            raise ValueError("Datetime column format must be unix or string format.")
        self.data = ta.add_all_ta_features(self.data, 'Open', 'High', 'Low', 'Close', 'Volume', fillna=True)
        self.data.set_index('Datetime', inplace=True)

    def set_data_from_csv(self, filename: str, date_time=0, open=1, high=2, low=3, close=4, volume=5):
        ohlcv_df = pd.read_csv(filename)
        self.set_data(ohlcv_df, date_time=date_time, open=open, high=high, low=low, close=close, volume=volume)

    def buy(self, amount=-1., ):
        if amount < 0:
            amount = self.current_base_balance
        if self.current_base_balance >= amount >= self.min_bid_threshold:
            self.current_base_balance -= amount
            self.current_trading_balance += (amount / self.current_price) * self.trading_fee_multiplier
            ###
            self.transaction_datetime.append(self.current_datetime)
            self.transaction_type_hist.append('buy')
            self.transaction_amount_hist.append(amount)
            ###
            print(f"|| {self.current_datetime} | Buy | {amount} {self.base_symbol} --> "
                  f"{(amount / self.current_price) * self.trading_fee_multiplier} {self.trading_symbol} || "
                  f"Balance: {self.current_base_balance} {self.base_symbol} | "
                  f"{self.current_trading_balance} {self.trading_symbol} ||")
        else:
            print("Buy attempt, but not enough money.")

    def sell(self, amount=-1., ):
        if amount < 0:
            amount = self.current_trading_balance
        if self.current_trading_balance >= amount >= self.min_ask_threshold:
            self.current_trading_balance -= amount
            self.current_base_balance += (amount * self.current_price) * self.trading_fee_multiplier
            ###
            self.transaction_datetime.append(self.current_datetime)
            self.transaction_type_hist.append('buy')
            self.transaction_amount_hist.append(amount)
            ###
            print(f"|| {self.current_datetime} | Sell | {amount} {self.trading_symbol} --> "
                  f"{(amount * self.current_price) * self.trading_fee_multiplier} {self.base_symbol} || "
                  f"Balance: {self.current_base_balance} {self.base_symbol} | "
                  f"{self.current_trading_balance} {self.trading_symbol} ||")
        else:
            print("Sell attempt, but not enough money.")


    def run_strategy(self):
        #####################################################
        fast_above_slow = self.data.trend_sma_fast[self.data.head(1).index[0]] > self.data.trend_sma_slow[self.data.head(1).index[0]]
        #####################################################
        self.base_balance_hist.append(self.current_base_balance)
        self.trading_balance_hist.append(self.current_trading_balance)
        for datetime, value in self.data.iterrows():
            self.current_datetime = datetime
            self.current_price = value.Close
            #####################################################
            if fast_above_slow and self.data.trend_sma_fast[datetime] < self.data.trend_sma_slow[datetime]:
                self.sell()
            elif not fast_above_slow and self.data.trend_sma_fast[datetime] > self.data.trend_sma_slow[datetime]:
                self.buy()
            fast_above_slow = self.data.trend_sma_fast[datetime] > self.data.trend_sma_slow[datetime]
            #####################################################
            self.base_balance_hist.append(self.current_base_balance)
            self.trading_balance_hist.append(self.current_trading_balance)
