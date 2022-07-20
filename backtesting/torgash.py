import pandas as pd
import ta
import plotly.graph_objects as go


class Torgash:
    def __init__(self):
        self.current_base_balance = 90000000
        self.base_symbol = 'usdt'
        self.current_trading_balance = 0
        self.trading_symbol = 'btc'
        self.trading_fee_multiplier = 1 - 0.0002
        self.min_order_threshold = 5  # equal 5usdt for binance futures
        self.min_order_step = 0.001   # min step for trading_symbol (btc) 0.001btc binance futures
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
        # calculate indicators to dataframe
        # self.data = self.data.join(ta.trend.sma_indicator(self.data.Close, window=50))
        # self.data = self.data.join(ta.trend.sma_indicator(self.data.Close, window=100))
        self.data.columns = ['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']  # name the columns the way you want
        #####################################################
        self.data.set_index('Datetime', inplace=True)

    def set_data_from_csv(self, filename: str, date_time=0, open=1, high=2, low=3, close=4, volume=5):
        ohlcv_df = pd.read_csv(filename)
        self.set_data(ohlcv_df, date_time=date_time, open=open, high=high, low=low, close=close, volume=volume)

    def buy(self, amount=-1., ):
        if amount < 0:
            amount = self.current_base_balance
        if self.current_base_balance >= amount >= self.min_order_threshold:
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
            self.print_ind()  # xtra output for necessary indicators
        else:
            print("Buy attempt, but not enough money.")

    def sell(self, amount=-1., ):
        if amount < 0:
            amount = self.current_trading_balance
        if self.current_trading_balance >= amount >= self.min_order_threshold:
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
            self.print_ind()  # xtra output for necessary indicators
        else:
            print("Sell attempt, but not enough money.")

    def print_ind(self):  # put indicators u need (ex: )
        print("")

    def run_strategy(self):
        # We cut the top of the data, because first 100 sma values are not calculated there.
        self.data = self.data.iloc[99:]
        # xtra variables
        fast_above_slow = self.data.sma_fast[self.data.head(1).index[0]] > self.data.sma_slow[
            self.data.head(1).index[0]]
        #####################################################
        self.base_balance_hist.append(self.current_base_balance)
        self.trading_balance_hist.append(self.current_trading_balance)
        for datetime, value in self.data.iterrows():
            self.current_datetime = datetime
            self.current_price = value.Close
            # There you should implement your strategy
            if fast_above_slow and self.data.sma_fast[datetime] < self.data.sma_slow[datetime]:
                self.sell()
            elif not fast_above_slow and self.data.sma_fast[datetime] > self.data.sma_slow[datetime]:
                self.buy()
            fast_above_slow = self.data.sma_fast[datetime] > self.data.sma_slow[datetime]
            #####################################################
            self.base_balance_hist.append(self.current_base_balance)
            self.trading_balance_hist.append(self.current_trading_balance)

    def plot(self, count_candlestick_per_plot=10000, from_candle=0, to_candle=0):  # Надо подписи на графике поменять
        """ displays count_candlestick_per_plot candlestick at one time on one chart """
        if to_candle == 0 or to_candle > len(self.data):
            to_candle = len(self.data)

        fig = go.Figure()
        for i in range(from_candle, to_candle, count_candlestick_per_plot):
            data_interval = self.data[i:i + count_candlestick_per_plot]
            fig.add_trace(
                go.Candlestick(visible=False,
                               x=data_interval.index.values,
                               open=data_interval['Open'], high=data_interval['High'],
                               low=data_interval['Low'], close=data_interval['Close'])
            )
        fig.data[0].visible = True

        #  Create slider
        steps = []
        for i in range(len(fig.data)):
            step = dict(
                method="update",
                args=[{"visible": [False] * len(fig.data)},
                      {"title": "Title: " + str(i)}],  # layout attribute
            )
            step["args"][0]["visible"][i] = True  # Toggle i'th trace to "visible"
            steps.append(step)

        sliders = [dict(
            active=0,
            currentvalue={"prefix": "Date: "},
            pad={"t": 30},
            steps=steps
        )]

        fig.update_layout(xaxis_rangeslider_visible=False, sliders=sliders)
        fig.show()
