import pandas as pd
import ta
import ccxt
import plotly.graph_objects as go
from typing import List
from plotly.subplots import make_subplots


class Order:
    """
    ccxt has more class fields
    """

    def __init__(self, id, datetime, symbol, type, side, average, amount, info=""):
        self.id = id
        self.datetime = datetime
        self.datetime_execute = datetime  # CCXT Order DON'T HAVE THIS VARIABLE
        self.symbol = symbol
        self.type = type  # 'limit', 'market'
        self.side = side  # 'buy', 'sell'
        self.average = average  # price
        self.amount = amount
        self.info = info

    def set_datetime_execute(self, datetime_execute):
        self.datetime_execute = datetime_execute


class Trades:
    def __init__(self):
        self.open_order: List[Order,] = []  # [Order, ] orders are open but not executed
        self.open_trade: List[Order,] = []  # [Order, ] open position
        self.close_trade: List[List[Order,]] = []  # [(Order, Order), ] two opposite order

    def get_open_trade_amount(self):
        amount = 0
        for trade in self.open_trade:
            if trade.side == 'buy':
                amount += trade.amount
            else:
                amount -= trade.amount
        return abs(amount)

    def get_open_trade_side(self):
        if len(self.open_trade) > 0:
            return self.open_trade[0].side
        return None


class Torgash:
    def __init__(self):
        self.current_base_balance = 1000
        self.base_symbol = 'usdt'
        self.current_trading_balance = 0
        self.trading_symbol = 'btc'
        self.symbol = "BTCUSDT"
        self.trading_fee_multiplier = 1 - 0.0002
        self.min_order_threshold = 5  # equal 5usdt for binance futures
        self.min_order_step = 0.001  # min step for trading_symbol (btc) 0.001btc binance futures
        ###################################
        self.transaction_type_hist = []  # надо убрать
        self.transaction_datetime = []  # надо убрать
        self.transaction_price_hist = []
        self.transaction_balance_hist = []
        self.trades = Trades()
        ###################################
        self.base_balance_hist = []
        self.trading_balance_hist = []
        ###################################
        self.data = None
        self._current_datetime = None
        self._current_price = 0
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

    def execute_order(self, order: Order):
        if not self.trades.open_trade:  # if there are no open trade
            self.trades.open_trade.append(order)
        elif self.trades.get_open_trade_side() == order.side:  # if the direction is the same
            self.trades.open_trade.append(order)
        else:  # orders are sent in different directions
            if self.trades.get_open_trade_amount() == order.amount:
                self.trades.open_trade.append(order)
                self.trades.close_trade.append(self.trades.open_trade)
                self.trades.open_trade = []
            elif self.trades.get_open_trade_amount() > order.amount:
                self.trades.open_trade.append(order)
            else:  # open trade amount less than new order amount
                raise ValueError("Incorrect order amount")

    def limits_check(self):
        for i in range(len(self.trades.open_order)):
            if self.trades.open_order[i].side == 'buy':
                # if buy but limit price > self._current_price (make market order) !!!!!

                if self.data[self._current_datetime].Low < self.trades.open_order[i].average:
                    order = self.trades.open_order.pop(i)
                    order.set_datetime_execute(self._current_datetime)
                    self.execute_order(order)
            elif self.trades.open_order[i].side == 'sell':
                # if sell but limit price < self._current_price (make market order) !!!!!

                if self.data[self._current_datetime].High > self.trades.open_order[i].average:
                    order = self.trades.open_order.pop(i)
                    order.set_datetime_execute(self._current_datetime)
                    self.execute_order(order)

    def createOrder(self, symbol, type, side, amount, price=None, params={}):
        """
        :param symbol: (String) required Unified CCXT market symbol
        :param type: "market" "limit" see #custom-order-params and #other-order-types for non-unified types
        :param side: "buy" | "sell"
        :param amount: how much of currency you want to trade usually, but not always, in units of the base currency
               of the trading pair symbol (the units for some exchanges are dependent on the side of the order:
               see their API docs for details.)
        :param price: the price at which the order is to be fullfilled at in units of the quote currency
               (ignored in market orders)
        :param params: (Dictionary) Extra parameters specific to the exchange API endpoint (e.g. {"settle": "usdt"})
        :return: A successful order call returns a order

        ex:
        mk.create_order(symbol='XRP/BTC', type='stop_loss_limit', side='sell', amount=1,
                        price=0.6, params={'stopPrice':0.60})
        """

        order = Order(0, self._current_datetime, symbol, type, side, average=price, amount=amount)
        if type == "limit":
            self.trades.open_order.append(order)
        elif type == "market":
            self.execute_order(order)
        else:
            raise ValueError("Bad type of transaction")

    def createLimitBuyOrder(self, symbol, amount, price, params={}):
        """
        How in ccxt

        :return:
        """
        return self.createOrder(symbol=symbol, type="limit", side="buy", amount=amount, price=price, params=params)

    def createLimitSellOrder(self, symbol, amount, price, params={}):
        """
        How in ccxt

        :return:
        """
        return self.createOrder(symbol=symbol, type="limit", side="sell", amount=amount, price=price, params=params)

    def createMarketOrder(self, symbol, side, amount, params={}):
        return self.createOrder(symbol=symbol, type="market", side=side, amount=amount, params=params)

    def fetchOpenOrders(self, ):  # вывод id всех открытых ордеров
        """
        How in ccxt

        :return:
        """
        pass

    def fetchClosedOrders(self):
        """
        How in ccxt

        :return:
        """
        pass

    def fetchPosition(self, symbol, params={}):
        """

        :param symbol: (String) required Unified CCXT market symbol ex:"BTC|USDT"
        :param params:
        :return:
        """
        pass

    def cancelOrder(self, id, symbol, params={}):
        """
        How in ccxt

        :param id:
        :param symbol:
        :param params:
        :return:
        """
        pass

    def buy(self, amount=-1., ):  # надо убрать
        if amount < 0:
            amount = self.current_base_balance
        if self.current_base_balance >= amount >= self.min_order_threshold:
            self.current_base_balance -= amount
            self.current_trading_balance += (amount / self._current_price) * self.trading_fee_multiplier
            ###
            self.transaction_datetime.append(self.current_datetime)
            self.transaction_type_hist.append('buy')
            self.transaction_balance_hist.append(
                self.current_base_balance + (self.current_trading_balance * self.current_price))
            self.transaction_price_hist.append(self.current_price)
            ###
            print(f"|| {self._current_datetime} | Buy | {amount} {self.base_symbol} --> "
                  f"{(amount / self._current_price) * self.trading_fee_multiplier} {self.trading_symbol} || "
                  f"Balance: {self.current_base_balance} {self.base_symbol} | "
                  f"{self.current_trading_balance} {self.trading_symbol} ||")
            self.print_ind()  # xtra output for necessary indicators
        else:
            print("Buy attempt, but not enough money.")

    def sell(self, amount=-1., ):  # надо убрать
        if amount < 0:
            amount = self.current_trading_balance
        if self.current_trading_balance >= amount >= self.min_order_threshold:
            self.current_trading_balance -= amount
            self.current_base_balance += (amount * self._current_price) * self.trading_fee_multiplier
            ###
            self.transaction_datetime.append(self.current_datetime)
            self.transaction_type_hist.append('buy')
            self.transaction_balance_hist.append(
                self.current_base_balance + (self.current_trading_balance * self.current_price))
            self.transaction_price_hist.append(self.current_price)
            ###
            print(f"|| {self._current_datetime} | Sell | {amount} {self.trading_symbol} --> "
                  f"{(amount * self._current_price) * self.trading_fee_multiplier} {self.base_symbol} || "
                  f"Balance: {self.current_base_balance} {self.base_symbol} | "
                  f"{self.current_trading_balance} {self.trading_symbol} ||")
            self.print_ind()  # xtra output for necessary indicators
        else:
            print("Sell attempt, but not enough money.")

    def print_ind(self):  # put indicators u need (ex: )
        print("")

    def step_strategy(self, datetime):
        pass

    def run_step_strategy(self):
        self.base_balance_hist.append(self.current_base_balance)
        for datetime, value in self.data.iterrows():
            self._current_datetime = datetime
            self._current_price = self.data[datetime].Open
            self.step_strategy(datetime)
            self.limits_check()

            #  self.base_balance_hist.append(self.current_base_balance)
            #  self.trading_balance_hist.append(self.current_trading_balance)

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
            self._current_datetime = datetime
            self._current_price = value.Close
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
        transaction_balance_df = pd.DataFrame(
            [self.transaction_datetime, self.transaction_balance_hist, self.transaction_price_hist,
             self.transaction_type_hist]).T
        fig = make_subplots(rows=2, cols=1, row_width=[0.3, 0.7], shared_xaxes=True)
        for i in range(from_candle, to_candle, count_candlestick_per_plot):
            data_interval = self.data[i:i + count_candlestick_per_plot]
            transaction_balance_df_interval = transaction_balance_df[
                transaction_balance_df[0].between(data_interval.index.values[0], data_interval.index.values[-1])]
            ############################################
            fig.add_trace(go.Candlestick(x=data_interval.index.values,
                                         open=data_interval['Open'], high=data_interval['High'],
                                         low=data_interval['Low'], close=data_interval['Close'],
                                         name=f'{self.base_symbol}/{self.trading_symbol}',
                                         hovertext=self.data.Difference,
                                         increasing_line_color='grey', decreasing_line_color='black', visible=False),
                          row=1, col=1)

            fig.add_trace(
                go.Scatter(x=transaction_balance_df_interval[0], y=transaction_balance_df_interval[1], name='balance',
                           visible=False),
                row=2, col=1)
            fig.add_trace(go.Scatter(mode='markers',
                                     x=transaction_balance_df_interval[transaction_balance_df_interval[3] == 'buy'][0],
                                     y=transaction_balance_df_interval[transaction_balance_df_interval[3] == 'buy'][2],
                                     marker_symbol=45,
                                     marker_color='green',
                                     marker_size=10, name='buy', visible=False), row=1, col=1)
            fig.add_trace(go.Scatter(mode='markers',
                                     x=transaction_balance_df_interval[transaction_balance_df_interval[3] == 'sell'][0],
                                     y=transaction_balance_df_interval[transaction_balance_df_interval[3] == 'sell'][2],
                                     marker_symbol=46,
                                     marker_color='red',
                                     marker_size=10, name='sell', visible=False), row=1, col=1)

            #################################
        fig.data[0].visible = True
        fig.data[1].visible = True
        fig.data[2].visible = True
        fig.data[3].visible = True
        #  Create slider
        steps = []
        for i in range(0, len(fig.data), 4):
            step = dict(
                method="update",
                args=[{"visible": [False] * len(fig.data)},
                      {"title": "Title: " + str(i)}],  # layout attribute
            )
            step["args"][0]["visible"][i] = True  # Toggle i'th trace to "visible"
            step["args"][0]["visible"][i + 1] = True
            step["args"][0]["visible"][i + 2] = True
            step["args"][0]["visible"][i + 3] = True
            steps.append(step)

        sliders = [dict(
            active=0,
            currentvalue={"prefix": "Date: "},
            pad={"t": 30},
            steps=steps
        )]

        fig.update_layout(xaxis_rangeslider_visible=False, height=800, sliders=sliders)
        fig.show()
