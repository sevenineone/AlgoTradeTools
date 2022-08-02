import re

import pandas as pd
import ta
import plotly.graph_objects as go
from typing import List
from plotly.subplots import make_subplots


class Order:
    """
    ccxt has more class fields
    """

    def __init__(self, id, datetime, symbol, type, side, average, amount, fee, info=""):
        self.id = id
        self.datetime = datetime
        self.datetime_execute = datetime  # CCXT Order DON'T HAVE THIS VARIABLE
        self.symbol = symbol
        self.type = type  # 'limit', 'market'
        self.side = side  # 'buy', 'sell'
        self.average = average  # price
        self.amount = amount
        self.info = info

        self.fee = fee  # in USDT

    def set_datetime_execute(self, datetime_execute):
        self.datetime_execute = datetime_execute


class Trades:
    def __init__(self):
        self.open_order: List[Order,] = []  # [Order, ] orders are open but not executed
        self.open_trade: List[Order,] = []  # [Order, ] open position
        self.close_trades: List[List[Order,]] = []  # [(Order, Order), ] two opposite order

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

    def calculate_trade_profit(self, number=-1):  # return profit for one close trade
        cost_long = 0
        cost_short = 0
        for trade in self.close_trades[number]:
            if trade.side == "buy":
                cost_long += trade.average * trade.amount + trade.fee
            elif trade.side == "sell":
                cost_short += trade.average * trade.amount - trade.fee
        return cost_short - cost_long

    def get_all_close_order(self):
        return [[item.datetime, item.average, item.side, item.amount] for i in self.close_trades for item in i]


class Torgash:
    def __init__(self):
        self.start_balance = 1000
        self.symbol = "ETH/USDT"  # always use /
        self.trading_market_fee_multiplier = 0.0004
        self.trading_limit_fee_multiplier = 0.0002
        self.trading_fee_multiplier = 1 - self.trading_market_fee_multiplier  # не использовать, удалить
        self.min_order_threshold = 5  # equal 5usdt for binance futures
        self.min_order_step = 0.001  # min step for trading_symbol (btc) 0.001btc binance futures
        ###################################
        self.trades = Trades()
        ###################################
        self.data = None
        self._current_datetime = None
        self._current_price = 0
        ###################################
        self.balance_hist = []
        self.datetime_balance_hist = []
        ###################################

    def set_data(self, ohlcv_df: pd.DataFrame, date_time=0, open=1, high=2, low=3, close=4, volume=5, symbol=""):
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

        if self.data.isnull().values.any():
            raise ValueError("Corrupted data (Nan).")

        if re.fullmatch('''[a-zA-Z]+/[a-zA-Z]''', symbol):
            self.symbol = symbol

        # calculate indicators to dataframe
        self.data = self.data.join(ta.trend.sma_indicator(self.data.Close, window=50))
        self.data = self.data.join(ta.trend.sma_indicator(self.data.Close, window=100))
        self.data.columns = ['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume', 'sma_fast',
                             'sma_slow']  # name the columns the way you want
        #####################################################
        self.data.set_index('Datetime', inplace=True)

    def set_data_from_csv(self, filename: str, date_time=0, open=1, high=2, low=3, close=4, volume=5):
        ohlcv_df = pd.read_csv(filename)
        symbol = f"{filename.split('_')[-3].split('/')[-1]}/{filename.split('_')[-2]}"
        self.set_data(ohlcv_df, date_time=date_time, open=open, high=high,
                      low=low, close=close, volume=volume, symbol=symbol)

    def execute_order(self, order: Order):
        if not self.trades.open_trade:  # if there are no open trade
            self.trades.open_trade.append(order)
        elif self.trades.get_open_trade_side() == order.side:  # if the direction is the same
            self.trades.open_trade.append(order)
        else:  # orders are sent in different directions
            if self.trades.get_open_trade_amount() == order.amount:
                self.trades.open_trade.append(order)
                self.trades.close_trades.append(self.trades.open_trade)
                self.trades.open_trade = []
                self.balance_hist.append(self.balance_hist[-1] + self.trades.calculate_trade_profit(-1))
                self.datetime_balance_hist.append(self._current_datetime)
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

    def calculate_order_amount(self, percent=100, usdt=100, based_in_percent=True):
        # open_position_cost = self.trades.get_open_trade_amount() * self._current_price  # open position price

        cost = 0
        if based_in_percent:
            cost = (self.balance_hist[-1] / 100 * percent)
        else:
            cost = usdt
        return cost / self._current_price // self.min_order_step * self.min_order_step  # amount

    def createOrder(self, type, side, amount, symbol="", price=None, params={}):
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

        if symbol == "":
            symbol = self.symbol

        if type == "limit":
            if amount * self._current_price < self.min_order_threshold:
                raise ValueError(f"Order price less then {self.min_order_threshold}$")
            fee = self.trading_limit_fee_multiplier * amount * price
            order = Order(0, self._current_datetime, symbol, type, side, average=price, amount=amount, fee=fee)
            self.trades.open_order.append(order)
        elif type == "market":
            if amount * self._current_price < self.min_order_threshold:
                raise ValueError(f"Order price less then {self.min_order_threshold}$")
            price = self._current_price
            fee = self.trading_market_fee_multiplier * amount * price
            order = Order(0, self._current_datetime, symbol, type, side, average=price, amount=amount, fee=fee)
            print(f"{side} {amount} {symbol.split('/')[0]} | Price: {price} {symbol.split('/')[1]}")
            self.execute_order(order)
        else:
            raise ValueError("Bad type of transaction")

    def createLimitBuyOrder(self, amount, price, symbol="", params={}):
        """
        How in ccxt

        :return:
        """
        return self.createOrder(symbol=symbol, type="limit", side="buy", amount=amount, price=price, params=params)

    def createLimitSellOrder(self, amount, price, symbol="", params={}):
        """
        How in ccxt

        :return:
        """
        return self.createOrder(symbol=symbol, type="limit", side="sell", amount=amount, price=price, params=params)

    def createMarketOrder(self, side, amount, symbol="", params={}):
        return self.createOrder(symbol=symbol, type="market", side=side, amount=amount, params=params)

    def cancelOrder(self, id, symbol, params={}):
        """
        How in ccxt

        :param id:
        :param symbol:
        :param params:
        :return:
        """
        pass

    def step_strategy(self, datetime):
        pass

    def run_step_strategy(self):
        # self.balance_hist.append(self.current_base_balance)
        for datetime, value in self.data.iterrows():
            self._current_datetime = datetime
            self._current_price = self.data[datetime].Open
            self.step_strategy(datetime)
            self.limits_check()

            #  self.balance_hist.append(self.current_base_balance)
            #  self.trading_balance_hist.append(self.current_trading_balance)

    def run_strategy(self):
        # We cut the top of the data, because first 100 sma values are not calculated there.
        self.data = self.data.iloc[99:]
        # xtra variables
        fast_above_slow = self.data.sma_fast[self.data.head(1).index[0]] > self.data.sma_slow[
            self.data.head(1).index[0]]
        #####################################################
        self.balance_hist.append(self.start_balance)
        # self.trading_balance_hist.append(self.current_trading_balance)
        for datetime, value in self.data.iterrows():
            self._current_datetime = datetime
            self._current_price = value.Close
            # There you should implement your strategy
            if fast_above_slow and (self.data.sma_fast[datetime] < self.data.sma_slow[datetime]):
                self.createMarketOrder("sell", 0.02)
            elif not fast_above_slow and self.data.sma_fast[datetime] > self.data.sma_slow[datetime]:
                self.createMarketOrder("buy", 0.02)
            fast_above_slow = self.data.sma_fast[datetime] > self.data.sma_slow[datetime]

    def plot_candlestick(self, count_candlestick_per_plot=10000, from_candle=0, to_candle=0):
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

    def plot(self, count_candlestick_per_plot=10000, from_candle=0, to_candle=0):  # Надо подписи на графике поменять
        """ displays count_candlestick_per_plot candlestick at one time on one chart """
        if to_candle == 0 or to_candle > len(self.data):
            to_candle = len(self.data)

        transaction_df = pd.DataFrame(data=self.trades.get_all_close_order(),
                                      columns=['datetime', 'average', 'side', 'amount'])
        balance_df = pd.DataFrame([self.datetime_balance_hist, self.balance_hist]).T
        fig = make_subplots(rows=2, cols=1, row_width=[0.3, 0.7], shared_xaxes=True)
        for i in range(from_candle, to_candle, count_candlestick_per_plot):
            data_interval = self.data[i:i + count_candlestick_per_plot]
            transaction_df_interval = transaction_df[
                transaction_df['datetime'].between(data_interval.index.values[0], data_interval.index.values[-1])]
            balance_df_interval = balance_df[
                balance_df[0].between(data_interval.index.values[0], data_interval.index.values[-1])]
            ############################################
            fig.add_trace(go.Candlestick(x=data_interval.index.values,
                                         open=data_interval['Open'], high=data_interval['High'],
                                         low=data_interval['Low'], close=data_interval['Close'],
                                         name=self.symbol,
                                         increasing_line_color='grey', decreasing_line_color='black', visible=False),
                          row=1, col=1)

            fig.add_trace(
                go.Scatter(x=balance_df_interval[0], y=balance_df_interval[1], name='balance', mode='lines+markers',
                           visible=False),
                row=2, col=1)
            fig.add_trace(go.Scatter(mode='markers',
                                     x=transaction_df_interval[transaction_df_interval['side'] == 'buy']['datetime'],
                                     y=transaction_df_interval[transaction_df_interval['side'] == 'buy']['average'],
                                     marker_symbol=45,
                                     marker_color='green',
                                     hovertext=transaction_df_interval[transaction_df_interval['side'] == 'sell'][
                                         'amount'],
                                     marker_size=10, name='buy', visible=False), row=1, col=1)
            fig.add_trace(go.Scatter(mode='markers',
                                     x=transaction_df_interval[transaction_df_interval['side'] == 'sell']['datetime'],
                                     y=transaction_df_interval[transaction_df_interval['side'] == 'sell']['average'],
                                     marker_symbol=46,
                                     marker_color='red',
                                     hovertext=transaction_df_interval[transaction_df_interval['side'] == 'sell'][
                                         'amount'],
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
