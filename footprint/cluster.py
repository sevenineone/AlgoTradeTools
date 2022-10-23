import pandas as pd
import numpy as np
from bokeh.plotting import figure, output_file, show
from bokeh.models import ColumnDataSource, LabelSet
from bokeh.models import FuncTickFormatter


def get_nearest_value(K, lst):
    return lst[min(range(len(lst)), key = lambda i: abs(lst[i]-K))]


def round_step(num, step):
    return round(round(num / step) * step, 2)


class Cluster:
    def __init__(self, plot_price_step=0.25, split_range=10):
        ###########
        self._data = pd.DataFrame
        self._plot_data = pd.DataFrame
        self._agg_data = pd.DataFrame
        ###########
        self._timeframe = '1min'
        self._price_step = plot_price_step
        self._split_range = split_range

    def set_data(self, path):
        self._data = pd.read_csv(path)

    def get_data(self):
        return self._data

    def get_plot_data(self):
        return self._plot_data

    def get_agg_data(self):
        return self._agg_data

    def aggregate(self):
        self._data['transact_time'] = pd.to_datetime(self._data['transact_time'], unit='ms')
        self._data['transact_time'] = self._data['transact_time'].dt.floor(self._timeframe)
        datetimes = pd.unique(self._data["transact_time"])
        self._data = pd.concat(
            [self._data['price'], self._data['quantity'], self._data['transact_time'], self._data['is_buyer_maker']],
            axis=1,
            keys=['price', 'quantity', 'transact_time', 'is_buyer_maker'])

        ######################################### for plotting
        self._plot_data = self._data

        clusters = pd.DataFrame(columns=['price', 'is_buyer_maker', 'quantity', 'transact_time'])
        self._plot_data = self._plot_data.apply(lambda x: round_step(x, self._price_step) if x.name == 'price' else x)

        for datetime in datetimes:
            cluster = self._plot_data.loc[self._plot_data["transact_time"] == datetime]
            cluster = cluster.groupby(['price', 'is_buyer_maker'], as_index=False)['quantity'].sum()
            cluster = cluster.join(pd.Series(data=[datetime] * cluster.shape[0], name='transact_time'))
            clusters = pd.concat([clusters, cluster], ignore_index=True)
        self._plot_data = clusters

        ######################################### for dataset
        self._agg_data = self._data

        clusters = pd.DataFrame(columns=['price', 'is_buyer_maker', 'quantity', 'transact_time'])

        for datetime in datetimes:
            cluster = self._agg_data.loc[self._agg_data["transact_time"] == datetime]
            price_range = np.linspace(cluster['price'].min(), cluster['price'].max(), num=self._split_range)
            cluster['price'] = cluster['price'].apply(lambda x: get_nearest_value(x, price_range.tolist()))

            start_cluster = pd.DataFrame(np.array(
                [price_range, [True] * self._split_range, [0] * self._split_range, [datetime] * self._split_range]).T
                                         , columns=['price', 'is_buyer_maker', 'quantity', 'transact_time'])
            cluster = pd.concat([start_cluster, cluster], ignore_index=True)
            start_cluster = pd.DataFrame(np.array(
                [price_range, [False] * self._split_range, [0] * self._split_range, [datetime] * self._split_range]).T
                                         , columns=['price', 'is_buyer_maker', 'quantity', 'transact_time'])
            cluster = pd.concat([start_cluster, cluster], ignore_index=True)
            #print(cluster.head())

            cluster = cluster.groupby(['price', 'is_buyer_maker'], as_index=False)['quantity'].sum()
            cluster = cluster.join(pd.Series(data=[datetime] * cluster.shape[0], name='transact_time'))
            clusters = pd.concat([clusters, cluster], ignore_index=True)
        self._agg_data = clusters

    def plot_footprint_chart(self, plot_file_name='my_footprint_chart'):
        x_interval = 10
        side = 4
        green = []
        red = []
        # labels_data = []
        ####################################### aggregation

        min_price = self._plot_data['price'].min()
        max_price = self._plot_data['price'].max()

        price_range = self._price_step * np.arange(min_price / self._price_step,
                                                   (max_price + self._price_step) / self._price_step)

        with np.nditer(price_range, op_flags=['readwrite']) as it:
            for x in it:
                x[...] = np.round(x, 2)
        price_rank = range(len(price_range))
        price_coord = dict(zip(price_range, price_rank))
        y_axis = dict(zip(price_rank, price_range))

        unique_datetimes = self._plot_data['transact_time'].unique()
        array = np.array(unique_datetimes)
        order = array.argsort()
        datetime_rank = order.argsort().tolist()

        datetime_coord = dict(zip(unique_datetimes, datetime_rank))
        x_axis_datetimes = [pd.to_datetime(str(x)) for x in unique_datetimes]
        x_axis_datetimes = [x.strftime('%H:%M') for x in x_axis_datetimes]  # x_axis datetime pattern
        x_axis = dict(zip([x * x_interval for x in datetime_rank], x_axis_datetimes))
        ####################################### plotting

        plot = figure(width=1350, height=600, x_range=(0, 50), y_range=(0, 25))
        plot.xaxis.axis_label_text_font_size = "15pt"
        for datetime in unique_datetimes:
            cluster = self._plot_data.loc[self._plot_data["transact_time"] == datetime]
            buy_max = cluster.loc[cluster['is_buyer_maker'] == True]['quantity'].max()
            sell_max = cluster.loc[cluster['is_buyer_maker'] == False]['quantity'].max()
            for index, row in cluster.iterrows():
                x = datetime_coord[datetime] * x_interval
                y = price_coord[round(row['price'], 2)]
                quantity = row['quantity']
                if row['is_buyer_maker']:
                    green.append([[y + 1], [y], [x], [x + (quantity / (buy_max + 0.00001)) * side]])
                    # labels_data.append([x + 0.2, y + 0.2, str(round(quantity, 2))])
                else:
                    red.append([[y + 1], [y], [x], [x - (quantity / (sell_max + 0.00001)) * side]])
                    # labels_data.append([x - 3, y + 0.2, str(round(quantity, 2))])

        green = [*zip(*green)]
        red = [*zip(*red)]
        # labels_data = [*zip(*labels_data)]
        plot.quad(top=green[0], bottom=green[1], left=green[2], right=green[3], color='green', line_color='black',
                  legend_label='buy')
        plot.quad(top=red[0], bottom=red[1], left=red[2], right=red[3], color='red', line_color='black',
                  legend_label='sell')

        # source = ColumnDataSource(data=dict(height=labels_data[1],
        #                                    weight=labels_data[0],
        #                                    names=labels_data[2]))

        # labels = LabelSet(x='weight', y='height', text='names', text_font_size='10pt', text_color='black',
        #              x_offset=3, y_offset=3, source=source)

        # plot.add_layout(labels)

        plot.yaxis.formatter = FuncTickFormatter(code=f"""
            return {min_price} + (tick * {self._price_step});
        """)

        plot.xaxis.ticker = [x * x_interval for x in datetime_rank]
        plot.xaxis.major_label_overrides = x_axis

        output_file(f'{plot_file_name}.html')
        show(plot)
