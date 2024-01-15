from uuid import UUID
from datetime import datetime, timezone, date, timedelta
from alpaca.trading.enums import OrderSide, TimeInForce, AssetClass, AssetExchange, PositionSide, OrderClass, \
OrderType, OrderStatus, AssetStatus, AccountStatus, ActivityType, TradeActivityType, NonTradeActivityStatus, \
CorporateActionType, CorporateActionSubType, CorporateActionDateType, DTBPCheck, PDTCheck, TradeConfirmationEmail
import json
from alpaca.data.requests import StockBarsRequest, StockQuotesRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from pathlib import Path
import csv
import pandas as pd
import time

alpaca_enums = (
                OrderSide, TimeInForce, AssetClass, AssetExchange, PositionSide, OrderClass,
                OrderType, OrderStatus, AssetStatus, AccountStatus, ActivityType, TradeActivityType,
                NonTradeActivityStatus, CorporateActionType, CorporateActionSubType, CorporateActionDateType,
                DTBPCheck, PDTCheck, TradeConfirmationEmail)
# https://stackoverflow.com/questions/36588126/uuid-is-not-json-serializable
class UUIDEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, UUID):
            # if the obj is uuid, we simply return the value of uuid
            return obj.hex
        # https://stackoverflow.com/questions/11875770/how-to-overcome-datetime-datetime-not-json-serializable
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        if isinstance(obj, alpaca_enums):
            return obj.value
        return json.JSONEncoder.default(self, obj)

def write_trade_data_to_file(market_order_data, file, encoder):
    file = open(f"{account_data_path}/{todays_date}_account_positions.json", "a", newline ='')
    current_time = datetime.now().strftime("%H:%M:%S")
    file.write(f"{current_time}\n")
    file.write("Trade Info:\n")
    json.dump(market_order_data.dict(), file, cls=encoder)
    file.write("\n")
    file.close()

def write_account_info_to_file(file, account, positions, encoder):
    current_time = datetime.now().strftime("%H:%M:%S")
    file.write(f"{current_time}\n")
    file.write(f"Account Info:\n")
    json.dump(account.dict(), file, cls=UUIDEncoder)
    file.write("\nPositions Info:\n")
    for position in positions:
        json.dump(position.dict(), file, cls=UUIDEncoder)
    file.write("\n")
    file.close()

def write_trade_info_to_file(trade_info, file_name):
    if not Path(file_name).exists():
        with open(file_name, 'a') as f:
            writer = csv.writer(f)
            writer.writerow(trade_info.columns.tolist())

    trade_info.to_csv(file_name, mode='a', header=False, index=False)
    return

def get_stock_info(stock_symbol, start_date, end_date, data_client):
    data_bars_params = StockBarsRequest(
                    symbol_or_symbols=stock_symbol,
                    # timeframe=TimeFrame(1, TimeFrameUnit.Minute),
                    timeframe=TimeFrame.Minute,
                    start=start_date,
                    end=end_date
                    )

    # Get data
    try:
        data_bars = data_client.get_stock_bars(data_bars_params)
    except Exception as e:
        print(f'Error: {e}')
        print(f'Data unavailable for {stock_symbol}')
        return None

    df = data_bars.df
    df = df.reset_index() # separates symbol and timestamp as columns rather than as multiindex

    return df

def get_stock_quotes(stock_symbol, start_date, end_date, data_client, est):
    data_bars_params = StockQuotesRequest(
                    symbol_or_symbols=stock_symbol,
                    # timeframe=TimeFrame(1, TimeFrameUnit.Minute),
                    start=start_date,
                    end=end_date
                    )

    # Get data
    try:
        data_bars = data_client.get_stock_quotes(data_bars_params)
    except Exception as e:
        print(f'Error: {e}')
        print(f'Data unavailable for {stock_symbol}')
        return None

    df = data_bars.df
    df = df.reset_index() # separates symbol and timestamp as columns rather than as multiindex

    return df

def simulate_execute_trades(stock_symbol, data, strategy_signal, strategy, investment_amount, delay, trade_info_file_name):
    trade_info = pd.DataFrame(columns=['symbol', 'timestamp', 'strategy', 'buy/sell', 'price_per_share', 'quantity', 'total_cost', 'gain/loss', 'total_cash_avail', 'total_account_value'])
    total_investment_amount = 1000
    # Filter out delay/lookback period
    print(f'Running simulation for {stock_symbol}')
    start = time.time()
    for index, row in data[data['timestamp'] >= data['timestamp'][0]+timedelta(days=delay)].iterrows():
        current_ask_price = row['close']
        if index not in strategy_signal.index:
            continue
        if (strategy_signal.loc[index]['signal'] == 1) and (trade_info.empty or trade_info.iloc[-1]['buy/sell'] == 'sell'): # and prev action is None or sell
            # buy
            if trade_info.empty:
                quantity =  total_investment_amount / current_ask_price
                total_cost = quantity * current_ask_price # might add a term here for trading fee
                total_cash_avail = total_investment_amount-total_cost
                total_account_value = total_cost+total_cash_avail
                trade_info_row = [stock_symbol, row['timestamp'], strategy, 'buy', current_ask_price, quantity, total_cost, 0, total_cash_avail, total_account_value]
            else:
                total_cash_avail = trade_info.iloc[-1]['total_cash_avail']
                investment_amount = min(total_investment_amount, total_cash_avail)
                quantity = investment_amount / current_ask_price
                total_cost = quantity * current_ask_price # might add a term here for trading fee
                total_cash_avail = investment_amount - total_cost
                total_account_value = total_cost+total_cash_avail
                trade_info_row = [stock_symbol, row['timestamp'], strategy, 'buy', current_ask_price, quantity, total_cost, 0, total_cash_avail, total_account_value]
            trade_info.loc[len(trade_info.index)] = trade_info_row
            write_trade_info_to_file(trade_info.iloc[[-1]], trade_info_file_name)
            # write last line to file
        elif (strategy_signal.loc[index]['signal'] == 0) and (not trade_info.empty) and (trade_info.iloc[-1]['buy/sell'] == 'buy'): # and prev action is None or buy
            # sell
            quantity = trade_info.iloc[-1]['quantity'] # sell previous amount bought
            total_profit = quantity * current_ask_price
            gain_loss = total_profit - trade_info.iloc[-1]['total_cost']
            total_cash_avail = total_profit + trade_info.iloc[-1]['total_cash_avail']
            total_account_value = total_cash_avail
            trade_info_row = [stock_symbol, row['timestamp'], strategy, 'sell', current_ask_price, quantity, total_profit, gain_loss, total_cash_avail, total_account_value]
            trade_info.loc[len(trade_info.index)] = trade_info_row
            write_trade_info_to_file(trade_info.iloc[[-1]], trade_info_file_name)
            # write last line to file
    end = time.time()
    print(f'Total simulation time for {stock_symbol}: {end-start}')
