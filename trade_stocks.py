from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce, AssetClass, AssetExchange, PositionSide, OrderClass, \
OrderType, OrderStatus, AssetStatus, AccountStatus, ActivityType, TradeActivityType, NonTradeActivityStatus, \
CorporateActionType, CorporateActionSubType, CorporateActionDateType, DTBPCheck, PDTCheck, TradeConfirmationEmail
from alpaca.trading.client import TradingClient
from alpaca.data.requests import StockLatestQuoteRequest, StockQuotesRequest, StockBarsRequest
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
import yaml
import pdb
import time
from datetime import datetime, timezone, date, timedelta
import csv
import json
from uuid import UUID
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

with open('./input.yaml', 'rb') as f:
    params = yaml.safe_load(f.read())

API_KEY = params.get("account_info").get("api_key")
SECRET_KEY = params.get("account_info").get("secret_key")
data_path = params.get("data_path")
account_data_path = params.get("account_data_path")
stock_symbol = "ABNB"

trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)
data_client = StockHistoricalDataClient(API_KEY, SECRET_KEY)

todays_date = datetime.now().strftime("%m_%d_%Y")
file = open(f"{account_data_path}/{todays_date}_account_positions.json", "w", newline ='')

# Store account info before trading
account = trading_client.get_account()
# Store positions before trading
positions = trading_client.get_all_positions()
# writing the data into the file
write_account_info_to_file(file, account, positions, UUIDEncoder)

latest_quote_params = StockLatestQuoteRequest(
                symbol_or_symbols=stock_symbol
                )

# Get historical data
sma_90_start_date = (datetime.today()+timedelta(days=-90))
sma_30_start_date = (datetime.today()+timedelta(days=-30))
sma_5_start_date = (datetime.today()+timedelta(days=-5))
sma_end_date = (datetime.today()+timedelta(days=-1))

sma_90_bars_params = StockBarsRequest(
                symbol_or_symbols=stock_symbol,
                timeframe=TimeFrame(1, TimeFrameUnit.Day),
                start=sma_90_start_date,
                end=sma_end_date
                )

sma_30_bars_params = StockBarsRequest(
                symbol_or_symbols=stock_symbol,
                timeframe=TimeFrame(1, TimeFrameUnit.Day),
                start=sma_30_start_date,
                end=sma_end_date
                )

sma_5_bars_params = StockBarsRequest(
                symbol_or_symbols=stock_symbol,
                timeframe=TimeFrame(1, TimeFrameUnit.Day),
                start=sma_5_start_date,
                end=sma_end_date
                )

sma_90_bars = data_client.get_stock_bars(sma_90_bars_params)
sma_30_bars = data_client.get_stock_bars(sma_30_bars_params)
sma_5_bars = data_client.get_stock_bars(sma_5_bars_params)

# Calculate this at the beginning of every day
# Alternatively have this info saved and add the current day's data at the end of each day
avg_close_price_90_day = 0
avg_close_price_30_day = 0
avg_close_price_5_day = 0
for bar in sma_90_bars[stock_symbol]:
    avg_close_price_90_day += bar.close
for bar in sma_30_bars[stock_symbol]:
    avg_close_price_30_day += bar.close
for bar in sma_5_bars[stock_symbol]:
    avg_close_price_5_day += bar.close
avg_close_price_90_day = avg_close_price_90_day/len(sma_90_bars[stock_symbol])
avg_close_price_30_day = avg_close_price_30_day/len(sma_30_bars[stock_symbol])
avg_close_price_5_day = avg_close_price_5_day/len(sma_5_bars[stock_symbol])
print(f"90 day average: {avg_close_price_90_day}")
print(f"30 day average: {avg_close_price_30_day}")
print(f"5 day average: {avg_close_price_5_day}")

# Setting parameters for our buy order
buy_order_params = MarketOrderRequest(
                    symbol=stock_symbol,
                    qty=1,
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.GTC
                    )
# Setting parameters for our sell order
sell_order_params = MarketOrderRequest(
                    symbol=stock_symbol,
                    qty=1,
                    side=OrderSide.SELL,
                    time_in_force=TimeInForce.GTC
                    )
total_shares = 0

# Buy 1 share at the beginning of the day
market_order_data = trading_client.submit_order(buy_order_params)
total_shares += 1
print("1 share bought")
time.sleep(10)
order_data = trading_client.get_order_by_id(market_order_data.id)
write_trade_data_to_file(order_data, file, UUIDEncoder)

while (datetime.now() >= datetime.now().replace(hour=9, minute=30, second=0, microsecond=0)
    and datetime.now() <= datetime.now().replace(hour=15, minute=45, second=0, microsecond=0)):
    # compare latest quote to sma
    latest_quote = data_client.get_stock_latest_quote(latest_quote_params)
    current_ask_price = latest_quote[stock_symbol].ask_price
    if current_ask_price < avg_close_price_5_day and total_shares < 2:
        # BUY
        # Submitting the buy order
        market_order_data = trading_client.submit_order(buy_order_params)
        total_shares += 1
        print("1 share bought")
        time.sleep(10)
        order_data = trading_client.get_order_by_id(market_order_data.id)
        write_trade_data_to_file(order_data, file, UUIDEncoder)
    elif current_ask_price > avg_close_price_5_day and total_shares > 1:
        # SELL
        # Submitting the sell order
        market_order_data = trading_client.submit_order(sell_order_params)
        total_shares -= 1
        print("1 share sold")
        time.sleep(10)
        order_data = trading_client.get_order_by_id(market_order_data.id)
        write_trade_data_to_file(order_data, file, UUIDEncoder)
    else:
        pass

    time.sleep(300) # Check price and potential trade every 5 minutes

if total_shares > 0:
    sell_order_params = MarketOrderRequest(
                        symbol=stock_symbol,
                        qty=total_shares,
                        side=OrderSide.SELL,
                        time_in_force=TimeInForce.GTC
                        )
    market_order_data = trading_client.submit_order(sell_order_params)
    print(f"{total_shares} share(s) sold")
    total_shares = 0
    time.sleep(10)
    order_data = trading_client.get_order_by_id(market_order_data.id)
    write_trade_data_to_file(order_data, file, UUIDEncoder)

# Store account info after trading
account = trading_client.get_account()
# Store positions after trading
positions = trading_client.get_all_positions()
# writing the data into the file
file = open(f"{account_data_path}/{todays_date}_account_positions.json", "a", newline ='')
write_account_info_to_file(file, account, positions, UUIDEncoder)

print("End of Trading day")
print(f"Total Shares: {total_shares}")
