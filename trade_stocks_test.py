from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.client import TradingClient
import yaml

with open('./input.yaml', 'rb') as f:
    params = yaml.safe_load(f.read())

API_KEY = params.get("account_info").get("api_key")
SECRET_KEY = params.get("account_info").get("secret_key")

trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)

# Getting account information and printing it
account = trading_client.get_account()
for property_name, value in account:
  print(f"\"{property_name}\": {value}")

# Setting parameters for our buy order
market_order_data = MarketOrderRequest(
                    symbol="BTC/USD",
                    qty=1,
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.GTC
                    )

# Submitting the order and then printing the returned object
market_order = trading_client.submit_order(market_order_data)
for property_name, value in market_order:
    print(f"\"{property_name}\": {value}")

# Get all open positions and print each of them
positions = trading_client.get_all_positions()
for position in positions:
    for property_name, value in position:
        print(f"\"{property_name}\": {value}")
