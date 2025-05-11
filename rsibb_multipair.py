import pandas as pd
import csv
import os
import logging
from datetime import datetime, timedelta
import time
import threading
import uuid
import ccxt
import numpy as np
import config
from colorama import Fore, Style, init
from recovery import TradingRecovery
from telegram_notifications import TelegramNotifier

init()

RSI_PERIOD = 14
BB_PERIOD = 20
BB_STD = 2.2
RSI_BUY_THRESHOLD = 32 
BASE_ORDER_SIZE = 20
MAX_ACTIVE_ORDERS = 2
MAX_SAFETY_ORDERS = 10
SAFETY_ORDER_STEP = 1.2 # %
TRADING_FEE = 0.1  # USDC - Default fee as fallback
BNB_DISCOUNT_ENABLED = True
MAKER_FEE_RATE = 0.1  # 0.1% maker fee
TAKER_FEE_RATE = 0.095  # 0.1% taker fee
BNB_DISCOUNT_RATE = 0.75  # 25% discount using BNB
SAFETY_ORDER_VOLUME_SCALE = 1.05
WAIT_TIME_AFTER_CLOSE = 40  # seconds
MAX_POSITION_SIZE = 220
MAX_TOTAL_INVESTMENT = 220
INITIAL_BALANCE_PER_BOT = MAX_POSITION_SIZE
CHECK_INTERVAL = 60  # seconds/candle
AUTO_PURCHASE_BNB = True  # Enable automatic BNB purchases when balance is low

# Constants for EUR Savings
PROFIT_PORTION_TO_EUR_SAVINGS = 3.0  # Amount of profit in USDC to allocate to EUR savings
USDC_ACCUMULATION_THRESHOLD_FOR_EUR = 12.0  # Accumulate this much USDC before converting to EUR
MIN_USDC_FOR_EUR_CONVERSION_TRADE = 10.0  # Minimum USDC amount for a EUR conversion trade
EUR_SAVINGS_TRADING_PAIR = 'EUR/USDC'  # Trading pair for USDC to EUR conversion


class CustomAdapter(logging.LoggerAdapter):
    def process(self, msg, kwargs):
        return f'[{self.extra["symbol"]}] {msg}', kwargs


logging.basicConfig(
    filename=os.path.join(
        'logs',
        f'rsibb_live_{datetime.now().strftime("%Y%m%d")}.log'),
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    encoding='utf-8')

for handler in logging.root.handlers:
    handler.formatter = logging.Formatter(
        fmt='%(asctime)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

logger = logging.getLogger()

log_file = os.path.join(
    'logs',
    f'rsibb_live_{datetime.now().strftime("%Y%m%d")}.log')
file_handler = logging.FileHandler(log_file, encoding='utf-8')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(
    logging.Formatter(
        '%(asctime)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'))

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter('%(message)s'))

logger.handlers = []
logger.addHandler(file_handler)

SYMBOL_COLORS = {
    "BTC/USDC": Fore.YELLOW,
    "XRP/USDC": Fore.BLUE,
    "HBAR/USDC": Fore.MAGENTA,
    "SOL/USDC": Fore.GREEN,
}
DEFAULT_SYMBOL_COLOR = Fore.CYAN


class SharedBalance:
    def __init__(self, initial_total_balance, exchange, quote_currency='USDC'):
        self.total_balance = initial_total_balance
        self.available_balance = initial_total_balance
        self.allocated_funds = {}
        self.lock = threading.Lock()
        self.exchange = exchange  # Store the shared exchange instance
        self.quote_currency = quote_currency
        self.last_update_time = datetime.now()

    def allocate(self, symbol, amount):
        with self.lock:
            if self.available_balance >= amount:
                self.available_balance -= amount
                if symbol not in self.allocated_funds:
                    self.allocated_funds[symbol] = 0
                self.allocated_funds[symbol] += amount
                return True
            return False

    def release(self, symbol, amount):
        with self.lock:
            if symbol in self.allocated_funds:
                # Don't release more than what's allocated
                release_amount = min(amount, self.allocated_funds[symbol])
                self.available_balance += release_amount
                self.allocated_funds[symbol] -= release_amount
                return True
            return False

    def get_symbol_allocation(self, symbol):
        return self.allocated_funds.get(symbol, 0)

    def update_balance_from_exchange(self):
        """Fetches the current total balance for the quote currency from the exchange."""
        try:
            balance_info = self.exchange.fetch_balance()
            # Find the free balance for the quote currency (e.g., USDC)
            new_total_balance = balance_info['free'][self.quote_currency]
            with self.lock:
                # Calculate the difference from the last known total
                diff = new_total_balance - self.total_balance
                # Adjust available balance by the same difference
                self.available_balance += diff
                # Update the total balance
                self.total_balance = new_total_balance
                self.last_update_time = datetime.now()
                logger.info(f"SharedBalance updated: Total {self.quote_currency}: {self.total_balance:.2f}, Available: {self.available_balance:.2f}")
                return True
        except (ccxt.NetworkError, ccxt.ExchangeError, KeyError, Exception) as e:
            logger.error(f"Error updating shared balance from exchange: {e}")
            return False


# Clase para monitorear el rendimiento por par de trading
class TradingPerformanceMonitor:
    def __init__(self, symbol):
        self.symbol = symbol
        self.trades_history = []
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_profit = 0
        self.total_loss = 0
        self.max_drawdown = 0
        self.best_trade_profit = 0
        self.worst_trade_loss = 0
        self.avg_profit_per_trade = 0
        self.avg_holding_time = timedelta(0)
        self.last_analysis_time = datetime.now()
        
        # Crear archivo CSV para estadísticas de rendimiento
        self.performance_file = os.path.join(
            'logs', 'performance', f"performance_{symbol.lower().replace('/', '_')}.csv")
        self.ensure_performance_file()
    
    def ensure_performance_file(self):
        # Asegurarse de que el directorio existe
        os.makedirs(os.path.dirname(self.performance_file), exist_ok=True)
        
        # Crear el archivo si no existe
        if not os.path.exists(self.performance_file):
            with open(self.performance_file, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([
                    'Timestamp', 'Symbol', 'Total Trades', 'Win Rate', 
                    'Total Profit', 'Avg Profit/Trade', 'Max Drawdown',
                    'Best Trade', 'Worst Trade', 'Avg Holding Time'
                ])
    
    def add_trade(self, trade_data):
        self.trades_history.append(trade_data)
        self.total_trades += 1
        
        profit = trade_data.get('profit', 0)
        if profit > 0:
            self.winning_trades += 1
            self.total_profit += profit
            if profit > self.best_trade_profit:
                self.best_trade_profit = profit
        else:
            self.losing_trades += 1
            self.total_loss += abs(profit)
            if abs(profit) > abs(self.worst_trade_loss):
                self.worst_trade_loss = profit
        
        # Actualizar estadísticas
        self.update_statistics()
    
    def update_statistics(self):
        if self.total_trades > 0:
            # Calcular win rate
            self.win_rate = (self.winning_trades / self.total_trades) * 100
            
            # Calcular promedio de ganancia por operación
            total_pnl = self.total_profit - self.total_loss
            self.avg_profit_per_trade = total_pnl / self.total_trades
            
            # Calcular tiempo promedio de retención
            if self.trades_history:
                total_holding_time = timedelta(0)
                valid_trades = 0
                for trade in self.trades_history:
                    if 'holding_time' in trade:
                        try:
                            # Convertir string a timedelta si es necesario
                            if isinstance(trade['holding_time'], str):
                                parts = trade['holding_time'].split(':')
                                if len(parts) == 3:
                                    hours, minutes, seconds = map(int, parts)
                                    td = timedelta(hours=hours, minutes=minutes, seconds=seconds)
                                    total_holding_time += td
                                    valid_trades += 1
                            elif isinstance(trade['holding_time'], timedelta):
                                total_holding_time += trade['holding_time']
                                valid_trades += 1
                        except Exception:
                            pass
                
                if valid_trades > 0:
                    self.avg_holding_time = total_holding_time / valid_trades
    
    def calculate_drawdown(self, current_positions, current_price):
        # Calcular drawdown actual basado en posiciones abiertas
        max_drawdown = 0
        for pos_id, position in current_positions.items():
            if position.get('status') == 'CLOSED':
                continue
                
            # Calcular valor actual
            total_amount = position.get('base_amount', 0)
            for order in position.get('safety_orders', []):
                total_amount += order.get('amount', 0)
            
            current_value = total_amount * current_price
            
            # Calcular costo total
            total_cost = position.get('total_cost', 0) + position.get('total_fees', 0)
            
            # Calcular drawdown en porcentaje
            if total_cost > 0:
                drawdown_pct = ((current_value - total_cost) / total_cost) * 100
                if drawdown_pct < 0 and abs(drawdown_pct) > abs(max_drawdown):
                    max_drawdown = drawdown_pct
        
        # Actualizar max drawdown si es necesario
        if abs(max_drawdown) > abs(self.max_drawdown):
            self.max_drawdown = max_drawdown
            
        return max_drawdown
    
    def save_performance_data(self):
        # Guardar datos de rendimiento en CSV
        with open(self.performance_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                self.symbol,
                self.total_trades,
                f"{self.win_rate:.2f}%" if hasattr(self, 'win_rate') else "0.00%",
                f"{(self.total_profit - self.total_loss):.2f}",
                f"{self.avg_profit_per_trade:.2f}" if hasattr(self, 'avg_profit_per_trade') else "0.00",
                f"{self.max_drawdown:.2f}%",
                f"{self.best_trade_profit:.2f}",
                f"{self.worst_trade_loss:.2f}",
                str(self.avg_holding_time).split('.')[0]  # Eliminar microsegundos
            ])
    
    def generate_performance_report(self):
        # Generar informe de rendimiento
        report = f"""
        ===== INFORME DE RENDIMIENTO PARA {self.symbol} =====
        Operaciones totales: {self.total_trades}
        Tasa de éxito: {self.win_rate:.2f}% ({self.winning_trades} ganadas, {self.losing_trades} perdidas)
        Beneficio total: {(self.total_profit - self.total_loss):.2f} USDC
        Beneficio promedio por operación: {self.avg_profit_per_trade:.2f} USDC
        Máximo drawdown: {self.max_drawdown:.2f}%
        Mejor operación: {self.best_trade_profit:.2f} USDC
        Peor operación: {self.worst_trade_loss:.2f} USDC
        Tiempo promedio de retención: {str(self.avg_holding_time).split('.')[0]}
        """
        return report


class BinanceTradingBot:
    # Removed exchange instance creation from bot, will use shared_balance.exchange
    def __init__(self, symbol, shared_balance, initial_allocation):
        self.symbol = symbol
        self.shared_balance = shared_balance
        self.initial_allocation = initial_allocation
        self.balance = initial_allocation
        self.positions = {}
        self.active_orders = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_invested = 0
        self.current_price = None
        self.last_trade_time = None

        self.wallet_savings = 0
        self.profit_reinvested = 0
        self.base_order_increment = 0
        self.realized_profit = 0
        self.unrealized_profit = 0
        self.total_fees_paid = 0
        self.pending_profit = 0  # Add this back for compatibility

        # EUR Savings attributes
        self.usdc_accumulated_for_eur = 0.0
        self.eur_savings_total = 0.0

        # Fee-related properties
        self.using_bnb_for_fees = False
        self.maker_fee = MAKER_FEE_RATE
        self.taker_fee = TAKER_FEE_RATE
        self.auto_purchase_bnb = AUTO_PURCHASE_BNB  # Add auto-purchase BNB setting

        # Use the shared exchange instance from SharedBalance
        self.exchange = shared_balance.exchange

        self.csv_file = os.path.join(
            'logs', 'trades', f"trades_{symbol.lower().replace('/', '_')}.csv")

        # Initialize logger first
        self.logger = CustomAdapter(logger, {'symbol': symbol})
        self.lock = threading.Lock()

        self.headers = [
            'Timestamp', 'Symbol', 'Order ID', 'Side', 'Order Type', 'Price',
            'Base Amount', 'USDC Amount', 'Fee', 'Profit', 'Parent Order ID',
            'Total Position Cost', 'Total Position Fees'
        ]
        self.initialize_csv()

        self.exchange_symbol = symbol

        self.logger.info(f"Initialized bot for {self.exchange_symbol}")

        # Initialize telegram notifier
        self.telegram = TelegramNotifier()

        # Update trading fees after logger is initialized
        self.update_trading_fees()

        self.first_fetch = True

        self.candle_history = pd.DataFrame()
        self.min_required_candles = max(RSI_PERIOD, BB_PERIOD) + 10

        # Inicializar el monitor de rendimiento para este par
        self.performance_monitor = TradingPerformanceMonitor(symbol)
        
        self.recovery = TradingRecovery(symbol)

        if self.recovery.restore_from_backup(self):
            self.logger.info("Bot state restored from backup")

    def log_message(self, message):
        self.logger.info(message)

    def initialize_csv(self):
        with self.lock:
            with open(self.csv_file, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(self.headers)

    def fetch_recent_candles(self):
        try:
            # Always fetch more candles than strictly needed for indicators
            limit = 100 if self.first_fetch else 30

            candles = self.exchange.fetch_ohlcv(
                symbol=self.exchange_symbol,
                timeframe='1m',
                limit=limit
            )

            df = pd.DataFrame(
                candles,
                columns=[
                    'timestamp',
                    'open',
                    'high',
                    'low',
                    'close',
                    'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

            if self.first_fetch:
                self.logger.info(
                    f"Initial fetch: {len(df)} candles from {df['timestamp'].min()} to {df['timestamp'].max()}")
                self.candle_history = df
                self.total_candles_processed = len(df)
                self.first_fetch = False
            else:
                # Add new candles to history, avoiding duplicates
                if not df.empty and not self.candle_history.empty:
                    # Identify new candles
                    new_timestamps = df['timestamp'].values
                    history_timestamps = self.candle_history['timestamp'].values
                    mask = np.array(
                        [ts not in history_timestamps for ts in new_timestamps])
                    new_candles = df[mask]

                    if not new_candles.empty:
                        # Append new candles and keep rolling window
                        self.candle_history = pd.concat(
                            [self.candle_history, new_candles])
                        self.candle_history = self.candle_history.tail(
                            300)  # Keep rolling window
                        self.total_candles_processed += len(new_candles)
                        self.logger.info(
                            f"Added {len(new_candles)} new candles to history (total processed: {self.total_candles_processed}, window size: {len(self.candle_history)})")

            return self.candle_history.copy()

        except Exception as e:
            self.logger.error(f"Error fetching candles: {str(e)}")
            return self.candle_history.copy() if not self.candle_history.empty else pd.DataFrame()

    def calculate_indicators(self, df):
        try:
            # Make a copy to avoid modifying original
            df = df.copy()

            # RSI
            delta = df['close'].diff()

            # Handle potential zeros in loss to avoid division by zero
            gain = delta.copy()
            gain[gain < 0] = 0
            loss = -delta.copy()
            loss[loss < 0] = 0

            # Use expanding window for first RSI_PERIOD values, then rolling
            avg_gain = gain.rolling(window=RSI_PERIOD, min_periods=1).mean()
            avg_loss = loss.rolling(window=RSI_PERIOD, min_periods=1).mean()

            # Avoid division by zero
            avg_loss = avg_loss.replace(0, 0.000001)

            rs = avg_gain / avg_loss
            df['rsi'] = 100 - (100 / (1 + rs))

            # Bollinger Bands
            df['sma'] = df['close'].rolling(
                window=BB_PERIOD, min_periods=1).mean()
            df['std'] = df['close'].rolling(
                window=BB_PERIOD, min_periods=1).std()
            df['upper_band'] = df['sma'] + (BB_STD * df['std'])
            df['lower_band'] = df['sma'] - (BB_STD * df['std'])

            return df

        except Exception as e:
            self.logger.error(f"Error calculating indicators: {str(e)}")
            return pd.DataFrame()  # Return empty DataFrame on error

    def validate_indicators(self, df):
        """Validate indicator calculation and data quality"""
        try:
            if df.empty:
                self.logger.error("Empty dataframe - no data received")
                return False

            # Check if we have enough data
            if len(df) < self.min_required_candles:
                self.logger.error(
                    f"Not enough data points: {len(df)} < {self.min_required_candles}")
                return False

            # Check latest values are not NaN
            latest = df.iloc[-1]
            if pd.isna(
                    latest['rsi']) or pd.isna(
                    latest['lower_band']) or pd.isna(
                    latest['close']):
                self.logger.error("Latest indicators contain NaN values")
                return False

            return True

        except Exception as e:
            self.logger.error(f"Error validating indicators: {str(e)}")
            return False

    def check_entry_conditions(self, row):
        # Check if we've reached the maximum number of active orders overall
        if len(self.positions) >= MAX_ACTIVE_ORDERS:
            self.logger.info(
                f"Maximum active orders reached: {len(self.positions)}/{MAX_ACTIVE_ORDERS}")
            return False

        # Count active positions for this symbol
        symbol_positions = 0
        for pos_id, position in self.positions.items():
            if position.get('symbol') == self.exchange_symbol and position.get(
                    'status') != 'CLOSED':
                symbol_positions += 1

        # Limit to MAX_ACTIVE_ORDERS positions per symbol
        if symbol_positions >= MAX_ACTIVE_ORDERS:
            self.logger.info(
                f"Maximum positions for {self.symbol} reached: {symbol_positions}/{MAX_ACTIVE_ORDERS}")
            return False

        try:
            # Ensure we're working with scalar values, not Series
            if isinstance(row, pd.Series):
                rsi_value = float(row['rsi'])
                lower_band = float(row['lower_band'])
                close_price = float(row['close'])
            else:  # DataFrame
                rsi_value = float(row['rsi'].iloc[-1])
                lower_band = float(row['lower_band'].iloc[-1])
                close_price = float(row['close'].iloc[-1])

            # Check for NaN values
            if pd.isna(rsi_value) or pd.isna(
                    lower_band) or pd.isna(close_price):
                self.logger.warning("NaN values found in indicators")
                return False

            # Calculate price to BB ratio
            price_bb_ratio = close_price / lower_band

            # Use scalar values in conditions
            rsi_condition = rsi_value < (RSI_BUY_THRESHOLD + 2)
            bb_condition = price_bb_ratio < 1.03

            # Additional logging for debugging
            self.logger.info(
                f"Entry check - RSI: {rsi_value:.2f}, Price/BB: {price_bb_ratio:.4f}, "
                f"RSI condition: {rsi_condition}, BB condition: {bb_condition}")

            return rsi_condition and bb_condition

        except Exception as e:
            self.logger.error(f"Error in check_entry_conditions: {str(e)}")
            return False

    def manage_profit_distribution(self, profit):
        """
        Add profit to the realized profit total.
        Distribution is now handled by the combined profit distribution function.
        """
        # Add to realized profit
        self.realized_profit += profit

        # For compatibility with other code that might access pending_profit
        self.pending_profit = 0

        # Log profit received
        self.logger.info(
            f"Received profit: {profit:.2f} USDC (Total realized: {self.realized_profit:.2f} USDC)")

        return self.base_order_increment

    def check_funds_available(self, required_amount):
        # Check if there's enough *available* balance for this specific order
        if self.shared_balance.available_balance < required_amount:
            self.logger.info(f"""
            Insufficient shared *available* balance:
            Required: {required_amount:.2f} USDC
            Available: {self.shared_balance.available_balance:.2f} USDC
            """)
            return False

        # Check if adding this amount exceeds the per-symbol limit
        symbol_allocation = self.shared_balance.get_symbol_allocation(self.symbol)
        if symbol_allocation + required_amount > MAX_POSITION_SIZE:
            self.logger.info(f"""
            Symbol allocation limit reached:
            Current Allocation: {symbol_allocation:.2f} USDC
            Required Additional: {required_amount:.2f} USDC
            Max Per Symbol: {MAX_POSITION_SIZE} USDC
            """)
            return False

        # Check if this allocation would push the *estimated* total balance over the limit
        # This acts as a final safeguard
        potential_total = self.shared_balance.total_balance + required_amount
        if potential_total > MAX_TOTAL_INVESTMENT:
            self.logger.info(f"""
            Allocation would exceed total investment limit:
            Current Total Balance: {self.shared_balance.total_balance:.2f} USDC
            Required Additional: {required_amount:.2f} USDC
            Potential Total: {potential_total:.2f} USDC
            Max Total Investment: {MAX_TOTAL_INVESTMENT} USDC
            """)
            # Consider if you want to allow partial allocation up to the limit
            # For now, denying the full amount is safer.
            return False

        # If all checks pass
        return True

    def calculate_fee(self, cost):
        """Calculate the fee for a trade based on the cost and current fee rate"""
        # Start with the default maker fee rate (0.1%)
        if hasattr(self, 'maker_fee') and self.maker_fee is not None:
            # Use exchange-provided fee if available
            fee_rate = self.maker_fee / 100  # Convert from percentage to decimal
        else:
            # Use default maker fee rate
            # Convert from percentage to decimal (0.1% -> 0.001)
            fee_rate = MAKER_FEE_RATE / 100

        # Apply BNB discount if enabled
        if BNB_DISCOUNT_ENABLED:
            fee_rate = fee_rate * BNB_DISCOUNT_RATE
            self.logger.debug(f"Applied BNB discount: {fee_rate:.6f}")

        # Calculate fee as a percentage of the cost
        fee = cost * fee_rate

        # Log the fee calculation for debugging
        self.logger.debug(
            f"Fee calculation: {cost} * {fee_rate:.6f} = {fee:.6f}")

        # If fee calculation fails or is too low, use the default fallback of
        # 0.1 USDC
        if fee <= 0 or fee > cost * 0.01:  # Cap at 1% as a sanity check
            fee = 0.1  # Default fallback fee of 0.1 USDC
            self.logger.debug(f"Using default fallback fee: {fee} USDC")

        return fee

    def execute_trade(
            self,
            side,
            order_type,
            amount,
            price=None,
            parent_id=None):
        """Execute a trade on the exchange"""
        # Initialize variables that might be used in exception handling
        actual_cost = 0
        actual_fee = 0
        release_amount = 0

        try:
            timestamp = datetime.now()
            formatted_timestamp = self.format_timestamp(
                timestamp, for_storage=True)

            # Log the trade attempt
            self.logger.info(
                f"Executing {side} {order_type} order for {amount:.8f} {self.exchange_symbol} at {'market price' if price is None else f'{price:.8f}'}")

            # Create the order parameters - use 'amount' not 'quantity'
            order_params = {
                'symbol': self.exchange_symbol,
                'side': side.lower(),
                'type': 'market' if price is None else 'limit',
                'amount': amount
            }

            if price is not None:
                order_params['price'] = price

            # Execute the order
            order = self.exchange.create_order(**order_params)
            order_id = order['id']
            self.logger.info(f"Order {order_id} created for {self.symbol} with params: {order_params}")

            # For BUY limit orders, check if filled within timeout
            if side.lower() == 'buy' and order_params.get('type') == 'limit':
                order_filled = False
                order_details_fetched = False # To track if we got order details from fetch_order
                start_time_check = time.time()
                timeout_seconds = 120  # 2 minutes timeout
                self.logger.info(f"BUY Limit order {order_id} for {self.symbol}. Waiting for fill (timeout: {timeout_seconds}s).")
                
                while time.time() - start_time_check < timeout_seconds:
                    try:
                        fetched_order = self.exchange.fetch_order(order_id, self.exchange_symbol)
                        order_details_fetched = True
                        if fetched_order['status'] == 'closed':  # 'closed' means filled for limit orders
                            order_filled = True
                            order = fetched_order  # Update order with filled details (price, amount, fees)
                            self.logger.info(f"BUY Limit order {order_id} for {self.symbol} FILLED. Price: {order.get('average', order.get('price'))}, Amount: {order.get('filled')}")
                            break
                        elif fetched_order['status'] == 'canceled':
                            self.logger.warning(f"BUY Limit order {order_id} for {self.symbol} was CANCELED externally before filling.")
                            self.telegram.send_notification(f"⚠️ {self.symbol} BUY Limit order {order_id[:8]} CANCELED externally.")
                            return None  # Order canceled, do not proceed
                        elif fetched_order['status'] == 'expired':
                            self.logger.warning(f"BUY Limit order {order_id} for {self.symbol} EXPIRED.")
                            self.telegram.send_notification(f"⚠️ {self.symbol} BUY Limit order {order_id[:8]} EXPIRED.")
                            return None # Order expired
                        
                        self.logger.debug(f"BUY Limit order {order_id} status: {fetched_order['status']}. Waiting... ({int(time.time() - start_time_check)}s / {timeout_seconds}s)")
                        time.sleep(10)  # Check every 10 seconds
                    except ccxt.OrderNotFound:
                        self.logger.warning(f"BUY Limit order {order_id} for {self.symbol} not found during status check. It might have filled and been removed or an issue occurred.")
                        # Attempt to fetch trades for the symbol to see if it filled
                        try:
                            trades = self.exchange.fetch_my_trades(symbol=self.exchange_symbol, limit=5)
                            for trade_item in trades: # renamed trade to trade_item to avoid conflict with outer scope if any
                                if trade_item['order'] == order_id:
                                    self.logger.info(f"BUY Limit order {order_id} found in recent trades. Assuming filled and attempting to re-fetch order details.")
                                    try:
                                        order = self.exchange.fetch_order(order_id, self.exchange_symbol) # Re-assign to 'order'
                                        if order['status'] == 'closed':
                                            order_filled = True
                                            order_details_fetched = True
                                            self.logger.info(f"Re-fetched BUY Limit order {order_id} for {self.symbol} - FILLED.")
                                        break # Exit trades loop
                                    except Exception as e_refetch:
                                        self.logger.error(f"Could not re-fetch order {order_id} after OrderNotFound but found in trades: {e_refetch}")
                            if order_filled: break # Exit while loop
                        except Exception as e_trades:
                            self.logger.error(f"Error fetching trades for {order_id}: {e_trades}")
                        time.sleep(10) # Wait before retrying fetch_order in the main loop
                    except Exception as e:
                        self.logger.error(f"Error fetching order {order_id} status: {e}. Retrying in 10s.")
                        time.sleep(10)
                        continue  # Retry fetching status
                
                if not order_filled:
                    self.logger.warning(f"BUY Limit order {order_id} for {self.symbol} did not fill within {timeout_seconds}s.")
                    try:
                        cancel_attempt = self.exchange.cancel_order(order_id, self.exchange_symbol)
                        self.logger.info(f"Attempted to cancel BUY Limit order {order_id}. Response: {cancel_attempt}")
                        time.sleep(2) # Give time for cancellation to process
                        final_check_order = self.exchange.fetch_order(order_id, self.exchange_symbol)
                        order_details_fetched = True
                        if final_check_order['status'] == 'canceled':
                            self.logger.warning(f"BUY Limit order {order_id} for {self.symbol} successfully CANCELED after timeout.")
                            self.telegram.send_notification(f"⚠️ {self.symbol} BUY Limit order {order_id[:8]} timed out and was CANCELED.")
                        elif final_check_order['status'] == 'closed':
                             self.logger.warning(f"BUY Limit order {order_id} for {self.symbol} was FILLED just before/during cancellation.")
                             order = final_check_order # Update order with filled details
                             order_filled = True # Treat as filled
                        else:
                            self.logger.error(f"BUY Limit order {order_id} for {self.symbol} timed out but FAILED to cancel or status unclear: {final_check_order['status']}. Manual check required.")
                            self.telegram.send_notification(f"🚨 CRITICAL: {self.symbol} BUY Limit order {order_id[:8]} timed out, FAILED to cancel. Status: {final_check_order['status']}. Manual check!")
                    except ccxt.OrderNotFound:
                         self.logger.warning(f"BUY Limit order {order_id} for {self.symbol} not found during cancellation. Assumed already canceled/expired or filled and removed.")
                         # If not found, it's likely not open. To be safe, don't assume filled unless confirmed.
                    except Exception as e:
                        self.logger.error(f"Error canceling timed out BUY limit order {order_id}: {e}. Order might still be active or filled.")
                        self.telegram.send_notification(f"🚨 ERROR: {self.symbol} BUY Limit order {order_id[:8]} timed out. Error during cancellation: {e}. Manual check!")
                    
                    if not order_filled: # If still not marked as filled after cancellation logic
                        return None  # Order not filled and (attempted) canceled, do not proceed

            # If the order was not a BUY LIMIT, or it was filled, or it was a market order.
            # For market orders or if fetch_order wasn't called successfully during limit order check, fetch details.
            if not order_details_fetched or (order_params.get('type') == 'market' and not order.get('filled')):
                try:
                    time.sleep(1) # Give a moment for market order to fill / details to be available
                    fetched_order = self.exchange.fetch_order(order_id, self.exchange_symbol)
                    order = fetched_order # Update order with latest details
                    self.logger.info(f"Fetched/updated details for order {order_id}: Status {order.get('status')}, Filled: {order.get('filled')}, Price: {order.get('average', order.get('price'))}")
                except Exception as e:
                    self.logger.error(f"Could not fetch/update details for order {order_id}: {e}")
                    if side.lower() == 'buy':
                        self.telegram.send_notification(f"🚨 WARNING: {self.symbol} BUY order {order_id[:8]} - could not confirm fill details. {e}")
                        # Depending on strictness, might return None here

            executed_price = float(order.get('average', order.get('price', 0)))
            executed_amount = float(order.get('filled', order.get('amount', 0)))
            
            if side.lower() == 'buy' and executed_amount == 0:
                current_status = order.get('status', 'unknown')
                self.logger.error(f"BUY order {order_id} for {self.symbol} resulted in 0 filled amount. Final Status: {current_status}.")
                if current_status not in ['canceled', 'expired']:
                    self.telegram.send_notification(f"🚨 CRITICAL: {self.symbol} BUY order {order_id[:8]} reported 0 filled amount. Status: {current_status}. Manual check needed!")
                return None

            actual_cost = executed_price * executed_amount
            if side.lower() == 'buy' and actual_cost == 0 and executed_amount > 0:
                 self.logger.warning(f"BUY order {order_id} for {self.symbol} has 0 cost (price: {executed_price}) but filled amount {executed_amount}. This might be a data issue or an exchange peculiarity.")


            # Calculate fee based on the actual cost
            actual_fee = self.calculate_fee(actual_cost)

            # Update balance
            if side == 'BUY':
                self.balance -= (actual_cost + actual_fee)
                self.total_invested += actual_cost
                self.total_fees_paid += actual_fee

                # Create a new position entry if this is a base order (not a
                # safety order)
                if not parent_id:
                    # This is a base order, create a new position
                    self.positions[order_id] = {
                        'symbol': self.exchange_symbol,
                        'timestamp': formatted_timestamp,
                        'base_order_id': order_id,
                        'base_price': executed_price,
                        'base_amount': executed_amount,
                        'base_cost': actual_cost,
                        'base_fees': actual_fee,
                        'total_cost': actual_cost,
                        'total_fees': actual_fee,
                        'total_invested': actual_cost + actual_fee,
                        'safety_orders': [],
                        'status': 'OPEN'
                    }
                    self.logger.info(
                        f"Created new position {order_id[:8]} for {self.symbol} at {executed_price}")
                elif parent_id in self.positions:
                    # This is a safety order, add it to the existing position
                    safety_order = {
                        'order_id': order_id,
                        'timestamp': formatted_timestamp,
                        'price': executed_price,
                        'amount': executed_amount,
                        'cost': actual_cost,
                        'fee': actual_fee
                    }
                    self.positions[parent_id]['safety_orders'].append(
                        safety_order)
                    self.positions[parent_id]['total_cost'] += actual_cost
                    self.positions[parent_id]['total_fees'] += actual_fee
                    # Update total_invested field for safety orders
                    if 'total_invested' in self.positions[parent_id]:
                        self.positions[parent_id]['total_invested'] += actual_cost + actual_fee
                    else:
                        self.positions[parent_id]['total_invested'] = self.positions[parent_id]['total_cost'] + \
                            self.positions[parent_id]['total_fees']
                    self.logger.info(
                        f"Added safety order to position {parent_id[:8]}, total cost: {self.positions[parent_id]['total_cost']:.2f}")
            else:  # SELL
                self.balance += (actual_cost - actual_fee)
                self.total_fees_paid += actual_fee

            # Log the trade
            self.log_trade(
                timestamp=timestamp,
                symbol=self.exchange_symbol,
                order_id=order['id'],
                side=side,
                order_type=order_type,
                price=executed_price,
                amount=executed_amount,
                fee=actual_fee,
                parent_id=parent_id,
                profit=0
            )

            # Send Telegram notification for the trade
            self.telegram.notify_trade_executed(
                symbol=self.symbol,
                side=side,
                amount=executed_amount,
                price=executed_price,
                order_type=order_type,
                position_size=actual_cost,
                total_invested=self.total_invested,
                total_fees=actual_fee,
                exchange="Binance",
                fee_percentage=self.taker_fee if hasattr(
                    self,
                    'taker_fee') else 0.001)

            return order_id

        except Exception as e:
            # Calculate the amount to release based on the initialized values
            release_amount = 0
            if actual_cost > 0:
                release_amount = actual_cost + actual_fee

            self.telegram.notify_error(
                self.symbol, f"Trade execution failed: {str(e)}")
            # If trade fails, release the allocated funds
            if release_amount > 0:
                self.shared_balance.release(self.symbol, release_amount)
            self.logger.error(f"Error executing trade: {str(e)}")
            return None

    def log_trade(
            self,
            timestamp,
            symbol,
            order_id,
            side,
            order_type,
            price,
            amount,
            fee=None,
            parent_id=None,
            profit=0):
        def format_value(value):
            """Format numerical values to max 2 decimal places, handle very small amounts"""
            if isinstance(value, (int, float)):
                if abs(value) < 0.01:
                    return f"{value:.8f}"
                else:
                    return f"{value:.2f}"
            return value

        safety_order_count = 0
        if parent_id and parent_id in self.positions:
            safety_order_count = len(
                self.positions[parent_id]['safety_orders'])

        # Format order type display
        if order_type == "CLOSE":
            order_type_display = "Close"
        elif not parent_id:
            order_type_display = "Base"
        elif parent_id and parent_id in self.positions:
            order_type_display = f"Safety_{safety_order_count}"
        else:
            order_type_display = order_type.capitalize() if order_type else "UNKNOWN"

        usdc_amount = amount * price

        if fee is None or fee == 0:
            fee = self.calculate_fee(usdc_amount)

        self.logger.debug(f"Logging trade with fee: {fee}")

        # Format timestamp for storage
        formatted_timestamp = self.format_timestamp(
            timestamp, for_storage=True)

        short_order_id = str(order_id)
        if len(short_order_id) > 5:
            short_order_id = short_order_id[-5:]

        total_position_cost = usdc_amount
        total_position_fees = fee

        if parent_id and parent_id in self.positions:
            position = self.positions[parent_id]
            total_position_cost += position.get('total_cost', 0)
            total_position_fees += position.get('total_fees', 0)

        # Add BNB discount info to log
        bnb_info = "Using BNB discount" if self.using_bnb_for_fees else "Standard fee"
        fee_type = "Taker" if side == "BUY" or side == "SELL" else "Maker"
        fee_percentage = self.taker_fee if fee_type == "Taker" else self.maker_fee

        self.logger.info(
            f"Trade log entry: {side} {order_type_display} {amount:.8f} @ {price:.8f} = {usdc_amount:.2f} USDC, Fee: {fee:.4f} USDC ({fee_percentage:.4f}%, {bnb_info}), Total Position Cost: {total_position_cost:.2f}, Total Position Fees: {total_position_fees:.2f}")

        # Truncate parent_id if it exists and is too long
        short_parent_id = ""
        if parent_id:
            short_parent_id = str(parent_id)
            if len(short_parent_id) > 5:
                short_parent_id = short_parent_id[-5:]

        formatted_trade = {
            'Timestamp': formatted_timestamp,
            'Symbol': symbol,
            'Order ID': short_order_id,
            'Side': side,
            'Order Type': order_type_display,
            'Price': f"{price:.5f}",  # Use 5 decimal places for price
            'Base Amount': format_value(amount),
            'USDC Amount': format_value(usdc_amount),
            'Fee': format_value(fee),
            'Profit': format_value(profit),
            'Parent Order ID': short_parent_id,
            'Total Position Cost': format_value(total_position_cost),
            'Total Position Fees': format_value(total_position_fees)
        }

        with self.lock:
            with open(self.csv_file, mode='a', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=self.headers)
                writer.writerow(formatted_trade)

    def calculate_position_profit(self, position, current_price):
        try:
            # Start with base amounts
            total_amount = float(position['base_amount'])
            total_cost = float(position['base_cost'])
            total_fees = float(position['base_fees'])

            # Add all safety orders
            for safety_order in position['safety_orders']:
                total_amount += float(safety_order['amount'])
                total_cost += float(safety_order['cost'])
                total_fees += float(safety_order['fee'])

            # Calculate average entry price
            avg_entry = total_cost / total_amount if total_amount > 0 else 0

            # Calculate current value of the entire position
            current_value = total_amount * float(current_price)

            self.logger.info(f"""
            Detailed Profit Calculation:
            Total Amount: {total_amount:.8f}
            Average Entry Price: {avg_entry:.8f}
            Current Price: {current_price:.8f}
            Total Cost: {total_cost:.2f}
            Total Fees: {total_fees:.2f}
            Current Value: {current_value:.2f}
            """)

            # Calculate raw profit (without fees)
            raw_profit = current_value - total_cost
            raw_profit_pct = (raw_profit / total_cost) * \
                100 if total_cost > 0 else 0

            # Calculate estimated closing fee for selling the position
            closing_fee = self.calculate_fee(current_value)

            # Calculate net profit (including both entry and exit fees)
            net_profit = raw_profit - total_fees - closing_fee
            net_profit_pct = (net_profit / total_cost) * \
                100 if total_cost > 0 else 0

            # For display purposes, we'll use a more optimistic "current profit" that only includes
            # a portion of the entry fee, proportional to how far the price has moved
            # This prevents showing a large negative profit immediately after
            # trade execution
            if raw_profit >= 0:
                # If we're in profit, we can start accounting for the entry fee
                fee_factor = min(
                    1.0, raw_profit / total_fees) if total_fees > 0 else 1.0
                current_profit = raw_profit - (total_fees * fee_factor)
            else:
                # If we're not in profit yet, we'll show the raw profit
                current_profit = raw_profit

            current_profit_pct = (
                current_profit / total_cost) * 100 if total_cost > 0 else 0

            self.logger.info(f"""
            Profit Calculation Results:
            Position: {position.get('exchange_order_id', '')[:8]}
            Raw Profit: {raw_profit:.2f} ({raw_profit_pct:.2f}%)
            Current Profit: {current_profit:.2f} ({current_profit_pct:.2f}%)
            Closing Fee: {closing_fee:.2f}
            Net Profit (after all fees): {net_profit:.2f} ({net_profit_pct:.2f}%)
            Average Entry: {avg_entry:.8f}
            """)

            # Return current profit as the main profit value for display
            return {
                'profit': current_profit,
                'profit_pct': current_profit_pct,
                'raw_profit': raw_profit,
                'raw_profit_pct': raw_profit_pct,
                'current_value': current_value,
                'avg_entry': avg_entry,
                'net_profit': net_profit,
                'net_profit_pct': net_profit_pct
            }
        except Exception as e:
            self.logger.error(f"Error in calculate_position_profit: {str(e)}")
            return {
                'profit': 0,
                'profit_pct': 0,
                'raw_profit': 0,
                'raw_profit_pct': 0,
                'current_value': 0,
                'avg_entry': 0,
                'net_profit': 0,
                'net_profit_pct': 0
            }

    def print_position_status(self, position_id, current_price, timestamp):
        with self.lock:
            position = self.positions[position_id]
            profit_result = self.calculate_position_profit(
                position, current_price)

            # Get different profit metrics
            raw_profit_pct = profit_result['raw_profit_pct']
            current_profit_pct = profit_result['profit_pct']
            net_profit_pct = profit_result['net_profit_pct']
            avg_entry = profit_result['avg_entry']

            total_fees = float(position['base_fees'])
            for so in position['safety_orders']:
                total_fees += float(so['fee'])

            total_amount = float(position['base_amount'])
            for so in position['safety_orders']:
                total_amount += float(so['amount'])

            safety_orders_count = len(position['safety_orders'])

            formatted_timestamp = self.format_timestamp(timestamp)
            print(f"\nPOSITION STATUS UPDATE - {formatted_timestamp}")
            print("=" * 50)

            print("\nPOSITION DETAILS:")
            print(f"    Symbol:          {self.symbol}")
            print(f"    Position ID:     {position_id[:8]}")
            print(
                f"    Order Type:      {'BASE + ' + str(safety_orders_count) + ' Safety Orders' if safety_orders_count > 0 else 'BASE'}")

            # Get the entry price (base price)
            base_price = float(position['base_price'])

            # Calculate average entry price if there are safety orders
            if safety_orders_count > 0:
                total_cost = float(position['base_cost'])
                total_amount = float(position['base_amount'])

                for so in position['safety_orders']:
                    total_cost += float(so['cost'])
                    total_amount += float(so['amount'])

                avg_entry_price = total_cost / total_amount if total_amount > 0 else 0

                # Display both base price and average entry price
                print(
                    f"    Entry: {Fore.CYAN}{base_price:.5f}{Style.RESET_ALL} | Avg: {Fore.CYAN}{avg_entry_price:.5f}{Style.RESET_ALL}")
            else:
                # For positions without safety orders, show entry price and
                # current price
                print(
                    f"    Entry: {Fore.CYAN}{base_price:.5f}{Style.RESET_ALL} | Current: {Fore.CYAN}{current_price:.5f}{Style.RESET_ALL}")

            print(f"    Total Amount:    {total_amount:.8f}")

            print(f"\n{Fore.YELLOW}FINANCIAL DETAILS:{Style.RESET_ALL}")
            print(f"    Base Cost:       {position['base_cost']:.2f} USDC")
            print(f"    Total Cost:      {position['total_cost']:.2f} USDC")
            print(f"    Total Fees:      {total_fees:.2f} USDC")
            print(
                f"    Current Value:   {profit_result['current_value']:.2f} USDC")

            # Display profit information with color coding
            raw_profit_color = Fore.GREEN if raw_profit_pct >= 0 else Fore.RED
            current_profit_color = Fore.GREEN if current_profit_pct >= 0 else Fore.RED
            net_profit_color = Fore.GREEN if net_profit_pct >= 0 else Fore.RED

            print(f"\n{Fore.YELLOW}PROFIT DETAILS:{Style.RESET_ALL}")
            print(
                f"    Raw Profit:      {raw_profit_color}{profit_result['raw_profit']:.2f} USDC ({raw_profit_pct:.2f}%){Style.RESET_ALL}")
            print(
                f"    Current Profit:  {current_profit_color}{profit_result['profit']:.2f} USDC ({current_profit_pct:.2f}%){Style.RESET_ALL}")
            print(
                f"    Net Profit:      {net_profit_color}{profit_result['net_profit']:.2f} USDC ({net_profit_pct:.2f}%){Style.RESET_ALL}")
            print(f"    (Net profit includes all fees)")

            # Calculate progress towards target using raw profit
            target_pct = config.PROFIT_TARGET
            progress = min(100, (raw_profit_pct / target_pct)
                           * 100) if target_pct > 0 else 0
            progress = max(0, progress)  # Ensure progress is not negative

            # Create a progress bar
            bar_length = 25
            filled_length = int(bar_length * progress / 100)
            bar = '|' + '█' * filled_length + '░' * \
                (bar_length - filled_length) + '|'

            print(f"\n{Fore.YELLOW}TARGET PROGRESS:{Style.RESET_ALL}")
            print(f"    Target:          {target_pct:.1f}%")
            print(f"    Progress:        {bar} {progress:.1f}%")

            print("=" * 50)

    def calculate_position_value(self, position, current_price):
        """Calculate current value of a position including all orders"""
        total_amount = position['base_amount']
        for order in position['safety_orders']:
            total_amount += order['amount']
        return total_amount * current_price

    def update_unrealized_profits(self, current_price):
        """Update unrealized profits and available balance"""
        total_position_value = 0
        total_position_cost = 0

        for pos_id, position in self.positions.items():
            if position.get('status') == 'CLOSED':
                continue

            # Calculate total amount in position
            total_amount = position['base_amount']
            for order in position['safety_orders']:
                total_amount += order['amount']

            # Calculate position value and cost
            position_value = total_amount * current_price
            position_cost = position['total_cost'] + position['total_fees']

            total_position_value += position_value
            total_position_cost += position_cost

        self.unrealized_profit = total_position_value - total_position_cost

        self.logger.info(f"""
        Balance Update:
        Initial Balance: {self.initial_allocation:.2f} USDC
        Current Balance: {self.balance:.2f} USDC
        Total Invested: {self.total_invested:.2f} USDC
        Total Position Value: {total_position_value:.2f} USDC
        Unrealized Profit: {self.unrealized_profit:.2f} USDC
        Available Balance: {self.shared_balance.available_balance:.2f} USDC
        """)

    def close_position(self, position_id, current_price=None):
        """Close a position and calculate profit"""
        if position_id not in self.positions:
            self.logger.error(f"Position {position_id} not found")
            return False

        position = self.positions[position_id]

        # Calculate total amount and cost
        total_amount = position['base_amount']
        total_cost = position['base_cost']
        total_fees = position['base_fees']

        for safety_order in position['safety_orders']:
            total_amount += safety_order['amount']
            total_cost += safety_order['cost']
            total_fees += safety_order['fee']

        # If current price not provided, get it from the exchange
        if current_price is None:
            current_price = self.fetch_current_price()

        # Log the decision price and expected profit before execution
        expected_value = total_amount * current_price
        expected_profit = expected_value - total_cost - total_fees
        expected_profit_pct = (expected_profit / total_cost) * 100
        self.logger.info(
            f"Expected close values - Price: {current_price:.8f}, Profit: {expected_profit:.8f} USDC ({expected_profit_pct:.2f}%)")

        try:
            # Execute market sell order
            timestamp = datetime.now()
            formatted_timestamp = self.format_timestamp(
                timestamp, for_storage=True)

            order = self.exchange.create_order(
                symbol=self.exchange_symbol,
                type='market',
                side='sell',
                amount=total_amount
            )

            executed_price = float(
                order['price']) if 'price' in order and order['price'] else current_price
            executed_amount = float(
                order['amount']) if 'amount' in order and order['amount'] else total_amount

            # Calculate current value and fees
            current_value = executed_amount * executed_price
            closing_fee = self.calculate_fee(current_value)

            # Calculate net proceeds and profit
            net_proceeds = current_value - closing_fee
            profit = net_proceeds - total_cost - total_fees
            profit_percentage = (profit / total_cost) * 100

            # Log price difference and slippage
            price_diff = executed_price - current_price
            price_slippage_pct = (price_diff / current_price) * 100
            self.logger.info(
                f"Price slippage: {price_diff:.8f} ({price_slippage_pct:.4f}%)")

            self.logger.info(f"""
            Position Closed:
            Position ID: {position_id}
            Total Amount: {total_amount:.8f}
            Total Cost: {total_cost:.2f} USDC
            Total Fees: {total_fees:.2f} USDC
            Decision Price: {current_price:.8f}
            Executed Price: {executed_price:.8f}
            Current Value: {current_value:.2f} USDC
            Closing Fee: {closing_fee:.2f} USDC
            Net Proceeds: {net_proceeds:.2f} USDC
            Profit: {profit:.2f} USDC ({profit_percentage:.2f}%)
            """)

            # Update balance with net proceeds
            self.balance += net_proceeds
            
            # Log the profit being added to realized profit for debugging
            self.logger.info(f"Received profit: {profit:.2f} USDC (Total realized: {self.realized_profit + profit:.2f} USDC)")
            
            self.realized_profit += profit
            self.total_fees_paid += closing_fee

            # Store position data before removing it
            position_data = self.positions[position_id].copy()
            position_data['status'] = 'CLOSED'
            position_data['close_price'] = executed_price
            position_data['close_timestamp'] = formatted_timestamp
            position_data['profit'] = profit
            position_data['profit_percentage'] = profit_percentage

            # Log the trade
            self.log_trade(
                timestamp=timestamp,
                symbol=self.exchange_symbol,
                order_id=order['id'],
                side="SELL",
                order_type="CLOSE",
                price=executed_price,
                amount=executed_amount,
                fee=closing_fee,
                parent_id=position_id,
                profit=profit
            )

            # Calculate time held
            start_time = datetime.strptime(
                position['timestamp'], '%Y-%m-%d %H:%M:%S')
            end_time = timestamp
            # Remove microseconds
            time_held = str(end_time - start_time).split('.')[0]

            # Send Telegram notification for position closure
            self.telegram.notify_position_closed(
                symbol=self.symbol,
                position_id=position_id,
                profit=profit,
                profit_percentage=profit_percentage,
                total_fees=total_fees + closing_fee,
                time_held=time_held,
                exchange="Binance",
                fee_percentage=self.taker_fee if hasattr(
                    self,
                    'taker_fee') else 0.001)

            # Remove the position from the active positions dictionary
            del self.positions[position_id]
            self.logger.info(
                f"Position {position_id} removed from active positions")

            # Registrar datos de la operación en el monitor de rendimiento
            trade_data = {
                'position_id': position_id,
                'symbol': self.symbol,
                'entry_price': position['base_price'],
                'exit_price': executed_price,
                'profit': profit,
                'profit_percentage': profit_percentage,
                'fees': total_fees + closing_fee,
                'holding_time': time_held,
                'timestamp': formatted_timestamp,
                'safety_orders_used': len(position['safety_orders'])
            }
            self.performance_monitor.add_trade(trade_data)
            
            # Guardar datos de rendimiento cada 5 operaciones o cada 24 horas
            if (self.performance_monitor.total_trades % 5 == 0 or 
                (datetime.now() - self.performance_monitor.last_analysis_time).total_seconds() > 86400):
                self.performance_monitor.save_performance_data()
                performance_report = self.performance_monitor.generate_performance_report()
                self.logger.info(f"Informe de rendimiento para {self.symbol}:\n{performance_report}")
                self.performance_monitor.last_analysis_time = datetime.now()
                
                # Enviar informe de rendimiento por Telegram
                # Changed from notify_custom_message to send_notification as the former doesn't exist
                self.telegram.send_notification(
                    f"📊 Informe de rendimiento para {self.symbol}:\n{performance_report}")

            # Manage profit distribution (profit is already added to self.realized_profit)
            if profit > 0:
                # self.manage_profit_distribution(profit) # This was causing double profit reporting

                # Allocate a portion of profit to EUR savings buffer
                if profit >= PROFIT_PORTION_TO_EUR_SAVINGS:
                    self.usdc_accumulated_for_eur += PROFIT_PORTION_TO_EUR_SAVINGS
                    self.logger.info(f"Added {PROFIT_PORTION_TO_EUR_SAVINGS:.2f} USDC to EUR savings buffer. Current buffer: {self.usdc_accumulated_for_eur:.2f} USDC")
                    self._convert_usdc_to_eur_savings() # Attempt conversion if threshold met
                else:
                    self.logger.info(f"Profit {profit:.2f} USDC is less than {PROFIT_PORTION_TO_EUR_SAVINGS:.2f} USDC. Not adding to EUR savings buffer.")

                # Update win/loss counters
                self.winning_trades += 1
            else:
                self.losing_trades += 1

            return True

        except Exception as e:
            self.telegram.notify_error(
                self.symbol, f"Position closing failed: {str(e)}")
            self.logger.error(
                f"Error closing position {position_id}: {str(e)}")
            return False

    def _convert_usdc_to_eur_savings(self):
        """Converts accumulated USDC to EUR if thresholds are met."""
        if self.usdc_accumulated_for_eur >= USDC_ACCUMULATION_THRESHOLD_FOR_EUR:
            amount_to_convert = self.usdc_accumulated_for_eur # Use the full accumulated amount if it's over threshold
            
            if amount_to_convert < MIN_USDC_FOR_EUR_CONVERSION_TRADE:
                self.logger.info(f"USDC for EUR conversion ({amount_to_convert:.2f}) is below minimum trade amount ({MIN_USDC_FOR_EUR_CONVERSION_TRADE:.2f} USDC). Holding.")
                return

            self.logger.info(f"Attempting to convert {amount_to_convert:.2f} USDC to EUR.")
            actual_trading_pair = EUR_SAVINGS_TRADING_PAIR # Default pair
            try:
                markets = self.exchange.load_markets()
                if actual_trading_pair not in markets:
                    self.logger.error(f"Trading pair {actual_trading_pair} not found on exchange.")
                    # Fallback logic or alternative pair can be added here if desired
                    # For now, we'll just log and return if the primary pair isn't found.
                    return
                
                market_info = markets[actual_trading_pair]

                ticker = self.exchange.fetch_ticker(actual_trading_pair)
                # We are buying EUR with USDC, so we need the ask price (price sellers are asking for EUR)
                current_price_eur_usdc = ticker['ask'] 
                if not current_price_eur_usdc or current_price_eur_usdc <= 0:
                    self.logger.error(f"Invalid or zero ask price for {actual_trading_pair}: {current_price_eur_usdc}")
                    return
                
                # Calculate how much EUR we can buy with the accumulated USDC
                eur_to_buy = amount_to_convert / current_price_eur_usdc
                # Apply exchange precision rules for the amount (base currency: EUR for EUR/USDC)
                eur_to_buy_adjusted = self.exchange.amount_to_precision(actual_trading_pair, eur_to_buy)
                
                # Ensure the cost of the adjusted amount is still above the minimums
                cost_of_adjusted_eur = float(eur_to_buy_adjusted) * current_price_eur_usdc
                min_cost_from_market = market_info.get('limits', {}).get('cost', {}).get('min', 0)
                # If min_cost_from_market is 0 or not defined, use our defined MIN_USDC_FOR_EUR_CONVERSION_TRADE
                effective_min_exchange_cost = min_cost_from_market if min_cost_from_market and min_cost_from_market > 0 else MIN_USDC_FOR_EUR_CONVERSION_TRADE

                self.logger.info(f"Calculated EUR to buy: {eur_to_buy:.8f}, adjusted: {eur_to_buy_adjusted} for {actual_trading_pair} at price {current_price_eur_usdc:.5f}. Estimated cost: {cost_of_adjusted_eur:.2f} USDC. Min exchange cost: {effective_min_exchange_cost:.2f}")

                if cost_of_adjusted_eur < effective_min_exchange_cost:
                     self.logger.warning(f"Adjusted EUR amount {eur_to_buy_adjusted} (costing {cost_of_adjusted_eur:.2f} USDC) is too small to meet minimum exchange trade value ({effective_min_exchange_cost:.2f} USDC). Skipping conversion.")
                     return
                if cost_of_adjusted_eur < MIN_USDC_FOR_EUR_CONVERSION_TRADE: # Double check against our own defined minimum
                    self.logger.warning(f"Adjusted EUR amount {eur_to_buy_adjusted} (costing {cost_of_adjusted_eur:.2f} USDC) is too small to meet defined minimum trade value ({MIN_USDC_FOR_EUR_CONVERSION_TRADE:.2f} USDC). Skipping conversion.")
                    return

                order = self.exchange.create_market_buy_order(actual_trading_pair, float(eur_to_buy_adjusted))

                filled_eur = float(order.get('filled', 0.0))
                actual_cost_usdc = float(order.get('cost', 0.0)) # Actual cost from the exchange

                if filled_eur > 0 and actual_cost_usdc > 0:
                    self.eur_savings_total += filled_eur
                    self.usdc_accumulated_for_eur -= actual_cost_usdc 
                    self.logger.info(f"Successfully converted {actual_cost_usdc:.2f} USDC to {filled_eur:.4f} EUR. Total EUR savings: {self.eur_savings_total:.4f} EUR. Remaining USDC buffer: {self.usdc_accumulated_for_eur:.2f} USDC")
                    if hasattr(self, 'telegram') and self.telegram is not None:
                        self.telegram.send_notification(f"💰 Converted {actual_cost_usdc:.2f} USDC to {filled_eur:.4f} EUR for savings. Total EUR savings: {self.eur_savings_total:.4f} EUR.")
                elif order.get('status') in ['open', 'pending', 'new'] and (filled_eur > 0 or actual_cost_usdc > 0):
                    self.logger.warning(f"EUR conversion order for {actual_trading_pair} placed but not immediately fully filled: {order}. Cost: {actual_cost_usdc}, Filled: {filled_eur}. Adjusting balances for partial fill.")
                    if filled_eur > 0:
                        self.eur_savings_total += filled_eur
                    if actual_cost_usdc > 0:
                         self.usdc_accumulated_for_eur -= actual_cost_usdc
                    self.logger.info(f"After partial fill: Total EUR savings: {self.eur_savings_total:.4f} EUR. Remaining USDC buffer: {self.usdc_accumulated_for_eur:.2f} USDC")
                else:
                    self.logger.error(f"EUR conversion order for {actual_trading_pair} did not fill as expected or failed: {order}")

            except ccxt.InsufficientFunds as e:
                self.logger.error(f"Insufficient USDC funds for EUR conversion ({actual_trading_pair}): {e}. Accumulated USDC: {self.usdc_accumulated_for_eur:.2f}, Amount to convert: {amount_to_convert:.2f}")
            except ccxt.NetworkError as e:
                self.logger.error(f"Network error during EUR conversion ({actual_trading_pair}): {e}")
            except ccxt.ExchangeError as e:
                self.logger.error(f"Exchange error during EUR conversion ({actual_trading_pair}): {e}")
            except Exception as e:
                self.logger.error(f"Unexpected error during EUR conversion ({actual_trading_pair}): {str(e)}", exc_info=True)
        elif self.usdc_accumulated_for_eur > 0: # Log only if there's something in buffer but not enough to convert
             self.logger.info(f"USDC for EUR buffer ({self.usdc_accumulated_for_eur:.2f}) is below threshold ({USDC_ACCUMULATION_THRESHOLD_FOR_EUR:.2f} USDC). Holding.")

    def calculate_current_balance(self):
        """Calculate current balance including profits and savings"""
        return (self.initial_allocation +
                self.realized_profit +
                self.wallet_savings +
                self.profit_reinvested)

    def calculate_open_pnl(self):
        open_pnl = 0
        for pos_id, position in self.positions.items():
            # Calculate the value of each open position at the current price
            total_amount = position['base_amount']
            for order in position['safety_orders']:
                total_amount += order['amount']
            current_value = total_amount * self.current_price

            # Subtract the total invested in the position to get the PnL
            # Use total_cost + total_fees if total_invested is not available
            if 'total_invested' in position:
                invested = position['total_invested']
            else:
                invested = position['total_cost'] + position['total_fees']

            open_pnl += current_value - invested
        return open_pnl

    def verify_account_state(self, to_console=False, to_file=False):
        """Verify and log the current state of the account"""
        if not (to_console or to_file):
            return

        total_position_value = 0
        if self.current_price is None:
            return

        for pos_id, position in self.positions.items():
            pos_amount = position['base_amount']
            for order in position['safety_orders']:
                pos_amount += order['amount']
            total_position_value += pos_amount * self.current_price

        state_message = f"""
        Account State:
        Balance: {self.balance:.2f} USDC
        Total Invested: {self.total_invested:.2f} USDC
        Total Position Value: {total_position_value:.2f} USDC
        Wallet Savings: {self.wallet_savings:.2f} USDC
        Profit Reinvested: {self.profit_reinvested:.2f} USDC
        Pending Profit: {self.pending_profit:.2f} USDC
        USDC for EUR Savings: {self.usdc_accumulated_for_eur:.2f} USDC
        EUR Savings: {self.eur_savings_total:.4f} EUR
        Active Positions: {len(self.positions)}
        Current Price: {self.current_price:.2f} USDC
        """

        if to_console:
            symbol_color = SYMBOL_COLORS.get(self.symbol, DEFAULT_SYMBOL_COLOR)
            print(
                f"\n{symbol_color}{'=' * 20} {self.symbol} ACCOUNT STATE {'=' * 20}{Style.RESET_ALL}")
            print(f"{symbol_color}{state_message}{Style.RESET_ALL}")
            print(f"{symbol_color}{'=' * 60}{Style.RESET_ALL}")

        if to_file:
            self.logger.info(f"Account State: {state_message}")
            # Force flush
            for handler in logger.handlers:
                handler.flush()

        # Calculate win rate
        total_trades = self.winning_trades + self.losing_trades
        win_rate = (
            self.winning_trades /
            total_trades *
            100) if total_trades > 0 else 0

        # Add daily balance notification
        if datetime.now().hour == 0 and datetime.now().minute == 0:  # At midnight
            self.telegram.notify_balance_update(
                symbol=self.symbol,
                realized_profit=self.realized_profit,
                unrealized_profit=self.calculate_open_pnl(),
                total_fees=self.total_fees_paid,
                reinvested=self.profit_reinvested,
                saved=self.wallet_savings,
                win_rate=win_rate,
                active_positions=len(self.positions)
            )

    def check_safety_orders(self, current_price, timestamp):
        """Check and execute safety orders for existing positions"""
        self.logger.info(
            f"Checking safety orders for all positions at price {current_price:.8f}")

        for pos_id, position in list(self.positions.items()):
            try:
                if position.get('status') == 'CLOSED':
                    continue

                profit_result = self.calculate_position_profit(
                    position, current_price)
                raw_profit_pct = profit_result['raw_profit_pct']

                current_drawdown = -raw_profit_pct

                self.logger.info(
                    f"Position {pos_id} - Current drawdown: {current_drawdown:.2f}%, Required for first SO: {SAFETY_ORDER_STEP:.2f}%")

                symbol_allocation = self.shared_balance.get_symbol_allocation(
                    self.symbol)
                total_allocated = sum(
                    self.shared_balance.allocated_funds.values())

                safety_orders_count = len(position['safety_orders'])
                if safety_orders_count >= MAX_SAFETY_ORDERS:
                    self.logger.info(
                        f"Position {pos_id} - Maximum safety orders reached ({MAX_SAFETY_ORDERS})")
                    continue

                proposed_order = (BASE_ORDER_SIZE + self.base_order_increment) * \
                    (SAFETY_ORDER_VOLUME_SCALE ** (safety_orders_count + 1))
                total_cost = proposed_order

                self.logger.info(
                    f"Position {pos_id} - Proposed safety order size: {proposed_order:.2f} USDC")

                # Ensure timestamp is a datetime object for comparison
                current_time = timestamp
                if isinstance(timestamp, str):
                    try:
                        current_time = datetime.strptime(
                            timestamp, '%Y-%m-%d %H:%M:%S.%f')
                    except ValueError:
                        try:
                            current_time = datetime.strptime(
                                timestamp, '%Y-%m-%d %H:%M:%S')
                        except ValueError:
                            self.logger.warning(
                                f"Could not parse timestamp: {timestamp}")
                            current_time = datetime.now()

                # Get the time since base order - handle both timestamp formats
                base_order_time = position['timestamp']
                if isinstance(base_order_time, str):
                    try:
                        base_order_time = datetime.strptime(
                            base_order_time, '%Y-%m-%d %H:%M:%S.%f')
                    except ValueError:
                        try:
                            base_order_time = datetime.strptime(
                                base_order_time, '%Y-%m-%d %H:%M:%S')
                        except ValueError:
                            self.logger.warning(
                                f"Could not parse base order timestamp: {base_order_time}")
                            base_order_time = datetime.now()

                time_since_base = (
                    current_time - base_order_time).total_seconds()
                self.logger.info(
                    f"Position {pos_id} - Time since base order: {time_since_base:.0f} seconds")

                # Minimum time after base order before first safety order
                if safety_orders_count == 0 and time_since_base < 120:
                    self.logger.info(
                        f"Position {pos_id} - Waiting for minimum time after base order: {time_since_base:.0f}s/60s")
                    continue

                required_drawdown = SAFETY_ORDER_STEP * \
                    (safety_orders_count + 1)
                self.logger.info(
                    f"Position {pos_id} - Required drawdown for SO #{safety_orders_count + 1}: {required_drawdown:.2f}%")

                base_price = float(position['base_price'])
                price_decreased = current_price < base_price

                if not price_decreased:
                    self.logger.info(
                        f"Position {pos_id} - Skipping safety order - current price ({current_price:.8f}) is not lower than base price ({base_price:.8f})")
                    continue

                self.logger.info(
                    f"Position {pos_id} - Current drawdown: {current_drawdown:.2f}%, Required: {required_drawdown:.2f}%")

                if current_drawdown >= required_drawdown:
                    self.logger.info(
                        f"Position {pos_id} - Drawdown condition met for safety order")

                    self.logger.info(f"""
                    Position {pos_id} - Fund checks:
                    Symbol Allocation: {symbol_allocation:.2f}/{MAX_POSITION_SIZE} USDC
                    Total Allocated: {total_allocated:.2f}/{MAX_TOTAL_INVESTMENT} USDC
                    Available Balance: {self.shared_balance.available_balance:.2f} USDC
                    Required: {total_cost:.2f} USDC
                    """)

                    if (symbol_allocation + total_cost <= MAX_POSITION_SIZE and
                        total_allocated + total_cost <= MAX_TOTAL_INVESTMENT and
                            self.shared_balance.available_balance >= total_cost):

                        if safety_orders_count > 0:
                            last_safety = position['safety_orders'][-1]
                            last_price = float(last_safety['price'])

                            try:
                                last_time = datetime.strptime(
                                    last_safety['timestamp'], '%Y-%m-%d %H:%M:%S.%f')
                            except ValueError:
                                try:
                                    last_time = datetime.strptime(
                                        last_safety['timestamp'], '%Y-%m-%d %H:%M:%S')
                                except ValueError:
                                    self.logger.warning(
                                        f"Could not parse safety order timestamp: {last_safety['timestamp']}")
                                    last_time = datetime.now() - timedelta(minutes=2)

                            time_since_last = (
                                current_time - last_time).total_seconds()

                            if time_since_last < 60:
                                self.logger.info(
                                    f"Position {pos_id} - Waiting for minimum time between safety orders: {time_since_last:.0f}s/60s")
                                continue

                            if current_price >= last_price:
                                self.logger.info(
                                    f"Position {pos_id} - Skipping safety order - current price ({current_price:.8f}) is not lower than last safety order price ({last_price:.8f})")
                                continue

                            price_change = (
                                (current_price - last_price) / last_price) * 100
                            required_price_movement = SAFETY_ORDER_STEP

                            self.logger.info(f"""
                            Position {pos_id} - Price Movement Check:
                            Time Since Last SO: {time_since_last:.0f}s
                            Price Change from Last SO: {price_change:.2f}%
                            Required Movement: {required_price_movement:.2f}%
                            Last SO Price: {last_price:.8f}
                            Current Price: {current_price:.8f}
                            """)

                            if abs(price_change) < required_price_movement:
                                self.logger.info(
                                    f"Position {pos_id} - Skipping SO - Need {required_price_movement:.2f}% movement, got {abs(price_change):.2f}%")
                                continue

                        self.logger.info(f"""
                        Position {pos_id} - Executing Safety Order:
                        Position ID: {pos_id[:8]}
                        Order Size: {proposed_order:.2f} USDC
                        Current Drawdown: {current_drawdown:.2f}%
                        Safety Order #: {safety_orders_count + 1}
                        Required Drawdown: {required_drawdown:.2f}%
                        """)

                        amount_to_buy = proposed_order / current_price

                        self.execute_trade(
                            side='BUY',
                            order_type='SAFETY',
                            amount=amount_to_buy,
                            price=current_price,
                            parent_id=pos_id)
                        time.sleep(1)

                        total_amount = position['base_amount']
                        total_cost = position['base_cost']
                        for order in position['safety_orders']:
                            total_amount += order['amount']
                            total_cost += order['cost']

                        avg_price = total_cost / total_amount if total_amount > 0 else 0
                        total_position_size = total_cost

                        self.telegram.notify_safety_order(
                            symbol=self.symbol,
                            position_id=pos_id,
                            safety_order_num=safety_orders_count + 1,
                            price=current_price,
                            total_position_size=total_position_size,
                            avg_price=avg_price,
                            exchange="Binance"
                        )
                    else:
                        self.logger.info(f"""
                        Position {pos_id} - Safety order skipped - limits reached:
                        Symbol Allocation: {symbol_allocation:.2f}/{MAX_POSITION_SIZE} USDC
                        Total Allocated: {total_allocated:.2f}/{MAX_TOTAL_INVESTMENT} USDC
                        Available Balance: {self.shared_balance.available_balance:.2f} USDC
                        Required: {total_cost:.2f} USDC
                        """)
                else:
                    self.logger.info(
                        f"Position {pos_id} - Insufficient decrease for safety order: {current_drawdown:.2f}% < {required_drawdown:.2f}%")

            except Exception as e:
                self.logger.error(
                    f"Error checking safety orders for position {pos_id}: {str(e)}")
                # Add more debug info
                self.logger.error(f"Position data: {position}")
                continue

    def fetch_current_price(self):
        try:
            ticker = self.exchange.fetch_ticker(self.exchange_symbol)
            return ticker['last']
        except Exception as e:
            self.logger.error(f"Error fetching current price: {str(e)}")
            return None

    def print_quick_status(self, current_price, timestamp):
        symbol_color = SYMBOL_COLORS.get(self.symbol, DEFAULT_SYMBOL_COLOR)
        formatted_timestamp = self.format_timestamp(timestamp)
        print(
            f"\n{symbol_color}{'=' * 20} {self.symbol} - {formatted_timestamp} {'=' * 20}{Style.RESET_ALL}")
        print(
            f"{symbol_color}[{self.symbol}] Price: {current_price:.8f} USDC{Style.RESET_ALL}")

        history_size = len(self.candle_history)
        if history_size == 0:
            data_status = f"{Fore.RED}No historical data available yet{Style.RESET_ALL}"
        elif history_size < self.min_required_candles:
            data_status = f"{Fore.YELLOW}Building history ({history_size}/{self.min_required_candles} candles){Style.RESET_ALL}"
        else:
            data_status = f"{Fore.GREEN}Data ready for trading ({history_size} candles){Style.RESET_ALL}"

        print(f"{symbol_color}Data Status: {data_status}{Style.RESET_ALL}")

        try:
            if not self.candle_history.empty:
                df = self.candle_history.copy()
                df = self.calculate_indicators(df)

                if len(df) > 0:
                    latest = df.iloc[-1]

                    is_rsi_valid = not pd.isna(latest['rsi']) if isinstance(
                        latest['rsi'], (float, int)) else not latest['rsi'].isna().any()
                    is_bb_valid = not pd.isna(latest['lower_band']) if isinstance(
                        latest['lower_band'], (float, int)) else not latest['lower_band'].isna().any()

                    if is_rsi_valid and is_bb_valid:
                        rsi_value = float(
                            latest['rsi']) if isinstance(
                            latest['rsi'], (pd.Series, np.ndarray)) else latest['rsi']
                        lower_band = float(
                            latest['lower_band']) if isinstance(
                            latest['lower_band'], (pd.Series, np.ndarray)) else latest['lower_band']
                        close_price = float(
                            latest['close']) if isinstance(
                            latest['close'], (pd.Series, np.ndarray)) else latest['close']

                        price_bb_ratio = close_price / lower_band

                        print(
                            f"\n{symbol_color}CURRENT INDICATORS:{Style.RESET_ALL}")
                        print(
                            f"  RSI: {Fore.CYAN}{rsi_value:.2f}{Style.RESET_ALL} (Buy below {RSI_BUY_THRESHOLD})")
                        print(
                            f"  Price/BB Ratio: {Fore.CYAN}{price_bb_ratio:.4f}{Style.RESET_ALL} (Buy below 1.03)")

                        rsi_below = rsi_value < (RSI_BUY_THRESHOLD + 2)
                        bb_below = price_bb_ratio < 1.03

                        rsi_color = Fore.GREEN if rsi_below else Fore.RED if rsi_value > (
                            RSI_BUY_THRESHOLD + 10) else Fore.YELLOW
                        bb_color = Fore.GREEN if bb_below else Fore.RED if price_bb_ratio > 1.05 else Fore.YELLOW

                        print(
                            f"  Entry conditions: RSI [{rsi_color}{'SI' if rsi_below else 'NO'}{Style.RESET_ALL}] " +
                            f"BB [{bb_color}{'SI' if bb_below else 'NO'}{Style.RESET_ALL}]")
                    else:
                        print(
                            f"\n{Fore.YELLOW}Waiting for valid indicators...{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}Error displaying indicators: {str(e)}{Style.RESET_ALL}")

        if not self.positions:
            print(
                f"\n{symbol_color}NO ACTIVE POSITIONS - Waiting for entry conditions{Style.RESET_ALL}")
            print(
                f"{symbol_color}RSI < {RSI_BUY_THRESHOLD} and price near lower Bollinger Band{Style.RESET_ALL}")
            print(f"{symbol_color}{'=' * 60}{Style.RESET_ALL}")
            return

        print(f"\n{symbol_color}ACTIVE POSITIONS:{Style.RESET_ALL}")

        for pos_id, position in self.positions.items():
            profit_result = self.calculate_position_profit(
                position, current_price)

            raw_profit_pct = profit_result['raw_profit_pct']
            current_profit_pct = profit_result['profit_pct']

            # Calculate progress based on raw profit (without fees)
            progress = (raw_profit_pct / config.PROFIT_TARGET) * 100
            progress = min(100, max(0, progress))

            bar_length = 23
            filled_length = int(bar_length * progress / 100)
            bar = '█' * filled_length + '░' * (bar_length - filled_length)

            # Determine color based on raw profit
            if raw_profit_pct >= 0:
                color = Fore.GREEN if raw_profit_pct > 0 else Fore.WHITE
            else:
                # Scale red intensity based on how negative
                if raw_profit_pct < -3:
                    color = Fore.RED
                elif raw_profit_pct < -1.5:
                    color = Fore.LIGHTRED_EX
                else:
                    color = Fore.YELLOW

            safety_count = len(position['safety_orders'])

            print(f"{symbol_color}{'~' * 60}{Style.RESET_ALL}")

            print(
                f"{symbol_color}Position {pos_id[:8]} - {safety_count} safety orders{Style.RESET_ALL}")

            base_price = float(position['base_price'])

            if safety_count > 0:
                total_cost = float(position['base_cost'])
                total_amount = float(position['base_amount'])

                for so in position['safety_orders']:
                    total_cost += float(so['cost'])
                    total_amount += float(so['amount'])

                avg_entry_price = total_cost / total_amount if total_amount > 0 else 0

                print(f"  Entry: {Fore.CYAN}{base_price:.5f}{Style.RESET_ALL} | Avg: {Fore.CYAN}{avg_entry_price:.5f}{Style.RESET_ALL} | Current: {Fore.CYAN}{current_price:.5f}{Style.RESET_ALL}")
            else:
                # For positions without safety orders, show entry price and
                # current price
                print(
                    f"  Entry: {Fore.CYAN}{base_price:.5f}{Style.RESET_ALL} | Current: {Fore.CYAN}{current_price:.5f}{Style.RESET_ALL}")

            print(
                f"  Profit: {color}{raw_profit_pct:+.2f}%{Style.RESET_ALL} (Target: {config.PROFIT_TARGET}%)")
            print(
                f"  Progress: {color}|{bar}| {progress:.1f}%{Style.RESET_ALL}")

        print(f"{symbol_color}{'=' * 60}{Style.RESET_ALL}")

    def send_periodic_summary(self):
        try:
            current_price = self.fetch_current_price()
            if not current_price:
                self.logger.error(
                    "Failed to fetch current price for periodic summary")
                return

            self.update_unrealized_profits(current_price)

            total_trades = self.winning_trades + self.losing_trades
            win_rate = (
                self.winning_trades /
                total_trades *
                100) if total_trades > 0 else 0

            active_positions_details = {}
            for pos_id, pos in self.positions.items():
                profit_result = self.calculate_position_profit(
                    pos, current_price)
                current_profit = profit_result['profit_pct']

                total_cost = pos['base_cost']
                total_amount = pos['base_amount']

                for so in pos['safety_orders']:
                    total_cost += so['cost']
                    total_amount += so['amount']

                avg_entry_price = total_cost / total_amount if total_amount > 0 else 0

                active_positions_details[pos_id] = {
                    'current_profit': current_profit,
                    'base_price': float(pos['base_price']),
                    'avg_entry_price': avg_entry_price,
                    'current_price': current_price,
                    'safety_orders': len(pos['safety_orders'])
                }

            self.telegram.notify_periodic_summary(
                symbol=self.symbol,
                positions=self.positions,
                realized_profit=self.realized_profit,
                unrealized_profit=self.unrealized_profit,
                total_fees=self.total_fees_paid,
                win_rate=win_rate,
                active_positions_details=active_positions_details,
                wallet_savings=self.wallet_savings,
                profit_reinvested=self.profit_reinvested,
                exchange="Binance"
            )

            self.logger.info(
                f"Sent periodic summary. Realized profit: {self.realized_profit:.2f}, Unrealized: {self.unrealized_profit:.2f}")
        except Exception as e:
            self.logger.error(f"Error sending periodic summary: {str(e)}")
            self.telegram.notify_error(
                self.symbol,
                f"Failed to send periodic summary: {str(e)}",
                exchange="Binance")

    def check_bnb_balance(self):
        """Check BNB balance and update fee settings"""
        try:
            # Only check BNB balance every hour to avoid excessive API calls
            current_time = datetime.now()
            if hasattr(
                    self,
                    'last_bnb_check') and (
                    current_time -
                    self.last_bnb_check).total_seconds() < 3600:
                return

            self.last_bnb_check = current_time

            # Update trading fees and check BNB balance
            self.update_trading_fees()

            # Auto-purchase BNB if balance is low and auto-purchase is enabled
            if hasattr(
                    self,
                    'using_bnb_for_fees') and self.using_bnb_for_fees and hasattr(
                    self,
                    'auto_purchase_bnb') and self.auto_purchase_bnb:
                # Get current BNB balance
                account_info = self.exchange.fetch_balance()
                bnb_balance = 0
                if 'BNB' in account_info:
                    bnb_balance = float(
                        account_info['BNB']['free']) if 'free' in account_info['BNB'] else 0

                # If BNB balance is below threshold, try to purchase more
                if bnb_balance < 0.005:
                    self.logger.info(
                        f"BNB balance is low ({bnb_balance:.6f}). Attempting to purchase more.")
                    self.auto_purchase_bnb()

            self.logger.info(
                f"BNB balance check completed. Using BNB for fees: {self.using_bnb_for_fees}")
        except Exception as e:
            self.logger.error(f"Error checking BNB balance: {str(e)}")

    def auto_purchase_bnb(self):
        """
        Automatically purchase BNB to pay for trading fees when balance is low.
        This function will use a small portion of wallet savings to buy BNB.
        """
        try:
            # Constants for BNB purchase
            BNB_PURCHASE_AMOUNT_USDC = 10.0  # Amount in USDC to spend on BNB
            MIN_SHARED_USDC_FOR_BNB = 5.0  # Minimum shared USDC balance required to purchase BNB

            # Check if we have enough shared USDC to purchase BNB
            if self.shared_balance.available_balance < MIN_SHARED_USDC_FOR_BNB:
                self.logger.warning(
                    f"Not enough shared USDC ({self.shared_balance.available_balance:.2f}) to purchase BNB. Minimum required: {MIN_SHARED_USDC_FOR_BNB:.2f} USDC")
                return False

            # Calculate amount to spend (limited by available shared USDC)
            purchase_amount = min(
                BNB_PURCHASE_AMOUNT_USDC,
                self.shared_balance.available_balance -
                1.0)  # Keep at least 1 USDC in shared balance

            if purchase_amount < 1.0: # Assuming 1 USDC is the absolute minimum practical amount for a BNB trade
                self.logger.warning(
                    f"Calculated BNB purchase amount ({purchase_amount:.2f} USDC) is too low due to insufficient shared balance. Aborting BNB purchase.")
                return False

            # Get current BNB/USDC price
            try:
                ticker = self.exchange.fetch_ticker('BNB/USDC')
                bnb_price = ticker['last']

                # Calculate BNB amount to buy
                bnb_amount = purchase_amount / bnb_price

                # Log the purchase attempt
                self.logger.info(
                    f"Attempting to purchase {bnb_amount:.6f} BNB for {purchase_amount:.2f} USDC at price {bnb_price:.2f}")

                # Execute the purchase
                order = self.exchange.create_order(
                    symbol='BNB/USDC',
                    type='market',
                    side='buy',
                    amount=bnb_amount
                )

                # Get executed price and amount
                executed_price = float(
                    order['price']) if 'price' in order and order['price'] else bnb_price
                executed_amount = float(
                    order['amount']) if 'amount' in order and order['amount'] else bnb_amount
                actual_cost = executed_price * executed_amount

                # Update shared balance
                with self.shared_balance.lock:
                    self.shared_balance.available_balance -= actual_cost

                # Log the successful purchase
                self.logger.info(
                    f"Successfully purchased {executed_amount:.6f} BNB for {actual_cost:.2f} USDC")
                self.logger.info(
                    f"Shared balance updated. Available: {self.shared_balance.available_balance:.2f} USDC")

                # Send Telegram notification
                if hasattr(self, 'telegram') and self.telegram is not None:
                    self.telegram.notify_bnb_purchase(
                        amount_usdc=actual_cost,
                        amount_bnb=executed_amount,
                        price=executed_price,
                        exchange="Binance"
                    )

                return True

            except Exception as e:
                error_message = str(e)
                self.logger.error(f"Error purchasing BNB: {error_message}")

                # Send Telegram notification about the failure
                if hasattr(self, 'telegram') and self.telegram is not None:
                    self.telegram.notify_bnb_purchase_failed(
                        amount_usdc=purchase_amount,
                        error_message=error_message,
                        exchange="Binance"
                    )

                return False

        except Exception as e:
            self.logger.error(f"Error in auto_purchase_bnb: {str(e)}")
            return False

    def run_trading_loop(self):
        """Main trading loop"""
        self.logger.info(f"Starting trading loop for {self.symbol}")

        last_check_time = datetime.now()
        last_summary_time = datetime.now()
        self.last_bnb_check = datetime.now()

        while True:
            try:
                current_time = datetime.now()

                if (current_time -
                        last_summary_time).total_seconds() > 1800:  # 30 minutes
                    self.send_periodic_summary()
                    last_summary_time = current_time

                self.check_bnb_balance()

                if (current_time - last_check_time).total_seconds() < CHECK_INTERVAL:
                    time.sleep(1)
                    continue

                df = self.fetch_recent_candles()
                if df.empty:
                    self.logger.warning("No candle data available")
                    time.sleep(5)
                    continue

                df = self.calculate_indicators(df)
                if not self.validate_indicators(df):
                    time.sleep(5)
                    continue

                current_price = self.fetch_current_price()
                if not current_price:
                    time.sleep(5)
                    continue

                self.current_price = current_price

                for pos_id, position in list(self.positions.items()):
                    profit_result = self.calculate_position_profit(
                        position, current_price)
                    raw_profit_pct = profit_result['raw_profit_pct']

                    if raw_profit_pct >= config.PROFIT_TARGET:
                        self.logger.info(
                            f"Position {pos_id} reached profit target: {raw_profit_pct:.2f}% (target: {config.PROFIT_TARGET}%)")
                        self.logger.info(
                            f"Decision price: {current_price:.8f}, Raw profit: {profit_result['raw_profit']:.8f} USDC")
                        self.close_position(pos_id, current_price)
                        time.sleep(WAIT_TIME_AFTER_CLOSE)
                        continue

                if self.check_entry_conditions(df.iloc[-1]):
                    base_order_size = BASE_ORDER_SIZE + self.base_order_increment
                    amount_to_buy = base_order_size / current_price

                    # Log the base order size with reinvestment
                    if self.base_order_increment > 0:
                        self.logger.info(
                            f"Creating base order with size {base_order_size:.2f} USDC (includes {self.base_order_increment:.2f} USDC reinvested profit)")
                    else:
                        self.logger.info(
                            f"Creating base order with size {base_order_size:.2f} USDC")

                    self.execute_trade(
                        side='BUY',
                        order_type='MARKET',
                        amount=amount_to_buy,
                        price=current_price)

                self.check_safety_orders(current_price, current_time)

                # Calcular drawdown actual y actualizar monitor de rendimiento
                current_drawdown = self.performance_monitor.calculate_drawdown(self.positions, current_price)
                if current_drawdown < 0:
                    self.logger.info(f"Drawdown actual para {self.symbol}: {current_drawdown:.2f}%")
                
                # Actualizar datos de rendimiento cada hora
                if (current_time - self.performance_monitor.last_analysis_time).total_seconds() >= 3600:
                    self.performance_monitor.save_performance_data()
                    self.performance_monitor.last_analysis_time = current_time
                    
                    # Generar informe de rendimiento periódico si hay suficientes operaciones
                    if self.performance_monitor.total_trades > 0:
                        performance_report = self.performance_monitor.generate_performance_report()
                        self.logger.info(f"Informe de rendimiento periódico para {self.symbol}:\n{performance_report}")

                self.update_unrealized_profits(current_price)

                self.print_quick_status(current_price, current_time)

                should_print_to_console = (
                    current_time - last_summary_time).total_seconds() >= 1800
                self.verify_account_state(
                    to_console=should_print_to_console, to_file=True)

                time.sleep(CHECK_INTERVAL)

            except Exception as e:
                self.logger.error(f"Error in trading loop: {str(e)}")
                time.sleep(5)

    def print_summary(self):
        with self.lock:
            try:
                trades_df = pd.read_csv(self.csv_file)
                close_trades = trades_df[trades_df['Order Type'] == 'CLOSE']

                winning_trades = len(close_trades[close_trades['Profit'] > 0])
                losing_trades = len(close_trades[close_trades['Profit'] <= 0])
                total_trades = winning_trades + losing_trades
                win_rate = (
                    winning_trades /
                    total_trades *
                    100) if total_trades > 0 else 0

                current_balance = self.calculate_current_balance()
                total_profit = self.realized_profit
                profit_percentage = (
                    total_profit /
                    self.initial_allocation *
                    100) if self.initial_allocation > 0 else 0
                open_pnl = self.calculate_open_pnl()

                summary = f"""
                Summary Data for {self.symbol}:
                Initial Balance: {self.initial_allocation:.2f} USDC
                Current Balance: {current_balance:.2f} USDC
                Total Profit: {total_profit:.2f}
                Profit %: {profit_percentage:.2f}%
                Win Rate: {win_rate:.2f}%
                Open PnL: {open_pnl:.2f}
                Total Fees Paid: {self.total_fees_paid:.2f} USDC

                Profit Distribution:
                Total Reinvested: {self.profit_reinvested:.2f} USDC
                Wallet Savings: {self.wallet_savings:.2f} USDC
                Pending Profit: {self.pending_profit:.2f} USDC
                USDC for EUR Savings: {self.usdc_accumulated_for_eur:.2f} USDC
                EUR Savings: {self.eur_savings_total:.4f} EUR
                Current Base Order Size: {(BASE_ORDER_SIZE + self.base_order_increment):.2f} USDC

                Trading Statistics:
                Winning Trades: {winning_trades}
                Losing Trades: {losing_trades}
                """

                print(summary)
                self.logger.info(summary)

            except Exception as e:
                self.logger.error(f"Error generating summary: {e}")
                print("Error generating summary. Check the logs for details.")

    def format_timestamp(
            self,
            timestamp=None,
            for_display=False,
            for_storage=False):
        if timestamp is None:
            timestamp = datetime.now()

        if isinstance(timestamp, str):
            try:
                timestamp = datetime.strptime(
                    timestamp, '%Y-%m-%d %H:%M:%S.%f')
            except ValueError:
                try:
                    timestamp = datetime.strptime(
                        timestamp, '%Y-%m-%d %H:%M:%S')
                except ValueError:
                    self.logger.warning(
                        f"Could not parse timestamp: {timestamp}")
                    timestamp = datetime.now()

        if for_storage:
            return timestamp.strftime('%Y-%m-%d %H:%M:%S')
        elif for_display:
            return timestamp.strftime('%Y-%m-%d %H:%M:%S')
        else:
            return timestamp.strftime('%Y-%m-%d %H:%M:%S')

    def update_trading_fees(self):
        """Update trading fees from the exchange and check BNB balance"""
        try:
            fees = self.exchange.fetch_trading_fees()

            if self.exchange_symbol in fees:
                symbol_fees = fees[self.exchange_symbol]
                if 'maker' in symbol_fees and symbol_fees['maker'] is not None:
                    self.maker_fee = float(symbol_fees['maker']) * 100
                if 'taker' in symbol_fees and symbol_fees['taker'] is not None:
                    self.taker_fee = float(symbol_fees['taker']) * 100

            # Check BNB balance and if it's being used for fees
            account_info = self.exchange.fetch_balance()

            # Better detection of BNB fee discount usage
            self.using_bnb_for_fees = False
            bnb_balance = 0

            # Check if 'BNB' is in the balance
            if 'BNB' in account_info:
                bnb_balance = float(
                    account_info['BNB']['free']) if 'free' in account_info['BNB'] else 0

                # Check if user has BNB and if the "using BNB for fees" setting is enabled
                # For Binance, we need to check the account settings
                try:
                    account_settings = self.exchange.private_get_account()
                    if 'canTrade' in account_settings and account_settings['canTrade']:
                        # Check if "using BNB for fees" is enabled
                        if 'enableSpotAndMarginTrading' in account_settings and account_settings[
                                'enableSpotAndMarginTrading']:
                            self.using_bnb_for_fees = True
                except Exception as e:
                    self.logger.warning(
                        f"Could not verify BNB fee discount settings: {str(e)}")
                    # Fallback: If user has BNB balance, assume they're using
                    # it for fees
                    self.using_bnb_for_fees = bnb_balance > 0

            # If using BNB, apply the discount
            if self.using_bnb_for_fees:
                self.maker_fee = self.maker_fee * (1 - BNB_DISCOUNT_RATE)
                self.taker_fee = self.taker_fee * (1 - BNB_DISCOUNT_RATE)
                self.logger.info(
                    f"Using BNB for fees with {BNB_DISCOUNT_RATE*100}% discount. BNB Balance: {bnb_balance:.4f}")

                # Check if BNB balance is low (less than approximately 5 trades worth of fees)
                # Assuming average trade size of 100 USDC with 0.1% fee = 0.1 USDC per trade
                # 5 trades = 0.5 USDC, and BNB/USDC rate is roughly 500, so
                # 0.001 BNB
                if bnb_balance < 0.005:
                    self.logger.warning(
                        f"LOW BNB BALANCE ALERT: {bnb_balance:.6f} BNB remaining")
                    # Send notification about low BNB balance if telegram is
                    # initialized
                    if hasattr(self, 'telegram') and self.telegram is not None:
                        self.telegram.notify_low_bnb_balance(
                            bnb_balance, exchange="Binance")
            else:
                self.logger.info(
                    f"Not using BNB for fees. Standard fees apply. BNB Balance: {bnb_balance:.4f}")

            self.logger.info(
                f"Updated trading fees: Maker Fee: {self.maker_fee:.6f}%, Taker Fee: {self.taker_fee:.6f}%")
        except Exception as e:
            self.logger.warning(
                f"Error updating trading fees: {str(e)}. Using default values.")
            self.maker_fee = MAKER_FEE_RATE
            self.taker_fee = TAKER_FEE_RATE


def ensure_directories():
    directories = [
        'logs',
        'logs/backups',
        'logs/trades',
        'logs/performance',  # Directorio para archivos de rendimiento por par
        'backups/states',
        'backups/trades'
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)


def compare_trading_pairs_performance(bots):
    """Compara el rendimiento entre diferentes pares de trading y genera un informe"""
    performance_data = []
    
    for symbol, bot in bots.items():
        if hasattr(bot, 'performance_monitor'):
            perf = bot.performance_monitor
            win_rate = perf.win_rate if hasattr(perf, 'win_rate') else 0
            total_profit = perf.total_profit - perf.total_loss
            avg_profit = perf.avg_profit_per_trade if hasattr(perf, 'avg_profit_per_trade') else 0
            
            performance_data.append({
                'symbol': symbol,
                'total_trades': perf.total_trades,
                'win_rate': win_rate,
                'total_profit': total_profit,
                'avg_profit': avg_profit,
                'max_drawdown': perf.max_drawdown
            })
    
    # Ordenar por rentabilidad total
    performance_data.sort(key=lambda x: x['total_profit'], reverse=True)
    
    # Generar informe
    report = "\n===== COMPARACIÓN DE RENDIMIENTO ENTRE PARES =====\n"
    report += "Ordenado por beneficio total:\n\n"
    
    for idx, data in enumerate(performance_data, 1):
        report += f"{idx}. {data['symbol']}:\n"
        report += f"   Operaciones: {data['total_trades']}\n"
        report += f"   Tasa de éxito: {data['win_rate']:.2f}%\n"
        report += f"   Beneficio total: {data['total_profit']:.2f} USDC\n"
        report += f"   Beneficio promedio: {data['avg_profit']:.2f} USDC/operación\n"
        report += f"   Máximo drawdown: {data['max_drawdown']:.2f}%\n\n"
    
    # Registrar en el log
    logger = logging.getLogger("performance_comparison")
    logger.info(report)
    
    # Guardar en archivo CSV
    csv_file = os.path.join('logs', 'performance', f"pairs_comparison_{datetime.now().strftime('%Y%m%d')}.csv")
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Symbol', 'Total Trades', 'Win Rate', 'Total Profit', 'Avg Profit/Trade', 'Max Drawdown'])
        for data in performance_data:
            writer.writerow([
                data['symbol'],
                data['total_trades'],
                f"{data['win_rate']:.2f}%",
                f"{data['total_profit']:.2f}",
                f"{data['avg_profit']:.2f}",
                f"{data['max_drawdown']:.2f}%"
            ])
    
    # Enviar informe de comparación por Telegram
    try:
        # Usar el primer bot disponible para enviar la notificación
        if bots and len(bots) > 0:
            first_bot = next(iter(bots.values()))
            if hasattr(first_bot, 'telegram') and first_bot.telegram is not None:
                # Crear mensaje para Telegram
                telegram_msg = "📊 COMPARACIÓN DE RENDIMIENTO ENTRE PARES\n\n"
                
                # Añadir emojis para clasificar el rendimiento
                for idx, data in enumerate(performance_data, 1):
                    # Determinar emoji basado en rentabilidad
                    if data['total_profit'] > 0:
                        emoji = "🟢" if data['win_rate'] > 50 else "🟡"
                    else:
                        emoji = "🔴"
                        
                    telegram_msg += f"{emoji} {idx}. {data['symbol']}:\n"
                    telegram_msg += f"   Operaciones: {data['total_trades']}\n"
                    telegram_msg += f"   Tasa de éxito: {data['win_rate']:.2f}%\n"
                    telegram_msg += f"   Beneficio: {data['total_profit']:.2f} USDC\n"
                    telegram_msg += f"   Drawdown máx: {data['max_drawdown']:.2f}%\n\n"
                
                # Enviar mensaje
                first_bot.telegram.notify_custom_message(telegram_msg)
                logger.info("Informe de comparación enviado por Telegram")
    except Exception as e:
        logger.error(f"Error al enviar informe de comparación por Telegram: {str(e)}")
    
    return report

def manage_combined_profit_distribution(bots):
    total_realized = 0
    total_reinvested = 0
    total_savings = 0

    for symbol, bot in bots.items():
        total_realized += bot.realized_profit
        total_reinvested += bot.profit_reinvested
        total_savings += bot.wallet_savings
    
    # Generar informe de comparación de rendimiento cada 24 horas
    current_time = datetime.now()
    last_comparison_time = getattr(manage_combined_profit_distribution, 'last_comparison_time', None)
    if last_comparison_time is None or (current_time - last_comparison_time).total_seconds() > 86400:
        compare_trading_pairs_performance(bots)
        manage_combined_profit_distribution.last_comparison_time = current_time

    available_profit = total_realized - total_reinvested - total_savings

    logger = logging.getLogger("combined_profit")
    logger.info(f"Combined profit distribution check:")
    logger.info(f"Total realized profit: {total_realized:.2f} USDC")
    logger.info(f"Total reinvested: {total_reinvested:.2f} USDC")
    logger.info(f"Total savings: {total_savings:.2f} USDC")
    logger.info(f"Available for distribution: {available_profit:.2f} USDC")

    reinvest_threshold = 5.0  # USD
    savings_threshold = 3.0   # USD

    # Determine the last action based on the ratio of reinvested to savings
    # If we have more savings than reinvestments (proportionally), we should reinvest next
    # Otherwise, we should save next
    reinvest_ratio = total_reinvested / \
        reinvest_threshold if reinvest_threshold > 0 else 0
    savings_ratio = total_savings / savings_threshold if savings_threshold > 0 else 0

    # If reinvest ratio is lower, we should reinvest next (last_action = 0 or 2)
    # If savings ratio is lower, we should save next (last_action = 1)
    last_action = 1 if reinvest_ratio > savings_ratio else 0

    logger.info(
        f"Reinvest ratio: {reinvest_ratio:.2f}, Savings ratio: {savings_ratio:.2f}, Next action: {'Save' if last_action == 1 else 'Reinvest'}")

    # Process profit distribution
    while available_profit >= min(reinvest_threshold, savings_threshold):
        if (last_action == 0 or last_action ==
                2) and available_profit >= reinvest_threshold:
            # Reinvest in base order size
            reinvest_amount = reinvest_threshold
            available_profit -= reinvest_amount

            # Distribute the reinvestment evenly across all bots
            per_bot_reinvest = reinvest_amount / len(bots)

            for symbol, bot in bots.items():
                bot.profit_reinvested += per_bot_reinvest
                bot.base_order_increment += per_bot_reinvest
                bot.logger.info(
                    f"Reinvested {per_bot_reinvest:.2f} USDC to base order size from combined profits " +
                    f"(Total reinvested: {bot.profit_reinvested:.2f} USDC)")

            logger.info(
                f"Reinvested {reinvest_amount:.2f} USDC to base order size across all bots")

            # Send Telegram notification about reinvestment
            for symbol, bot in bots.items():
                if hasattr(bot, 'telegram') and bot.telegram is not None:
                    bot.telegram.send_notification(
                        f"🔄 Reinvested {reinvest_amount:.2f} USDC to base order size across all bots")
                    break  # Only need to send one notification

            last_action = 1

        elif last_action == 1 and available_profit >= savings_threshold:
            save_amount = savings_threshold
            available_profit -= save_amount

            # Distribute the savings evenly across all bots
            per_bot_save = save_amount / len(bots)

            for symbol, bot in bots.items():
                bot.wallet_savings += per_bot_save
                bot.logger.info(
                    f"Added {per_bot_save:.2f} USDC to wallet from combined profits " +
                    f"(Total saved: {bot.wallet_savings:.2f} USDC)")

            logger.info(
                f"Added {save_amount:.2f} USDC to wallet savings across all bots")

            # Send Telegram notification about savings
            for symbol, bot in bots.items():
                if hasattr(bot, 'telegram') and bot.telegram is not None:
                    bot.telegram.send_notification(
                        f"🏦 Added {save_amount:.2f} USDC to wallet savings across all bots")
                    break  # Only need to send one notification

            last_action = 0
        else:
            break

    return bots


def main():
    symbols = ["BTC/USDC", "XRP/USDC", "HBAR/USDC", "SOL/USDC"]
    quote_currency = 'USDC' # Define the quote currency

    # Create a single shared exchange instance
    try:
        exchange = ccxt.binance({
            'apiKey': config.TAPI_KEY,
            'secret': config.TAPI_SECRET,
        })
        exchange.set_sandbox_mode(True) # Set sandbox mode if needed
        logger.info("Shared Binance exchange instance created.")
    except Exception as e:
        logger.error(f"Failed to create shared Binance exchange instance: {e}")
        print(f"{Fore.RED}Fatal Error: Could not connect to Binance. Exiting.{Style.RESET_ALL}")
        return # Exit if connection fails

    shared_balance = SharedBalance(MAX_TOTAL_INVESTMENT, exchange, quote_currency)

    bots = {}
    threads = []

    print(f"\n{Fore.WHITE}{'=' * 60}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}INITIALIZING MULTI-PAIR TRADING BOT{Style.RESET_ALL}")
    print(f"{Fore.WHITE}{'=' * 60}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}Total Pairs: {len(symbols)}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}Total Initial Balance: {MAX_TOTAL_INVESTMENT} USDC{Style.RESET_ALL}")
    print(f"{Fore.CYAN}Per Coin Balance: {MAX_POSITION_SIZE} USDC{Style.RESET_ALL}")
    print(f"{Fore.WHITE}{'=' * 60}{Style.RESET_ALL}")

    for symbol in symbols:
        symbol_color = SYMBOL_COLORS.get(symbol, DEFAULT_SYMBOL_COLOR)
        print(f"\n{symbol_color}Starting bot for {symbol}...{Style.RESET_ALL}")
        try:
            bot = BinanceTradingBot(symbol, shared_balance, MAX_POSITION_SIZE)
            bots[symbol] = bot

            bot_thread = threading.Thread(
                target=bot.run_trading_loop,
                name=f"Bot-{symbol}",
                daemon=True
            )
            threads.append(bot_thread)
            bot_thread.start()

            print(
                f"{symbol_color}Bot for {symbol} started successfully!{Style.RESET_ALL}")
            print(f"{symbol_color}{'-' * 60}{Style.RESET_ALL}")

            time.sleep(2)

        except Exception as e:
            print(
                f"{Fore.RED}Error initializing bot for {symbol}: {str(e)}{Style.RESET_ALL}")

    last_combined_summary_time = datetime.now()
    last_profit_distribution_time = datetime.now()
    last_balance_update_time = datetime.now() # Track balance update time

    try:
        while True:
            current_time = datetime.now()

            # Check for profit distribution every 5 minutes
            # Update shared balance periodically (e.g., every 5 minutes)
            if (current_time - last_balance_update_time).total_seconds() >= 300:
                logger.info("Attempting to update shared balance from exchange...")
                shared_balance.update_balance_from_exchange()
                last_balance_update_time = current_time

            # Check for profit distribution every 5 minutes
            if (current_time - last_profit_distribution_time).total_seconds() >= 300:
                bots = manage_combined_profit_distribution(bots)
                last_profit_distribution_time = current_time

            if (current_time - last_combined_summary_time).total_seconds() >= 1800:
                summaries = {}
                total_realized = 0
                total_unrealized = 0
                total_reinvested = 0
                total_savings = 0
                total_fees = 0

                for symbol, bot in bots.items():
                    total_trades = bot.winning_trades + bot.losing_trades
                    win_rate = (
                        bot.winning_trades /
                        total_trades *
                        100) if total_trades > 0 else 0

                    summaries[symbol] = {
                        'realized_profit': bot.realized_profit,
                        'unrealized_profit': bot.calculate_open_pnl(),
                        'active_positions': len(bot.positions),
                        'win_rate': win_rate
                    }

                    # Log individual bot's realized profit for debugging
                    logger.info(f"Adding {symbol} realized profit: {bot.realized_profit:.2f} USDC to total: {total_realized:.2f} USDC")
                    
                    # Ensure we're adding the correct profit value
                    total_realized += bot.realized_profit
                    total_unrealized += bot.calculate_open_pnl()
                    total_reinvested += bot.profit_reinvested
                    total_savings += bot.wallet_savings
                    total_fees += bot.total_fees_paid

                bots[symbols[0]].telegram.notify_combined_summary(
                    summaries=summaries,
                    total_realized=total_realized,
                    total_unrealized=total_unrealized,
                    total_reinvested=total_reinvested,
                    total_savings=total_savings,
                    total_fees=total_fees,
                    exchange="Binance"
                )

                last_combined_summary_time = current_time

            for thread in threads:
                if not thread.is_alive():
                    pass

            time.sleep(60)

    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    ensure_directories()
    main()
