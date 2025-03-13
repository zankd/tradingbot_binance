import pickle
import logging
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from tenacity import retry, wait_exponential, stop_after_attempt
from typing import Dict, List, Any
import json
import os
import time
import pandas as pd

class TradingRecovery:
    def __init__(self, symbol):
        self.symbol = symbol
        self.symbol_filename = symbol.lower().replace('/', '_')
        
        # Update directory paths
        self.backup_dir = os.path.join('backups', 'states')
        self.trades_backup_dir = os.path.join('backups', 'trades')
        self.recovery_file = f"recovery_{self.symbol_filename}.json"
        
        # Add retry configuration
        self.max_retries = 3
        self.retry_delay = 5  # seconds
        
        # Ensure directories exist
        os.makedirs(self.backup_dir, exist_ok=True)
        os.makedirs(self.trades_backup_dir, exist_ok=True)
        os.makedirs(os.path.join('logs', 'backups'), exist_ok=True)
        
        # Initialize logger
        self.logger = logging.getLogger(f"recovery_{self.symbol_filename}")
        self.logger.setLevel(logging.INFO)
        
        # Add file handler if not already added
        if not self.logger.handlers:
            log_file = os.path.join('logs', 'backups', f"recovery_{self.symbol_filename}.log")
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            self.logger.addHandler(file_handler)

    def save_state(self, bot):
        """Save current trading state"""
        try:
            if not self.symbol or not self.recovery_file:
                raise ValueError("Symbol and recovery file must be set")
                
            state = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'symbol': self.symbol,
                'positions': bot.positions,
                'balance': bot.balance,
                'total_invested': bot.total_invested,
                'realized_profit': bot.realized_profit,
                'unrealized_profit': bot.unrealized_profit,
                'total_fees_paid': bot.total_fees_paid,
                'wallet_savings': bot.wallet_savings,
                'profit_reinvested': bot.profit_reinvested,
                'pending_profit': bot.pending_profit,
                'base_order_increment': bot.base_order_increment
            }
            
            # Save to main recovery file
            recovery_path = os.path.join(self.backup_dir, self.recovery_file)
            with open(recovery_path, 'w') as f:
                json.dump(state, f, indent=4)
            
            # Create hourly backup
            current_time = datetime.now()
            if current_time.minute == 0:
                backup_file = f"{self.symbol_filename}_{current_time.strftime('%Y%m%d_%H%M')}.json"
                backup_path = os.path.join(self.backup_dir, backup_file)
                with open(backup_path, 'w') as f:
                    json.dump(state, f, indent=4)
                
                self.cleanup_old_backups()
                
            self.logger.info(f"State saved successfully for {self.symbol}: {len(state['positions'])} positions")
            
        except Exception as e:
            self.logger.error(f"Error saving state for {self.symbol}: {str(e)}")

    def cleanup_old_backups(self):
        """Clean up old backup files using a tiered retention strategy"""
        try:
            current_time = datetime.now()
            
            # Group files by type
            hourly_backups = []
            daily_backups = []
            monthly_backups = []
            
            for file in os.listdir(self.backup_dir):
                if not file.startswith(self.symbol_filename) or not file.endswith('.json'):
                    continue
                    
                if file == self.recovery_file:
                    continue
                
                file_path = os.path.join(self.backup_dir, file)
                file_time = datetime.fromtimestamp(os.path.getctime(file_path))
                age_hours = (current_time - file_time).total_seconds() / 3600
                
                backup_info = {'path': file_path, 'time': file_time, 'name': file}
                
                # Group into appropriate retention bucket
                if age_hours <= 24:
                    hourly_backups.append(backup_info)
                elif age_hours <= 168:  # 7 days
                    daily_backups.append(backup_info)
                else:
                    monthly_backups.append(backup_info)
            
            # Cleanup trade backups
            self.cleanup_trade_backups()
            
            # Apply retention policies
            self._apply_retention_hourly(hourly_backups)
            self._apply_retention_daily(daily_backups)
            self._apply_retention_monthly(monthly_backups)
            
        except Exception as e:
            self.logger.error(f"Error in backup cleanup: {str(e)}")

    def _apply_retention_hourly(self, backups):
        """Keep last 24 hours of hourly backups"""
        try:
            # Sort by time, newest first
            backups.sort(key=lambda x: x['time'], reverse=True)
            
            # Keep one backup per hour for the last 24 hours
            kept_hours = set()
            for backup in backups:
                hour_key = backup['time'].strftime('%Y%m%d%H')
                if hour_key in kept_hours:
                    os.remove(backup['path'])
                    self.logger.info(f"Removed hourly backup: {backup['name']}")
                else:
                    kept_hours.add(hour_key)
        except Exception as e:
            self.logger.error(f"Error in hourly retention: {str(e)}")

    def _apply_retention_daily(self, backups):
        """Keep last 7 days of daily backups"""
        try:
            backups.sort(key=lambda x: x['time'], reverse=True)
            kept_days = set()
            
            for backup in backups:
                day_key = backup['time'].strftime('%Y%m%d')
                if day_key in kept_days or len(kept_days) >= 7:
                    os.remove(backup['path'])
                    self.logger.info(f"Removed daily backup: {backup['name']}")
                else:
                    kept_days.add(day_key)
        except Exception as e:
            self.logger.error(f"Error in daily retention: {str(e)}")

    def _apply_retention_monthly(self, backups):
        """Keep last 30 days of monthly backups"""
        try:
            backups.sort(key=lambda x: x['time'], reverse=True)
            kept_months = set()
            
            for backup in backups:
                month_key = backup['time'].strftime('%Y%m')
                if month_key in kept_months or len(kept_months) >= 30:
                    os.remove(backup['path'])
                    self.logger.info(f"Removed monthly backup: {backup['name']}")
                else:
                    kept_months.add(month_key)
        except Exception as e:
            self.logger.error(f"Error in monthly retention: {str(e)}")

    def cleanup_trade_backups(self):
        """Clean up trade CSV backups using similar retention policy"""
        try:
            current_time = datetime.now()
            for file in os.listdir(self.trades_backup_dir):
                if not file.startswith(f"backup_trades_{self.symbol_filename}"):
                    continue
                    
                file_path = os.path.join(self.trades_backup_dir, file)
                file_time = datetime.fromtimestamp(os.path.getctime(file_path))
                age_hours = (current_time - file_time).total_seconds() / 3600
                
                # Keep only last 24 hours of trade backups
                if age_hours > 24:
                    os.remove(file_path)
                    self.logger.info(f"Removed old trade backup: {file}")
                
        except Exception as e:
            self.logger.error(f"Error cleaning up trade backups: {str(e)}")

    def load_state(self) -> Dict:
        try:
            if Path(self.recovery_file).exists():
                with open(self.recovery_file, 'r') as f:
                    state = json.load(f)
                self.logger.info("Estado cargado exitosamente")
                return state
            return None
        except Exception as e:
            self.logger.error(f"Error al cargar el estado: {str(e)}")
            backup_file = f"{self.recovery_file}.backup"
            if Path(backup_file).exists():
                with open(backup_file, 'r') as f:
                    state = json.load(f)
                self.logger.info("Estado cargado desde backup")
                return state
            raise

    @retry(wait=wait_exponential(multiplier=1, min=4, max=60),
           stop=stop_after_attempt(5))
    async def recover_missing_data(self, last_known_time: datetime) -> List[Dict]:
        try:
            current_time = datetime.utcnow()
            missing_data = await self.exchange_api.get_history(
                timeframe='1m',
                start=last_known_time,
                end=current_time
            )
            self.logger.info(
                f"Recuperadas {len(missing_data)} velas desde {last_known_time} hasta {current_time}"
            )
            return missing_data
        except Exception as e:
            self.logger.error(f"Error recuperando datos faltantes: {str(e)}")
            raise

    @retry(wait=wait_exponential(multiplier=1, min=4, max=60),
           stop=stop_after_attempt(5))
    async def reconcile_orders(self, local_orders: List[Dict]) -> None:
        try:
            exchange_orders = await self.exchange_api.get_open_orders()
            local_ids = {o['id'] for o in local_orders}
            exchange_ids = {o['id'] for o in exchange_orders}

            # Órdenes que existen en el exchange pero no localmente (órdenes huérfanas)
            orphaned_orders = [o for o in exchange_orders if o['id'] not in local_ids]
            # Órdenes que existen localmente pero ya se han cerrado en el exchange
            closed_orders = [o for o in local_orders if o['id'] not in exchange_ids]

            for order in orphaned_orders:
                await self.handle_orphaned_order(order)

            for order in closed_orders:
                await self.handle_closed_order(order)

            self.logger.info(
                f"Reconcilación de órdenes completada. Huérfanas: {len(orphaned_orders)}, Cerradas: {len(closed_orders)}"
            )
        except Exception as e:
            self.logger.error(f"Error reconcilando órdenes: {str(e)}")
            raise

    async def handle_orphaned_order(self, order: Dict) -> None:
        try:
            current_price = await self.exchange_api.get_current_price(order['symbol'])
            if self.should_close_position(order, current_price):
                await self.exchange_api.cancel_order(order['id'])
                self.logger.info(f"Se canceló la orden huérfana {order['id']} por alcanzar el target de beneficio")
            else:
                self.add_to_local_orders(order)
                self.logger.info(f"Se añadió la orden huérfana {order['id']} al seguimiento local")
        except Exception as e:
            self.logger.error(f"Error manejando la orden huérfana {order['id']}: {str(e)}")
            raise

    async def handle_closed_order(self, order: Dict) -> None:
        try:
            order_info = await self.exchange_api.get_order(order['id'])
            if order_info.get('status') == 'FILLED':
                profit = self.calculate_profit(order_info)
                self.logger.info(f"La orden {order['id']} se completó con beneficio: {profit:.2f} USDT")
            self.remove_from_local_orders(order['id'])
        except Exception as e:
            self.logger.error(f"Error manejando la orden cerrada {order['id']}: {str(e)}")
            raise

    async def startup_recovery(self) -> None:
        try:
            state = self.load_state()
            if not state:
                self.logger.info("No se encontró un estado previo")
                return

            time_diff = datetime.utcnow() - state['timestamp']
            if time_diff.total_seconds() > 300:
                missing_data = await self.recover_missing_data(state['last_processed'])
                await self.process_missing_data(missing_data)

            self.local_orders = state['orders']
            await self.reconcile_orders(self.local_orders)
            self.logger.info("Recuperación de inicio completada exitosamente")
        except Exception as e:
            self.logger.error(f"Error durante la recuperación de inicio: {str(e)}")
            raise

    def should_close_position(self, order: Dict, current_price: float) -> bool:
        entry_price = float(order['price'])
        if order['side'] == 'BUY':
            profit_pct = (current_price - entry_price) / entry_price * 100
        else:
            profit_pct = (entry_price - current_price) / entry_price * 100
        return profit_pct >= self.profit_target

    def add_to_local_orders(self, order: Dict) -> None:
        if not any(o['id'] == order['id'] for o in self.local_orders):
            self.local_orders.append(order)
            self.logger.info(f"Orden {order['id']} agregada a las órdenes locales")
        else:
            self.logger.info(f"La orden {order['id']} ya existe en las órdenes locales")

    def remove_from_local_orders(self, order_id: str) -> None:
        before_count = len(self.local_orders)
        self.local_orders = [o for o in self.local_orders if o['id'] != order_id]
        after_count = len(self.local_orders)
        self.logger.info(f"Orden {order_id} removida de órdenes locales. Conteo: {before_count} -> {after_count}")

    def calculate_profit(self, order_info: Dict) -> float:
        try:
            entry_price = float(order_info['price'])
            filled_price = float(order_info.get('filled_price', entry_price))
            filled_qty = float(order_info.get('filled_qty', 0))
            if order_info['side'] == 'BUY':
                profit = (filled_price - entry_price) * filled_qty
            else:
                profit = (entry_price - filled_price) * filled_qty
            return profit
        except Exception as e:
            self.logger.error(f"Error calculando beneficio para la orden {order_info.get('id', 'desconocida')}: {str(e)}")
            return 0.0

    async def process_missing_data(self, missing_data: List[Dict]) -> None:
        """
        Procesa los datos de velas perdidas.
        Aquí integrarías la información recuperada en tu sistema de trading.
        """
        self.logger.info(f"Procesando {len(missing_data)} velas perdidas")
        # Implementa la lógica de integración según tus necesidades.
        await asyncio.sleep(0)  # Simula procesamiento asíncrono

    @retry(wait=wait_exponential(multiplier=1, min=4, max=60),
           stop=stop_after_attempt(5))
    async def ensure_connection(self) -> None:
        try:
            await self.exchange_api.ping()
            self.logger.info("Conexión al exchange verificada correctamente.")
        except Exception as e:
            self.logger.error(f"Fallo en la verificación de conexión: {str(e)}")
            raise

    def retry_operation(self, operation, *args, **kwargs):
        """Retry an operation with exponential backoff"""
        for attempt in range(self.max_retries):
            try:
                return operation(*args, **kwargs)
            except Exception as e:
                wait_time = self.retry_delay * (2 ** attempt)
                self.logger.warning(f"Operation failed (attempt {attempt + 1}/{self.max_retries}): {str(e)}")
                self.logger.info(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
        
        self.logger.error(f"Operation failed after {self.max_retries} attempts")
        return None

    def backup_trades_csv(self, csv_file):
        """Backup trades CSV file"""
        try:
            if os.path.exists(csv_file):
                df = pd.read_csv(csv_file)
                backup_file = f"backup_trades_{self.symbol_filename}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                backup_path = os.path.join(self.trades_backup_dir, backup_file)
                df.to_csv(backup_path, index=False)
                self.logger.info(f"CSV backup created: {backup_file}")
        except Exception as e:
            self.logger.error(f"Error backing up CSV: {str(e)}")

    def restore_from_backup(self, bot):
        """Restore bot state from backup"""
        try:
            state = self.load_state()
            if state:
                # Verify timestamp is recent enough (within 24 hours)
                state_time = datetime.strptime(state['timestamp'], '%Y-%m-%d %H:%M:%S')
                if (datetime.now() - state_time).total_seconds() > 86400:
                    self.logger.warning("Backup is too old (>24h), skipping restore")
                    return False

                # Restore bot state
                bot.positions = state['positions']
                bot.balance = state['balance']
                bot.total_invested = state['total_invested']
                bot.realized_profit = state['realized_profit']
                bot.unrealized_profit = state['unrealized_profit']
                bot.total_fees_paid = state['total_fees_paid']
                bot.wallet_savings = state['wallet_savings']
                bot.profit_reinvested = state['profit_reinvested']
                bot.pending_profit = state['pending_profit']
                bot.base_order_increment = state['base_order_increment']

                self.logger.info("Bot state restored successfully")
                return True
        except Exception as e:
            self.logger.error(f"Error restoring state: {str(e)}")
        return False
