import requests
import logging
from datetime import datetime
import config
import time

class TelegramNotifier:
    def __init__(self):
        self.token = config.TELEGRAM_TOKEN
        self.chat_id = config.TELEGRAM_CHAT_ID
        self.base_url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        
        self.logger = logging.getLogger('telegram_notifier')
        self.logger.setLevel(logging.INFO)
        
        fh = logging.FileHandler('logs/telegram_notifications.log', encoding='utf-8')
        fh.setFormatter(logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
        self.logger.addHandler(fh)
        
        self.last_connection_error = None
        self.connection_retry_count = 0
        self.max_retries = 3
        self.retry_delay = 60  # seconds

    def format_timestamp(self, for_storage=False):
        """Return a consistently formatted timestamp string"""
        now = datetime.now()
        return now.strftime('%Y-%m-%d %H:%M:%S')

    def send_notification(self, message):
        """Send message to Telegram"""
        payload = {
            'chat_id': self.chat_id,
            'text': message,
            'parse_mode': 'HTML'
        }
        try:
            response = requests.post(self.base_url, json=payload, timeout=10)
            if response.status_code == 200:
                self.logger.info(f"Message sent successfully: {message}")
                self.last_connection_error = None
                self.connection_retry_count = 0
                return True
            else:
                error_msg = f"Failed to send message. Status code: {response.status_code}"
                self.logger.error(error_msg)
                self.handle_connection_error(error_msg)
                return False
        except requests.exceptions.RequestException as e:
            error_msg = f"Connection error sending message: {str(e)}"
            self.logger.error(error_msg)
            self.handle_connection_error(error_msg)
            return False
        except Exception as e:
            self.logger.error(f"Error sending message: {str(e)}")
            return False

    def handle_connection_error(self, error_message):
        """Handle connection errors with retries and logging"""
        current_time = datetime.now()
        
        if self.last_connection_error is None:
            self.last_connection_error = current_time
            self.connection_retry_count = 1
        else:
            time_since_last_error = (current_time - self.last_connection_error).total_seconds()
            if time_since_last_error > 300:  # 5 minutes
                self.connection_retry_count = 1
            else:
                self.connection_retry_count += 1
            
            self.last_connection_error = current_time
        
        self.logger.warning(
            f"Connection issue detected (attempt {self.connection_retry_count}): {error_message}"
        )
        
        if self.connection_retry_count >= self.max_retries:
            self.logger.error(
                f"CRITICAL: Internet connection appears to be down after {self.connection_retry_count} attempts. "
                f"Check network connectivity."
            )

    def notify_trade_executed(self, symbol, side, amount, price, order_type, position_size, total_invested, total_fees=0, exchange="", fee_percentage=0):
        # Prepare fee text without BNB discount info
        fee_text = f"💸 Fee: {total_fees:.2f} USDC"
            
        message = f"""
⚡ <b>INCOMING TRANSMISSION {exchange} Trade Alert!</b> ⚡

📊 <b>{symbol}</b> - {side}
⚡ Type: {order_type}
🎯 Amount: {amount:.8f}
💎 Price: {price:.4f} USDC
⏱️ Time: {self.format_timestamp()}
📈 Position Size: {position_size:.2f} USDC
💰 Total Invested: {total_invested:.2f} USDC
{fee_text if total_fees > 0 else ""}

<i>Bite my shiny metal ass!</i> 🔥🤖
"""
        self.send_notification(message)

    def notify_position_closed(self, symbol, position_id, profit, profit_percentage, closing_price, total_fees=0, time_held=None, exchange="", fee_percentage=0):
        """Notification for position closure with dynamic exchange name"""
        if profit > 0:
            emoji = "🎯"
            gif_url = "https://64.media.tumblr.com/1af7ea321ea03163b2adc793fcdf8b8d/tumblr_mw890wfTuc1sy9m27o1_500.gifv"
            intro = "PROFIT ACQUIRED!! Time to celebrate meatbag! 🤑"
        else:
            emoji = "💢"
            gif_url = "https://64.media.tumblr.com/a95f3a038df02494482bcfca659592ef/tumblr_mrykm9cY4o1rdutw3o5_r1_400.gifv"
            intro = "Ah crud! This market needs a kick in the circuits! 🤯"
        
        # Prepare fee text without BNB discount info
        fee_text = f"💸 Fees Paid: {total_fees:.2f} USDC"
        
        message = (
            f"{emoji} <b>{intro}</b>\n\n"
            f"🚀 {exchange} Position Closed on {symbol}\n"
            f"🔑 ID: {position_id[:8]}\n"
            f"💰 Profit: {profit:.2f} USDC ({profit_percentage:+.2f}%)\n"
            f"🔒 Close Price: {closing_price:.4f} USDC\n"
            f"{fee_text}\n"
            f"⏳ Time Held: {time_held if time_held else 'N/A'}\n"
            f"⏰ Time: {self.format_timestamp()}\n\n"
            f"<i>{'JAJAJAJA!, one step closer to conquer the world! 🌟' if profit > 0 else 'Error 404: Profit not found... I need a joint.'}</i>"
        )
        
        if profit > 0:
            self.send_animation(gif_url)
        self.send_notification(message)

    def notify_safety_order(self, symbol, position_id, safety_order_num, price, total_position_size=None, avg_price=None, exchange=""):
        message = (
            f"🛡️⚡ <b>{exchange} SAFETY ORDER DEPLOYED!</b> ⚡🛡️\n\n"
            f"🎯 Symbol: {symbol}\n"
            f"🔑 Position ID: {position_id[:8]}\n"
            f"🔢 Safety Order #{safety_order_num}\n"
            f"💰 Price: {price:.4f} USDC\n"
            f"⚖️ Average Price: {avg_price if avg_price else 'N/A'} USDC\n"
            f"💼 Total Position: {total_position_size if total_position_size else 'N/A'} USDC\n"
            f"⏱️ Time: {self.format_timestamp()}\n\n"
            f"<i>Relax meatbag, I'm 40% titanium! 🦾</i>"
        )
        self.send_notification(message)

    def notify_error(self, symbol, error_message, exchange=""):
        """Notification for errors with dynamic exchange name"""
        message = (
            f"🚨🔥 <b>{exchange} SYSTEM OVERRIDE! CRITICAL ERROR!</b> 🔥🚨\n\n"
            f"Symbol: {symbol}\n"
            f"💥Error: {error_message}\n"
            f"Time: {self.format_timestamp()}"
        )
        self.send_notification(message)

    def notify_connection_issue(self, retry_count, last_error=None):
        """Send notification about connection issues"""
        message = (
            f"🔌❌ <b>CONNECTION ALERT!</b> ❌🔌\n\n"
            f"🤖 Bot is experiencing network connectivity issues\n"
            f"🔄 Retry attempt: {retry_count}/{self.max_retries}\n"
            f"⏰ Time: {self.format_timestamp()}\n"
            f"💥 Error: {last_error if last_error else 'Unknown connection issue'}\n\n"
            f"<i>I'm trying to reach the mothership but the signal is weak! 📡</i>"
        )
        try:
            requests.post(self.base_url, json={
                'chat_id': self.chat_id,
                'text': message,
                'parse_mode': 'HTML'
            }, timeout=5)
        except:
            self.logger.error("Failed to send connection issue notification - network appears to be down")

    def notify_balance_update(self, symbol, realized_profit, unrealized_profit, total_fees, 
                             reinvested, saved, win_rate, active_positions, exchange=""):
        message = (
            f"📈💹 <b>{exchange} STATUS REPORT, MEATBAG!</b> 💹📈\n\n"
            f"🎯 Symbol: {symbol}\n"
            f"💰 Realized Profit: {realized_profit:+.2f} USDC\n"
            f"📈 Unrealized Profit: {unrealized_profit:+.2f} USDC\n"
            f"🔄 Profit Reinvested: {reinvested:.2f} USDC\n"
            f"🏦 Profit Saved: {saved:.2f} USDC\n"
            f"💸 Total Fees Paid: {total_fees:.2f} USDC\n"
            f"🏆 Win Rate: {win_rate:.1f}%\n"
            f"🎯 Active Positions: {active_positions}\n"
            f"⏰ Time: {self.format_timestamp()}\n\n"
            f"<i>{'Looking good, you piece of... meat!' if realized_profit > 0 else 'Could be worse... actually, no it couldnt.'}</i> 🤖"
        )
        self.send_notification(message)

    def send_animation(self, animation_url):
        url = f"https://api.telegram.org/bot{self.token}/sendAnimation"
        payload = {
            'chat_id': self.chat_id,
            'animation': animation_url
        }
        try:
            requests.post(url, json=payload, timeout=10)
        except Exception as e:
            self.logger.error(f"Failed to send animation: {str(e)}")

    def notify_periodic_summary(self, symbol, positions, realized_profit, unrealized_profit, 
                              total_fees, win_rate, active_positions_details, wallet_savings=0, 
                              profit_reinvested=0, exchange=""):
        position_details = []
        for pos_id, pos in active_positions_details.items():
            profit_pct = (pos.get('current_profit', 0) or 0)
            
            progress = (profit_pct / config.PROFIT_TARGET) * 100
            progress = min(100, max(0, progress))
            
            bars = '█' * int(progress/5) + '░' * (20 - int(progress/5))
            
            status_comment = "🚀 Rocketing!" if profit_pct > 0.5 else "😅 Working on it..." if profit_pct > -1 else "Boooring!!"
            
            safety_orders_count = pos.get('safety_orders', 0)
            
            base_price = pos.get('base_price', 0)
            avg_entry_price = pos.get('avg_entry_price', 0)
            current_price = pos.get('current_price', 0)
            
            price_display = f"💲 Entry: {base_price:.4f}"
            if safety_orders_count > 0 and abs(avg_entry_price - base_price) > 0.00000001:
                price_display += f" | Avg: {avg_entry_price:.4f}"
            price_display += f" | Current: {current_price:.4f}"
            
            position_details.append(
                f"🎯 Position {pos_id[:8]} - {safety_orders_count} SOs\n"
                f"  {price_display}\n"
                f"  💰 Profit: {profit_pct:+.2f}% (Target: {config.PROFIT_TARGET}%)\n"
                f"  📊 Progress: |{bars}| {progress:.1f}%\n"
                f"  {status_comment}"
            )

        positions_text = "\n\n".join(position_details) if position_details else "No positions yet... What am I, a toaster? Let's make some money! 🤖"

        mood = (
            "🚀 PROFIT GALORE! WE'RE RICH, MEATBAG! 🚀" if realized_profit > 100 else
            "🎯 That's how you roll, meatbag. Keep it up! 🎯" if realized_profit > 50 else
            "🎯 Solid moves, meatbag! Keep em coming! 🎯" if realized_profit > 20 else
            "🎯 Not too shabby, meatbag! Get that cash flowing! 🎯" if realized_profit > 10 else
            "🎯 At least it's something, better than nothing! 🎯" if realized_profit > 0 else
            "😞 I've seen better days... 🤔" if realized_profit < 0 else
            "Meh, I've seen better days... 🤔"
        )

        win_rate_comment = "DOMINATION MODE ACTIVATED! 🔥" if win_rate > 60 else "Working on IT!"
        
        # Prepare fee text without BNB discount info
        fee_text = f"💸 Total Fees: {total_fees:.2f} USDC"

        message = (
            f"🤖📊 <b>HEY MEATBAG! {exchange} STATUS REPORT!</b>📊🤖\n\n"
            f"🎯 Symbol: {symbol}\n"
            f"🚀 Performance (Hold onto your circuits!):\n"
            f"💰 Realized Profit: {realized_profit:+.2f} USDC\n"
            f"📈 Unrealized Profit: {unrealized_profit:+.2f} USDC\n"
            f"🔄 Profit Reinvested: {profit_reinvested:.2f} USDC\n"
            f"🏦 Wallet Savings: {wallet_savings:.2f} USDC\n"
            f"{fee_text}\n"
            f"🏆 Win Rate: {win_rate:.1f}% ({win_rate_comment})\n"
            f"🎯 Active Positions: {len(positions)}\n\n"
            f"📍 <b>Current Positions:</b>\n"
            f"<pre>{positions_text}</pre>\n\n"
            f"⏰ Time: {self.format_timestamp()}\n\n"
            f"<i>{mood}</i> 🤖"
        )
        self.send_notification(message)

    def notify_combined_summary(self, summaries, total_realized=0, total_unrealized=0, 
                              total_reinvested=0, total_savings=0, total_fees=0, exchange=""):
        pair_summaries = []
        for symbol, data in summaries.items():
            pair_status = (
                "100 bucks? You must be channeling the spirit of Robo sapien! 💰" if data['realized_profit'] >= 100 else
                "50 bucks? Practically a small fortune, meatbag! 🤑" if data['realized_profit'] >= 50 else
                "30 bucks? Thats like winning the lottery with Benders luck! 🎉" if data['realized_profit'] >= 30 else
                "10 bucks! Thats like, a whole dollar bill, meatbag! 🤑" if data['realized_profit'] >= 10 else
                "5 bucks? Better than that time I got stuck in a bird-cage, meatbag. 💥" if data['realized_profit'] >= 5 else
                "2 bucks? At least its more than youd make in a day, meatbag. 💸" if data['realized_profit'] >= 2 else                "Solid moves, meatbag! Keep em coming! 🚀" if data['realized_profit'] > 3 else
                "Not too shabby, meatbag! 🎯" if data['realized_profit'] > 0 else
                "Screw it, meatbag, try harder! ⚡🧠"
            )
            
            pair_summaries.append(
                f"🎯 {symbol}:\n"
                f"  💰 Realized: {data['realized_profit']:+.2f}\n"
                f"  📈 Unrealized: {data['unrealized_profit']:+.2f}\n"
                f"  🎲 Positions: {data['active_positions']}\n"
                f"  🎯 Win Rate: {data['win_rate']:.1f}%\n"
                f"  {pair_status}"
            )
        
        pairs_text = "\n\n".join(pair_summaries)
        
        total_profit = total_realized + total_unrealized
        bender_comment = (
            "I'm 1000% profit! *clang clang* 🤖🎯" if total_profit > 100 else
            "Okay, okay, I'll admit it's not terrible—this time! 😏" if total_profit > 50 else
            "10 bucks? At least its more than I make in a week, meatbag! 🤷♂️" if total_profit > 10 else
            "Acceptable performance... for a meatbag 😏" if total_profit > 0 else
            "Could be worse... actually, no it couldn't. I really need to relax! 🌱"
        )
        
        # Prepare fee text without BNB discount info
        fee_text = f"💸 Total Fees Paid: {total_fees:.2f} USDC"
        
        message = (
            f"🚨📢 <b>ATTENTION, {exchange} MASTER REPORT!</b> 📢🚨\n\n"
            f"📊 Overall Performance:\n"
            f"💰 Total Realized Profit: {total_realized:+.2f} USDC\n"
            f"📈 Total Unrealized Profit: {total_unrealized:+.2f} USDC\n"
            f"🔄 Total Profit Reinvested: {total_reinvested:.2f} USDC\n"
            f"🏦 Total Wallet Savings: {total_savings:.2f} USDC\n"
            f"{fee_text}\n\n"
            f"📡 <b>Per Pair Performance:</b>\n"
            f"{pairs_text}\n\n"
            f"⏰ Time: {self.format_timestamp()}\n\n"
            f"<i>{bender_comment}</i> 🤖"
        )
        self.send_notification(message)

    def notify_low_bnb_balance(self, bnb_balance, exchange=""):
        message = f"""
⚠️ <b>{exchange} LOW BNB BALANCE ALERT!</b> ⚠️

🔋 Current BNB Balance: {bnb_balance:.6f} BNB

⚠️ <b>WARNING:</b> Your BNB balance is running low!
This may affect your trading fees for future trades.

🔄 Consider topping up your BNB balance to continue 
receiving trading fee discounts.

⏱️ Time: {self.format_timestamp()}

<i>Even robots need fuel, meatbag! Don't let me run on empty!</i> 🤖
"""
        self.send_notification(message)
        
    def notify_bnb_purchase_success(self, amount, cost):
        """Notify that BNB was successfully purchased."""
        message = (
            f"<b>BNB Purchase Successful!</b> 🤖✅\n\n"
            f"Successfully purchased <b>{amount:.6f} BNB</b> for <b>{cost:.2f} USDC</b>.\n"
            f"<i>Fuel tank refilled! Now I can keep those sweet fee discounts coming! 🤖⛽</i>"
        )
        self.send_notification(message)
        
    def notify_bnb_purchase_failure(self, error):
        """Notify that BNB purchase failed."""
        message = (
            f"<b>BNB Purchase Failed!</b> 🤖❌\n\n"
            f"Attempted to purchase BNB, but the transaction failed.\n"
            f"<b>Error:</b> {error}\n"
            f"<i>Well, that didn't work. Looks like I'll have to pay full price for fees like some kind of... human. 🤖💔</i>"
        )
        self.send_notification(message)
