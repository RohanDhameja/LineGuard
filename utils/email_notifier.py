"""
Email Notification System for Fire App
Sends alerts when high-risk vegetation zones are detected
"""

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
import os


class EmailNotifier:
    """Handle email notifications for vegetation fire risk alerts"""
    
    def __init__(self, smtp_server=None, smtp_port=None, sender_email=None, 
                 sender_password=None, use_tls=True):
        """
        Initialize email notifier
        
        Args:
            smtp_server: SMTP server address (e.g., 'smtp.gmail.com')
            smtp_port: SMTP port (587 for TLS, 465 for SSL)
            sender_email: Sender email address
            sender_password: Sender email password or app-specific password
            use_tls: Whether to use TLS (default: True)
        """
        # Use environment variables if not provided
        self.smtp_server = smtp_server or os.getenv('SMTP_SERVER', 'smtp.gmail.com')
        self.smtp_port = smtp_port or int(os.getenv('SMTP_PORT', '587'))
        self.sender_email = sender_email or os.getenv('SENDER_EMAIL')
        self.sender_password = sender_password or os.getenv('SENDER_PASSWORD')
        self.use_tls = use_tls
        
        # Validate configuration
        self.is_configured = all([
            self.smtp_server,
            self.smtp_port,
            self.sender_email,
            self.sender_password
        ])
    
    def send_alert_email(self, recipient_email, alert_zones, date_str=None):
        """
        Send email alert for high-risk zones
        
        Args:
            recipient_email: Email address to send alert to
            alert_zones: List of zone dictionaries with alert information
            date_str: Date string for the alert (optional)
            
        Returns:
            dict: Status of email sending
        """
        if not self.is_configured:
            return {
                'status': 'error',
                'message': 'Email configuration not set. Please configure SMTP settings.'
            }
        
        try:
            # Prepare email content
            subject = f"🔥 URGENT: {len(alert_zones)} High-Risk Vegetation Zone(s) Detected"
            
            # Create HTML email body
            html_body = self._create_alert_email_html(alert_zones, date_str)
            
            # Create plain text version
            text_body = self._create_alert_email_text(alert_zones, date_str)
            
            # Send email
            result = self._send_email(
                recipient_email,
                subject,
                text_body,
                html_body
            )
            
            return result
            
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Failed to send email: {str(e)}'
            }
    
    def send_weekly_summary(self, recipient_email, summary_data):
        """
        Send weekly summary report
        
        Args:
            recipient_email: Email address to send summary to
            summary_data: Dictionary with summary statistics
            
        Returns:
            dict: Status of email sending
        """
        if not self.is_configured:
            return {
                'status': 'error',
                'message': 'Email configuration not set'
            }
        
        try:
            subject = f"📊 Weekly Vegetation Monitoring Summary - {summary_data.get('week_ending', 'N/A')}"
            
            html_body = self._create_summary_email_html(summary_data)
            text_body = self._create_summary_email_text(summary_data)
            
            result = self._send_email(
                recipient_email,
                subject,
                text_body,
                html_body
            )
            
            return result
            
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Failed to send summary: {str(e)}'
            }
    
    def _send_email(self, recipient, subject, text_body, html_body=None):
        """
        Internal method to send email via SMTP
        
        Args:
            recipient: Recipient email address
            subject: Email subject
            text_body: Plain text email body
            html_body: HTML email body (optional)
            
        Returns:
            dict: Status of email sending
        """
        try:
            # Create message
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = self.sender_email
            msg['To'] = recipient
            
            # Attach text and HTML parts
            text_part = MIMEText(text_body, 'plain')
            msg.attach(text_part)
            
            if html_body:
                html_part = MIMEText(html_body, 'html')
                msg.attach(html_part)
            
            # Connect to SMTP server and send
            if self.use_tls:
                server = smtplib.SMTP(self.smtp_server, self.smtp_port)
                server.starttls()
            else:
                server = smtplib.SMTP_SSL(self.smtp_server, self.smtp_port)
            
            server.login(self.sender_email, self.sender_password)
            server.send_message(msg)
            server.quit()
            
            return {
                'status': 'success',
                'message': f'Email sent successfully to {recipient}',
                'recipient': recipient,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': f'SMTP error: {str(e)}',
                'recipient': recipient
            }
    
    def _create_alert_email_html(self, alert_zones, date_str):
        """Create HTML email body for alerts"""
        date_display = date_str or datetime.now().strftime('%Y-%m-%d')
        
        html = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                .header {{ background-color: #d32f2f; color: white; padding: 20px; text-align: center; }}
                .content {{ padding: 20px; }}
                .alert-box {{ background-color: #fff3cd; border-left: 4px solid #ff9800; padding: 15px; margin: 10px 0; }}
                .zone-item {{ background-color: #f8f9fa; padding: 12px; margin: 8px 0; border-radius: 4px; }}
                .critical {{ border-left: 4px solid #d32f2f; }}
                .high {{ border-left: 4px solid #ff9800; }}
                .footer {{ background-color: #f1f1f1; padding: 15px; text-align: center; font-size: 12px; }}
                .metric {{ font-weight: bold; color: #d32f2f; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🔥 FireGuard AI Alert</h1>
                <p>High-Risk Vegetation Detected Near Power Lines</p>
            </div>
            
            <div class="content">
                <div class="alert-box">
                    <h2>⚠️ URGENT ACTION REQUIRED</h2>
                    <p><strong>{len(alert_zones)}</strong> vegetation zone(s) require immediate attention on <strong>{date_display}</strong></p>
                </div>
                
                <h3>Alert Details:</h3>
        """
        
        for i, zone in enumerate(alert_zones, 1):
            risk_class = 'critical' if zone.get('clearance_m', 10) < 3.0 else 'high'
            html += f"""
                <div class="zone-item {risk_class}">
                    <h4>Zone #{i}</h4>
                    <p><strong>Location:</strong> {zone.get('lat', 'N/A'):.6f}, {zone.get('lon', 'N/A'):.6f}</p>
                    <p><strong>Vegetation Height:</strong> <span class="metric">{zone.get('veg_height_m', 0):.2f}m</span></p>
                    <p><strong>Clearance:</strong> <span class="metric">{zone.get('clearance_m', 0):.2f}m</span></p>
                    <p><strong>Risk Level:</strong> <span class="metric">{'🔴 CRITICAL' if risk_class == 'critical' else '🟠 HIGH'}</span></p>
                </div>
            """
        
        html += """
                <h3>Recommended Actions:</h3>
                <ul>
                    <li>🚁 Dispatch field crew for immediate inspection</li>
                    <li>✂️ Schedule emergency vegetation trimming</li>
                    <li>📞 Contact local fire department if conditions are dry</li>
                    <li>⚡ Consider temporary line de-energization if critical</li>
                </ul>
                
                <p><strong>This is an automated alert from FireGuard AI Vegetation Monitoring System.</strong></p>
            </div>
            
            <div class="footer">
                <p>FireGuard AI - Power Line Vegetation Fire Risk Monitoring</p>
                <p>For support, contact your system administrator</p>
            </div>
        </body>
        </html>
        """
        
        return html
    
    def _create_alert_email_text(self, alert_zones, date_str):
        """Create plain text email body for alerts"""
        date_display = date_str or datetime.now().strftime('%Y-%m-%d')
        
        text = f"""
🔥 FIREGUARD AI ALERT - HIGH-RISK VEGETATION DETECTED
===============================================

⚠️ URGENT ACTION REQUIRED

{len(alert_zones)} vegetation zone(s) require immediate attention on {date_display}

ALERT DETAILS:
--------------
"""
        
        for i, zone in enumerate(alert_zones, 1):
            risk_level = '🔴 CRITICAL' if zone.get('clearance_m', 10) < 3.0 else '🟠 HIGH'
            text += f"""
Zone #{i}:
  Location: {zone.get('lat', 'N/A'):.6f}, {zone.get('lon', 'N/A'):.6f}
  Vegetation Height: {zone.get('veg_height_m', 0):.2f}m
  Clearance: {zone.get('clearance_m', 0):.2f}m
  Risk Level: {risk_level}

"""
        
        text += """
RECOMMENDED ACTIONS:
--------------------
🚁 Dispatch field crew for immediate inspection
✂️ Schedule emergency vegetation trimming
📞 Contact local fire department if conditions are dry
⚡ Consider temporary line de-energization if critical

---
This is an automated alert from FireGuard AI Vegetation Monitoring System.
For support, contact your system administrator.
"""
        
        return text
    
    def _create_summary_email_html(self, summary_data):
        """Create HTML email body for weekly summary"""
        html = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                .header {{ background-color: #1976d2; color: white; padding: 20px; text-align: center; }}
                .content {{ padding: 20px; }}
                .stats {{ display: flex; justify-content: space-around; margin: 20px 0; }}
                .stat-box {{ background-color: #f8f9fa; padding: 15px; border-radius: 8px; text-align: center; min-width: 150px; }}
                .stat-number {{ font-size: 32px; font-weight: bold; color: #1976d2; }}
                .footer {{ background-color: #f1f1f1; padding: 15px; text-align: center; font-size: 12px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>📊 Weekly Vegetation Monitoring Summary</h1>
                <p>Week Ending: {summary_data.get('week_ending', 'N/A')}</p>
            </div>
            
            <div class="content">
                <div class="stats">
                    <div class="stat-box">
                        <div class="stat-number">{summary_data.get('total_alerts', 0)}</div>
                        <div>Total Alerts</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-number">{summary_data.get('monitored_zones', 0)}</div>
                        <div>Monitored Zones</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-number">{summary_data.get('avg_vegetation', 0):.2f}m</div>
                        <div>Avg Vegetation</div>
                    </div>
                </div>
                
                <h3>Key Insights:</h3>
                <ul>
                    <li>Vegetation growth trend: {summary_data.get('growth_trend', 'Stable')}</li>
                    <li>Highest risk zone: {summary_data.get('highest_risk_zone', 'N/A')}</li>
                    <li>ML confidence: {summary_data.get('ml_confidence', 0):.1f}%</li>
                </ul>
            </div>
            
            <div class="footer">
                <p>FireGuard AI - Automated Weekly Report</p>
            </div>
        </body>
        </html>
        """
        return html
    
    def _create_summary_email_text(self, summary_data):
        """Create plain text email body for weekly summary"""
        text = f"""
📊 WEEKLY VEGETATION MONITORING SUMMARY
=======================================

Week Ending: {summary_data.get('week_ending', 'N/A')}

STATISTICS:
-----------
Total Alerts: {summary_data.get('total_alerts', 0)}
Monitored Zones: {summary_data.get('monitored_zones', 0)}
Average Vegetation: {summary_data.get('avg_vegetation', 0):.2f}m

KEY INSIGHTS:
-------------
• Vegetation growth trend: {summary_data.get('growth_trend', 'Stable')}
• Highest risk zone: {summary_data.get('highest_risk_zone', 'N/A')}
• ML confidence: {summary_data.get('ml_confidence', 0):.1f}%

---
FireGuard AI - Automated Weekly Report
"""
        return text

