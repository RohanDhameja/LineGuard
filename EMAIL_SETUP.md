# 📧 Email Notification Setup Guide

The Fire App includes automated email notifications for high-risk vegetation alerts.

## 🔧 Configuration Options

### Option 1: Gmail (Easiest for Testing)

1. **Enable 2-Factor Authentication** on your Gmail account
2. **Create an App Password**:
   - Go to: https://myaccount.google.com/apppasswords
   - Select "Mail" and "Other (Custom name)"
   - Name it "FireGuard AI"
   - Copy the 16-character password

3. **Set Environment Variables**:

```bash
# On PythonAnywhere (or Linux/Mac):
export SMTP_SERVER="smtp.gmail.com"
export SMTP_PORT="587"
export SENDER_EMAIL="your-email@gmail.com"
export SENDER_PASSWORD="your-16-char-app-password"
```

```bash
# On Windows:
set SMTP_SERVER=smtp.gmail.com
set SMTP_PORT=587
set SENDER_EMAIL=your-email@gmail.com
set SENDER_PASSWORD=your-16-char-app-password
```

### Option 2: SendGrid (Better for Production)

SendGrid offers 100 free emails/day, perfect for monitoring systems.

1. **Sign up**: https://signup.sendgrid.com/
2. **Create API Key**: Settings → API Keys → Create API Key
3. **Set Environment Variables**:

```bash
export SMTP_SERVER="smtp.sendgrid.net"
export SMTP_PORT="587"
export SENDER_EMAIL="your-verified-email@yourdomain.com"
export SENDER_PASSWORD="your-sendgrid-api-key"
```

### Option 3: Other SMTP Services

- **Mailgun**: 5,000 free emails/month
- **AWS SES**: Very cheap, pay-as-you-go
- **Outlook/Office 365**: smtp-mail.outlook.com:587

## 🚀 PythonAnywhere Setup

### Step 1: Set Environment Variables

On PythonAnywhere, add to your WSGI file (`/var/www/your_pythonanywhere_com_wsgi.py`):

```python
import os

# Email configuration
os.environ['SMTP_SERVER'] = 'smtp.gmail.com'
os.environ['SMTP_PORT'] = '587'
os.environ['SENDER_EMAIL'] = 'your-email@gmail.com'
os.environ['SENDER_PASSWORD'] = 'your-app-password'

# ... rest of WSGI file
```

**⚠️ Security Warning**: Don't commit passwords to GitHub! On PythonAnywhere paid tiers, use environment variables in the Web tab instead.

### Step 2: Reload Your Web App

Click the green **"Reload"** button on the Web tab.

### Step 3: Test Configuration

Visit: `https://your-site.pythonanywhere.com/api/email_status`

You should see:
```json
{
  "configured": true,
  "message": "Email system ready",
  "sender_email": "your-email@gmail.com",
  "smtp_server": "smtp.gmail.com"
}
```

## 📨 How to Use Email Notifications

### API Endpoint 1: Send Alert Email

**POST** `/api/send_alert_email`

```json
{
  "recipient_email": "manager@company.com",
  "alert_zones": [
    {
      "lat": 35.7674,
      "lon": -120.4420,
      "veg_height_m": 5.2,
      "clearance_m": 2.8
    }
  ],
  "date": "2025-10-28"
}
```

### API Endpoint 2: Send Weekly Summary

**POST** `/api/send_weekly_summary`

```json
{
  "recipient_email": "manager@company.com",
  "summary_data": {
    "week_ending": "2025-10-28",
    "total_alerts": 7,
    "monitored_zones": 10,
    "avg_vegetation": 2.45,
    "growth_trend": "Increasing",
    "highest_risk_zone": "Zone 6",
    "ml_confidence": 94.5
  }
}
```

### API Endpoint 3: Check Email Status

**GET** `/api/email_status`

Returns configuration status.

## 🧪 Test Email Locally

```bash
cd ~/powerline_monitor
source venv/bin/activate

# Set environment variables
export SMTP_SERVER="smtp.gmail.com"
export SMTP_PORT="587"
export SENDER_EMAIL="your-email@gmail.com"
export SENDER_PASSWORD="your-app-password"

# Run the app
python3 app.py
```

Then test with curl:

```bash
curl -X POST http://localhost:5000/api/send_alert_email \
  -H "Content-Type: application/json" \
  -d '{
    "recipient_email": "your-test-email@gmail.com",
    "alert_zones": [
      {
        "lat": 35.7674,
        "lon": -120.4420,
        "veg_height_m": 5.2,
        "clearance_m": 2.8
      }
    ],
    "date": "2025-10-28"
  }'
```

## 🔐 Security Best Practices

1. **Never commit passwords** to Git
2. **Use environment variables** for all secrets
3. **Rotate passwords** regularly
4. **Use app-specific passwords** for Gmail (not your main password)
5. **Whitelist IPs** if your SMTP provider supports it

## 🐛 Troubleshooting

### "Email notification system not configured"
- Check environment variables are set correctly
- Verify WSGI file loads environment variables
- Check `/api/email_status` endpoint

### "SMTP authentication failed"
- **Gmail**: Make sure you're using an App Password, not your regular password
- **SendGrid**: Verify API key is correct and has "Mail Send" permissions
- Check username is correct (usually the email address)

### "Connection refused" or "Timeout"
- **PythonAnywhere Free**: Cannot use external SMTP on free tier
  - Solution: Upgrade to paid tier ($5/month) OR use SendGrid API instead of SMTP
- Check SMTP server and port are correct
- Verify firewall isn't blocking port 587

### "TLS/SSL errors"
- Port 587 = TLS (starttls)
- Port 465 = SSL
- Make sure you're using the right port for your provider

## 📊 Email Notification Features

### Alert Emails Include:
- ✅ Number of high-risk zones
- ✅ Location coordinates for each zone
- ✅ Vegetation height and clearance
- ✅ Risk level (🔴 Critical or 🟠 High)
- ✅ Recommended actions
- ✅ Professional HTML formatting

### Weekly Summary Emails Include:
- ✅ Total alerts for the week
- ✅ Number of monitored zones
- ✅ Average vegetation height
- ✅ Growth trends
- ✅ ML model confidence
- ✅ Highest risk zone identification

## 💡 Pro Tips

1. **Automated Weekly Reports**: Set up a cron job on PythonAnywhere to send weekly summaries automatically
2. **Multiple Recipients**: Create a distribution list and send to that email
3. **SMS Alerts**: Use services like Twilio to convert critical emails to SMS
4. **Slack Integration**: Forward critical alerts to Slack channels

## 📧 Example Email Output

When an alert is sent, recipients receive a professional HTML email with:

**Subject**: 🔥 URGENT: 3 High-Risk Vegetation Zone(s) Detected

**Body**:
- Clear alert header with urgency indicator
- Detailed zone information with risk levels
- Recommended immediate actions
- Professional formatting with color coding

Perfect for utility managers, fire departments, and operations teams!

