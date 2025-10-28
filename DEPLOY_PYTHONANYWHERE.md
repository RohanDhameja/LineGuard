# 🚀 PythonAnywhere Deployment Guide

Complete step-by-step guide to deploy FireGuard AI on PythonAnywhere.

## 📋 Pre-Deployment Checklist

- [ ] Test app locally at `http://127.0.0.1:5000`
- [ ] Verify ML models are trained (`models/` directory exists)
- [ ] Verify synthetic data is generated (`data/` directory exists)
- [ ] All dependencies listed in `requirements.txt`
- [ ] Code pushed to GitHub

## 🌐 PythonAnywhere Account Setup

### 1. Create Account
1. Go to [pythonanywhere.com](https://www.pythonanywhere.com)
2. Click **"Pricing & signup"**
3. Choose plan:
   - **Free**: Good for testing (limited CPU, always-on: No)
   - **$5 Hacker**: Recommended for production (more CPU, always-on: Yes)
4. Create account and verify email

## 📦 Method 1: Deploy from GitHub (Recommended)

### Step 1: Push to GitHub
```bash
cd ~/Desktop/Fire\ App

# Initialize git (if not already done)
git init
git add .
git commit -m "Initial commit - FireGuard AI"

# Create repo on GitHub, then:
git remote add origin <your-repo-url>
git push -u origin main
```

### Step 2: Clone on PythonAnywhere
1. Log into PythonAnywhere
2. Go to **Consoles** tab
3. Start a **Bash console**
4. Run:
```bash
cd ~
git clone <your-repo-url> fire-app
cd fire-app
```

### Step 3: Set Up Virtual Environment
```bash
# Create virtual environment
python3.10 -m venv venv

# Activate it
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 4: Generate Data and Models
```bash
# Still in activated venv
python generate_datasets.py
python train_risk_model.py
```

### Step 5: Configure Web App
1. Go to **Web** tab
2. Click **"Add a new web app"**
3. Choose your domain: `<username>.pythonanywhere.com`
4. Select **"Manual configuration"**
5. Choose **Python 3.10**

### Step 6: Configure WSGI File
1. In **Web** tab, click on **WSGI configuration file** link
2. Replace contents with:

```python
import sys
import os

# Add your project directory to the sys.path
project_home = '/home/<YOUR_USERNAME>/fire-app'
if project_home not in sys.path:
    sys.path.insert(0, project_home)

# Activate virtual environment
activate_this = '/home/<YOUR_USERNAME>/fire-app/venv/bin/activate_this.py'
if os.path.exists(activate_this):
    with open(activate_this) as f:
        exec(f.read(), dict(__file__=activate_this))

# Import Flask app
from app import app as application
```

**⚠️ Replace `<YOUR_USERNAME>` with your actual PythonAnywhere username!**

### Step 7: Configure Virtualenv
1. Still in **Web** tab
2. Find **"Virtualenv"** section
3. Enter: `/home/<YOUR_USERNAME>/fire-app/venv`

### Step 8: Configure Static Files
1. In **Web** tab, find **"Static files"** section
2. Add mapping:
   - **URL**: `/static/`
   - **Directory**: `/home/<YOUR_USERNAME>/fire-app/static/`

### Step 9: Reload and Test
1. Click big green **"Reload"** button at top of Web tab
2. Click on your app URL: `https://<username>.pythonanywhere.com`
3. Test all features!

## 📦 Method 2: Upload ZIP File

### Step 1: Create ZIP
```bash
cd ~/Desktop
zip -r fire-app.zip "Fire App" \
  -x "*.pyc" \
  -x "*__pycache__*" \
  -x "*.DS_Store" \
  -x "*test_*.py"
```

### Step 2: Upload
1. Log into PythonAnywhere
2. Go to **Files** tab
3. Click **"Upload a file"**
4. Upload `fire-app.zip`

### Step 3: Extract
In a **Bash console**:
```bash
cd ~
unzip fire-app.zip
mv "Fire App" fire-app  # Rename to avoid spaces
cd fire-app
```

Then follow **Steps 3-9** from Method 1 above.

## 🔧 Troubleshooting

### Issue: Import Error
**Error**: `ModuleNotFoundError: No module named 'flask'`

**Solution**: Activate virtualenv and reinstall
```bash
cd ~/fire-app
source venv/bin/activate
pip install -r requirements.txt
```

### Issue: ML Models Not Loading
**Error**: `ML model files not found`

**Solution**: Train models
```bash
cd ~/fire-app
source venv/bin/activate
python train_risk_model.py
```

### Issue: Static Files Not Loading
**Error**: CSS/JS files return 404

**Solution**: 
1. Check **Static files** mapping in Web tab
2. Verify directory path: `/home/<username>/fire-app/static/`
3. Reload web app

### Issue: 502 Bad Gateway
**Error**: Site shows 502 error

**Solution**:
1. Check **Error log** in Web tab
2. Common causes:
   - WSGI file has wrong username
   - Python version mismatch
   - Import error in `app.py`
3. Fix error, then reload

### Issue: Slow Performance
**Error**: App is very slow or times out

**Solution**:
- Free tier has CPU limits
- Upgrade to Hacker plan ($5/month)
- Or optimize by:
  - Reducing model size
  - Caching predictions
  - Limiting simulation weeks

## 📊 Checking Logs

### Error Log
View in **Web** tab → **"Error log"** link

### Server Log  
View in **Web** tab → **"Server log"** link

### Access Log
View in **Web** tab → **"Access log"** link

## 🔄 Updating Your App

After making changes locally:

### If using GitHub:
```bash
# Local machine
cd ~/Desktop/Fire\ App
git add .
git commit -m "Update: description"
git push

# PythonAnywhere console
cd ~/fire-app
git pull
# Reload web app
```

### If using ZIP:
1. Create new ZIP
2. Upload to PythonAnywhere
3. Extract and replace files
4. Reload web app

## ⚡ Performance Optimization

### 1. Enable WSGI Caching
Add to top of WSGI file:
```python
import sys
sys.dont_write_bytecode = True
```

### 2. Use Production-Ready Server
For paid accounts, consider gunicorn:
```bash
pip install gunicorn
```

### 3. Optimize ML Model
- Use model with fewer features
- Cache predictions
- Pre-compute common queries

### 4. Database Option
For large-scale deployment, consider adding PostgreSQL:
- Available on paid plans
- Store historical data
- Cache predictions

## 🔐 Security Best Practices

1. **Never commit secrets**: Use environment variables
2. **Update dependencies**: `pip list --outdated`
3. **HTTPS only**: PythonAnywhere provides free SSL
4. **Rate limiting**: Add to prevent abuse
5. **API keys**: If integrating external services

## 📞 Getting Help

### PythonAnywhere Support
- **Forums**: [pythonanywhere.com/forums](https://www.pythonanywhere.com/forums/)
- **Email**: help@pythonanywhere.com
- **Documentation**: [help.pythonanywhere.com](https://help.pythonanywhere.com)

### App-Specific Issues
- Check `ML_RISK_MODEL_README.md` for ML issues
- Check `README.md` for general app issues

## ✅ Post-Deployment Checklist

After deployment, verify:

- [ ] Home page loads correctly
- [ ] Map displays with zones
- [ ] Week slider works
- [ ] Zone tooltips show data
- [ ] ML predictions display in metrics panel
- [ ] "Notify Authority" button works
- [ ] Tab switching works
- [ ] All API endpoints respond
- [ ] No errors in error log
- [ ] Performance is acceptable

## 🎉 You're Live!

Your FireGuard AI app should now be accessible at:
```
https://<your-username>.pythonanywhere.com
```

Share the link and help prevent wildfires! 🔥

---

**Need help?** Open an issue on GitHub or check the PythonAnywhere forums.

