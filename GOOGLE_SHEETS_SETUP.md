# 📊 Google Sheets Feedback Setup Guide

## Step 1: Create Google Cloud Project & Enable API

1. **Go to Google Cloud Console:** https://console.cloud.google.com/
2. **Create a new project:**
   - Click "Select a project" → "New Project"
   - Name: "streamlit-feedback" 
   - Click "Create"

3. **Enable Google Sheets API:**
   - Go to "APIs & Services" → "Library"
   - Search for "Google Sheets API"
   - Click on it and click "Enable"

## Step 2: Create Service Account

1. **Create Service Account:**
   - Go to "APIs & Services" → "Credentials"
   - Click "Create Credentials" → "Service Account"
   - Service account name: "feedback-collector"
   - Click "Create and Continue"
   - Skip optional steps → "Done"

2. **Generate Key:**
   - Click on your service account email
   - Go to "Keys" tab
   - Click "Add Key" → "Create new key"
   - Choose "JSON" format
   - Download the file (save as `service_account.json`)

## Step 3: Create Google Sheet

1. **Create new Google Sheet:** https://sheets.google.com
2. **Name it:** "App Feedback Responses"
3. **Set up headers in row 1:**
   - A1: "Timestamp"
   - B1: "Feedback"
   - C1: "User_Info"

4. **Get Sheet ID:**
   - Copy from URL: `https://docs.google.com/spreadsheets/d/SHEET_ID_HERE/edit`

5. **Share with Service Account:**
   - Click "Share" button
   - Add the service account email (from the JSON file)
   - Give "Editor" permission

## Step 4: Configure Streamlit Secrets

Create a file `.streamlit/secrets.toml` in your project:

```toml
[google_sheets]
type = "service_account"
project_id = "your-project-id"
private_key_id = "your-private-key-id"
private_key = "-----BEGIN PRIVATE KEY-----\nYOUR_PRIVATE_KEY_HERE\n-----END PRIVATE KEY-----\n"
client_email = "your-service-account@your-project.iam.gserviceaccount.com"
client_id = "your-client-id"
auth_uri = "https://accounts.google.com/o/oauth2/auth"
token_uri = "https://oauth2.googleapis.com/token"
auth_provider_x509_cert_url = "https://www.googleapis.com/oauth2/v1/certs"
client_x509_cert_url = "https://www.googleapis.com/robot/v1/metadata/x509/your-service-account%40your-project.iam.gserviceaccount.com"

SHEET_ID = "your_google_sheet_id_here"
```

## Step 5: Update Code (Already Done)

The code is already set up with:
- ✅ Google Sheets function placeholder
- ✅ Graceful fallback to local file
- ✅ Required dependencies in requirements.txt
- ✅ Error handling

## Step 6: Deploy to Streamlit Cloud

1. **Add secrets to Streamlit Cloud:**
   - Go to your app settings in Streamlit Cloud
   - Add the secrets from your `secrets.toml` file

2. **Uncomment the Google Sheets code:**
   - In `econometric_app.py`, uncomment the Google Sheets lines
   - Add your actual SHEET_ID

## Result

- ✅ Feedback submitted to Google Sheets automatically
- ✅ You can view all feedback in real-time
- ✅ Data is persistent and accessible from anywhere
- ✅ Easy to export/analyze feedback data

## Viewing Feedback

Simply open your Google Sheet to see all user feedback with timestamps!
