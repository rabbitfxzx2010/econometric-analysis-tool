# Email Setup Instructions for Formspree

## Current Issue
Formspree is failing to send emails to `zhangren080@gmail.com` while working for other addresses.

## Solution 1: Configure Formspree Dashboard (Recommended)

1. **Log into your Formspree account**: https://formspree.io/forms
2. **Find your form**: Look for form ID `xjkeegpn`
3. **Go to Form Settings**:
   - Click on your form
   - Navigate to "Settings" tab
   - Look for "Email Notifications" or "Recipients" section

4. **Add both email addresses as recipients**:
   - Add `r_z79@txstate.edu` as primary recipient
   - Add `zhangren080@gmail.com` as secondary recipient
   - Save the settings

5. **Check Spam Filters**:
   - Ask the Gmail user to check their spam/junk folder
   - Add `noreply@formspree.io` to Gmail contacts
   - Whitelist emails from `@formspree.io` domain

## Solution 2: Alternative Email Service

If Formspree continues to have issues with Gmail, consider these alternatives:

### Option A: Use EmailJS (Free tier available)
```javascript
// EmailJS can be integrated into Streamlit apps
// Better Gmail compatibility
```

### Option B: Use Google Forms
```html
<!-- Redirect feedback to a Google Form -->
<!-- Automatically emails both addresses -->
```

### Option C: SMTP Email (Requires email server)
```python
import smtplib
from email.mime.text import MIMEText
# Direct SMTP sending (more reliable but requires setup)
```

## Solution 3: Verify Gmail Settings

The Gmail account owner should:

1. **Check Gmail Filters**:
   - Go to Gmail Settings → Filters and Blocked Addresses
   - Ensure no filters are blocking Formspree emails

2. **Check Security Settings**:
   - Ensure 2FA is not blocking automated emails
   - Check Google Account security settings

3. **Add to Contacts**:
   - Add `noreply@formspree.io` to contacts
   - This helps bypass spam filters

## Current Form Configuration

The form is now configured with:
- Professional sender email: `feedback@econometrictool.app`
- Clean subject line with emoji: `📊 New Feedback - [timestamp]`
- Formatted message content
- Single request approach (no double-sending)

## Testing Steps

1. Submit a test feedback through the app
2. Check Formspree dashboard for delivery status
3. Check both email inboxes (including spam folders)
4. If still failing, try Solution 1 (dashboard configuration)

## Troubleshooting

If emails still don't work:
1. Check Formspree dashboard for error messages
2. Verify the form endpoint URL is correct
3. Ensure both email addresses are valid
4. Consider upgrading Formspree plan if using free tier limits

## Contact

For technical support:
- Formspree Support: https://help.formspree.io/
- Check Formspree status: https://status.formspree.io/
