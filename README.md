# 🫀 Hypertension Risk Early Warning System

**Version:** 1.0.0 (Week 8 — Final Release)
**Build:** 8-week development cycle
**Status:** Production-ready prototype

---

## 📖 Overview

A comprehensive, explainable, multi-language Hypertension Risk Early Warning System
built with Flask. Combines machine learning, SHAP explainability, complete care
coordination, and analytics.

## ✨ Features

### Core
- 🔐 **User Authentication** — bcrypt-hashed passwords, session management
- 📊 **Risk Assessment** — 6 health indicators, real-time prediction
- 🔍 **SHAP Explainability** — Understand exactly what drives risk
- 📚 **Personalized Education** — Tailored health advice per user

### Care Coordination
- 🏥 **Facility Locator** — OpenStreetMap-powered map with distances
- 🩺 **Telemedicine** — Jitsi Meet video consultation links
- 📅 **Appointments** — In-person and telemedicine booking
- ⏰ **Reminders** — Auto and custom health reminders
- 🚨 **Emergency SOS** — Emergency contact + one-tap calling
- 👨‍⚕️ **Doctor Directory** — Filterable specialist listings

### Engagement
- 🔔 **Notifications** — In-app center with preferences
- 📄 **PDF Reports** — Full health record (ReportLab)
- 📊 **CSV Export** — Assessment data export
- 🌍 **Multi-Language** — English, French, Kiswahili
- 📈 **Advanced Analytics** — Trends, insights, population health
- 📱 **QR Sharing** — Instant cross-device access

### Production
- 🚀 **Health Check Endpoint** — `/health`
- ✅ **Unit Tests** — `python tests.py`
- ⚙️ **WSGI Ready** — Gunicorn compatible
- 🔒 **Env Variables** — `.env` support

---

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train ML model
python model.py

# 3. Seed sample doctors (optional)
python seed_doctors.py

# 4. Run the application
python app.py