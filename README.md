# Fluentique

Fluentique is an AI-powered English-to-French language learning application developed as a university final year project.

The application combines vocabulary practice, grammar exercises, speech recognition, pronunciation feedback, conversational AI, text-to-speech, user progress tracking, and Firebase-backed authentication in a Kivy desktop interface with a FastAPI backend.

## Features

- User registration and login with Firebase Authentication
- Vocabulary quizzes with definitions and translations
- Grammar correction and sentence exercises
- French speech recognition using Wav2Vec2 with Google Speech-to-Text fallback
- AI-powered conversational tutor
- Pronunciation and speech feedback
- Amazon Polly French text-to-speech
- XP, streak, league, and leaderboard tracking
- User profiles and profile-picture uploads
- Exercise, speech, and conversation logging
- Local machine-learning models for grammar correction and speech recognition

## Tech Stack

### Frontend
- Python
- Kivy

### Backend
- FastAPI
- Uvicorn

### AI and Machine Learning
- PyTorch
- Hugging Face Transformers
- Wav2Vec2
- OpenAI API
- scikit-learn

### Cloud Services
- Firebase Authentication
- Firebase Firestore
- Firebase Storage
- Google Cloud Speech-to-Text
- Amazon Polly

### Audio and Data
- librosa
- soundfile
- sounddevice
- pydub
- pandas
- NumPy

## Project Structure

```text
Fluentique/
├── backend/
│   ├── data/
│   ├── routers/
│   ├── services/
│   ├── training/
│   ├── utils/
│   ├── config.py
│   └── server.py
├── frontend/
│   ├── assets/
│   ├── kv/
│   └── main_kivy.py
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

The following are intentionally excluded from Git:

```text
.env
secrets/
backend/models/
backend/logs/
logs/
```

These contain local credentials, downloaded models, or runtime data.

## Setup

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd <repository-folder>
```

### 2. Create a virtual environment

On Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

Some audio functionality may also require FFmpeg to be installed on the system.

## Environment Configuration

Create a local `.env` file in the project root using `.env.example` as the template:

```env
# === Firebase ===
FIREBASE_CREDENTIALS=secrets/firebase-admin.json
GOOGLE_APPLICATION_CREDENTIALS=secrets/google-creds.json
FIREBASE_WEB_API_KEY=your_firebase_web_api_key
FIREBASE_STORAGE_BUCKET=fluentiqueapp.firebasestorage.app

# === OpenAI ===
OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL=gpt-3.5-turbo

# === AWS Polly ===
AWS_REGION=eu-west-2
POLLY_VOICE_ID=Celine
POLLY_LANGUAGE_CODE=fr-FR

# === Backend ===
CORS_ORIGINS=*
FLUENTIQUE_API_URL=http://127.0.0.1:8000
```

Do not commit the real `.env` file or any private service-account credentials.

## Firebase and Google Cloud Credentials

Create a local folder:

```text
secrets/
```

Place the required service-account files inside it:

```text
secrets/
├── firebase-admin.json
└── google-creds.json
```

These files are excluded from Git and must never be committed.

## AWS Configuration

Amazon Polly uses the standard AWS credential chain.

Configure AWS credentials locally with:

```bash
aws configure
```

For this project, the default region is:

```text
eu-west-2
```

AWS access keys should not be stored inside the repository.

## Model Downloads

The trained models are too large to include in the Git repository.

### Grammar Correction Model

https://drive.google.com/drive/folders/1M2-dSHCIycibue8xoK9BjukzHmYU6n90?usp=sharing

### Fluentique Speech Recognition Model

https://drive.google.com/drive/folders/1sC5yiRhxDFsH-jjeD0RF240-Dyh65zPO?usp=sharing

Place the downloaded model files under:

```text
backend/models/
```

## Running Fluentique

### Start the backend

From the project root:

```bash
python -m backend.server
```

Alternatively:

```bash
uvicorn backend.server:app --host 127.0.0.1 --port 8000 --reload
```

### Start the frontend

Open a second terminal:

```bash
cd frontend
python main_kivy.py
```

The backend should be running before using features that depend on the API.

## Security

Sensitive credentials are deliberately kept outside version control.

The repository's `.gitignore` excludes:

- `.env` files containing real configuration values
- Firebase and Google Cloud service-account credentials
- AWS credential exports
- downloaded ML models
- runtime logs
- local progress data
- recorded user audio

Use `.env.example` only as a configuration template.

## Background

Fluentique was developed as a university final year project exploring how AI and cloud services can be combined to create a more interactive French-learning experience.

The project brings together natural language processing, speech recognition, text-to-speech, authentication, cloud storage, gamification, and a custom Kivy user interface in a single application.
