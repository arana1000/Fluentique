# Fluentique

Fluentique is an AI-powered English-to-French language-learning application developed as a university final year project.

The application combines vocabulary practice, grammar exercises, French speech transcription, pronunciation feedback, conversational AI, text-to-speech, gamification, progress tracking, and Firebase-backed authentication in a Kivy desktop interface with a FastAPI backend.

## Features

- User registration and login with Firebase Authentication
- Vocabulary listening quizzes using curated local French exercise data
- Grammar translation and sentence-building exercises
- French speech transcription through OpenAI
- AI-generated grammar, pronunciation, and learning feedback
- Free-flowing French conversational tutor
- French-to-English translation support
- Amazon Polly text-to-speech
- XP, streak, league, and leaderboard tracking
- User profiles and profile-picture uploads with Firebase Storage
- Exercise, speech, quiz, and conversation logging
- Personalised tutor context based on previous learning mistakes

## Current Architecture

Fluentique uses a lightweight API-based runtime rather than loading large machine-learning models locally.

### Frontend

- Python
- Kivy
- Requests
- SoundDevice / SoundFile
- Matplotlib

### Backend

- FastAPI
- Uvicorn
- Firebase Admin SDK

### AI and Language Services

- OpenAI for:
  - French speech transcription
  - grammar correction and explanations
  - pronunciation/conversation feedback
  - conversational tutoring
  - French-to-English translation for AI-driven features
  - tutoring-session summaries
- Amazon Polly for French and English text-to-speech
- Amazon Translate for vocabulary-quiz English meanings
- Curated local JSON datasets for quizzes and grammar exercises

### Cloud Services

- Firebase Authentication
- Firebase Firestore
- Firebase Storage
- Amazon Polly

## Project Structure

```text
Fluentique/
├── backend/
│   ├── data/
│   ├── routers/
│   ├── services/
│   ├── utils/
│   ├── config.py
│   └── server.py
├── experiments/
│   ├── train_grammar_correction_model.py
│   └── train_speech_recognition_model.py
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
data/progress.json
recorded_audio*
```

These may contain private credentials, historical model files, runtime data, logs, or temporary user recordings.

## Setup

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd <repository-folder>
```

### 2. Create a virtual environment

Python 3.10 or 3.11 is recommended.

On Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

Some audio functionality also requires FFmpeg to be installed and available on the system PATH because `pydub` is used when combining generated audio.

## Environment Configuration

Create a local `.env` file in the project root using `.env.example` as the template:

```env
# === Firebase ===
FIREBASE_CREDENTIALS=secrets/firebase-admin.json
FIREBASE_WEB_API_KEY=your_firebase_web_api_key
FIREBASE_STORAGE_BUCKET=your_firebase_storage_bucket

# === OpenAI ===
OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL=gpt-4o-mini
OPENAI_TRANSCRIPTION_MODEL=gpt-4o-mini-transcribe

# === AWS Polly ===
AWS_REGION=eu-west-2
POLLY_VOICE_ID=Celine
POLLY_ENGLISH_VOICE_ID=Amy
POLLY_LANGUAGE_CODE=fr-FR

# === Backend / Frontend API ===
CORS_ORIGINS=*
FLUENTIQUE_API_URL=http://127.0.0.1:8000
```

Do not commit the real `.env` file or any private credentials.

## Firebase Credentials

Create a local folder in the project root:

```text
secrets/
```

Place the Firebase Admin service-account file inside it, for example:

```text
secrets/
└── firebase-admin.json
```

The path must match `FIREBASE_CREDENTIALS` in your local `.env`.

The `secrets/` directory is excluded from Git and service-account files must never be committed.

## AWS Configuration

Amazon Polly uses the standard AWS credential chain rather than hard-coded access keys.

Configure AWS locally with:

```bash
aws configure
```

The default Fluentique region is:

```text
eu-west-2
```

AWS access keys should never be stored inside the repository.

## Running Fluentique

The backend should be running before opening features in the frontend that depend on the API.

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

Open a second terminal from the project root:

```bash
python frontend/main_kivy.py
```

`FLUENTIQUE_API_URL` can be changed in `.env` if the backend is running at a different address.

## Runtime Data and Logs

Fluentique stores some learning history locally during use, including quiz results, grammar mistakes, speech feedback, chat transcripts, and local progress cache data.

These runtime files are intentionally ignored by Git.

Firebase is used for account-related data such as authentication, XP, league information, profile information, and profile-picture storage.

## Original Machine-Learning Experiments

The original university project explored local machine-learning approaches for grammar correction and French speech recognition, including model training/fine-tuning workflows.

Those experiments are preserved under:

```text
experiments/
```

They are retained as evidence of the original FYP research and implementation work, but they are **not part of the current production runtime**.

The portfolio version migrated runtime inference to API-based services to make Fluentique easier to run, maintain, deploy, and review without requiring large local model downloads or heavyweight ML dependencies.

As a result, the current application does not require PyTorch, Hugging Face Transformers, Wav2Vec2 model downloads, or Google Speech-to-Text credentials.

## Security

Sensitive credentials and generated runtime files are deliberately kept outside version control.

The repository's `.gitignore` protects items including:

- real `.env` files
- Firebase service-account credentials
- AWS credential exports
- historical/downloaded local model directories
- runtime logs
- local progress data
- recorded user audio
- Python virtual environments and cache files

Use `.env.example` only as a configuration template.

If a credential has ever been exposed publicly, removing it from the repository is not enough. The credential should also be revoked or rotated with the relevant provider.

## Background

Fluentique was developed as a university final year project exploring how artificial intelligence, language technologies, cloud services, and gamification can be combined into an interactive French-learning application.

The project originally included experiments with locally trained and fine-tuned ML models. The current portfolio version keeps that experimental work separately while using a cleaner production architecture based on OpenAI, Amazon Polly, Firebase, FastAPI, Kivy, and curated local learning datasets.

### OpenAI API availability

OpenAI-powered features require a user-supplied API key with available API credits.
If the configured key is invalid, rate-limited, or out of credits, Fluentique keeps
non-OpenAI features available and displays a user-friendly service message rather
than crashing. API usage may incur charges.
