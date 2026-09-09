"""
main_kivy.py

Main application script for the Fluentique language learning app. This file handles all screen management,
authentication, exercises, speech feedback, AI interaction, and logging functionality using Kivy.

Modules covered:
- User registration/login
- Vocabulary quiz engine
- Grammar correction and sentence puzzles
- Speech practice with AI pronunciation feedback
- AI conversation tutor (text and voice)
- Profile management and result logging
"""

# === System and Standard Library Imports ===
import os
import sys
import json
import pickle
import threading
import tempfile
import io
from pathlib import Path
from datetime import datetime
from random import shuffle, randint
from collections import defaultdict, Counter
import urllib.request

# Ensure backend module imports work when running from the 'app/' directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# === Third-Party Libraries ===
import pandas as pd
import requests
import torch
import librosa
import sounddevice as sd
import soundfile as sf
import scipy.io.wavfile as wavfile
import matplotlib.pyplot as plt
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import google.cloud.speech as speech
from plyer import filechooser

# === Kivy Core ===
from kivy.app import App
from kivy.lang import Builder
from kivy.core.window import Window
from kivy.core.text import LabelBase
from kivy.core.audio import SoundLoader
from kivy.clock import Clock
from kivy.metrics import dp
from kivy.properties import (
    NumericProperty, ListProperty, StringProperty
)
from kivy.uix.screenmanager import Screen, ScreenManager
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.popup import Popup
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.relativelayout import RelativeLayout
from kivy.uix.behaviors import ButtonBehavior
from kivy.uix.image import Image, AsyncImage
from kivy.uix.widget import Widget
from kivy.graphics import (
    Color, Rectangle, RoundedRectangle, Line, Ellipse,
    StencilPush, StencilUse, StencilUnUse, StencilPop
)

# === Internal Utilities and Services ===
from backend.utils.tts_utils import play_word_with_polly
from backend.services.fluentique_quiz import get_word_info
from backend.utils.progress_manager import ProgressManager
from backend.utils.firebase_utils import (
    get_all_users, upload_profile_image, update_profile_picture_url
)
from backend.utils.auth_utils import verify_token
from backend.config import GOOGLE_CREDS_PATH


# main_kivy.py
# ----------------------
# This is the main script that defines the user interface and manages 
# navigation within the app. It integrates various components, such as 
# authentication, quizzes, grammar exercises, and conversational AI, 
# providing a seamless experience for the user. It connects the frontend 
# (Kivy) to the backend (FastAPI) for real-time interactions.
# 
# Functions:
# - setup_ui: Initializes the user interface, including all screens and navigation.
# - handle_navigation: Manages transitions between different screens (login, quizzes, etc.).
# - interact_with_backend: Sends requests to the backend to fetch or submit data.


# === Application Constants ===

# Define central project paths and backend connection settings
PROJECT_ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIR = Path(__file__).resolve().parent
ASSETS_DIR = FRONTEND_DIR / "assets"
KV_DIR = FRONTEND_DIR / "kv"
BASE_LOG_DIR = PROJECT_ROOT / "backend" / "logs"

API_BASE_URL = os.getenv(
    "FLUENTIQUE_API_URL",
    "http://127.0.0.1:8000",
).rstrip("/")

REQUEST_TIMEOUT = 15

# Global instance of ProgressManager for handling XP, levels, and streaks
progress_manager = ProgressManager()

# Function to speak a word using Amazon Polly on a separate thread to prevent UI blocking
def speak_word_threaded(word):
    threading.Thread(
        target=play_word_with_polly,
        args=(word,),
        daemon=True,
    ).start()

# Register custom fonts for consistent visual branding
LabelBase.register(name="ProximaSoft", fn_regular="assets/ProximaSoft-Regular.ttf")
LabelBase.register(name="ProximaSoft-Bold", fn_regular="assets/ProximaSoft-Bold.ttf")

# Set default window size (mobile simulation: 360x740)
Window.size = (360, 740)


class LeaderboardScreen(Screen):
    """
    Screen responsible for displaying the leaderboard, showing users sorted by XP in descending order.
    Includes current user highlighting, avatars, XP display, and league information.
    """

    def on_pre_enter(self):
        """
        This method runs automatically just before the screen is shown.
        It ensures that the leaderboard is up to date whenever the user enters the screen.
        """
        self.update_leaderboard()

    def update_leaderboard(self):
        """
        Fetches fresh leaderboard data from Firebase and renders it on the screen.
        """
        self.leaderboard_data = self.get_leaderboard_data()
        self.display_leaderboard()

    def display_leaderboard(self):
        """
        Creates the visual layout for the leaderboard entries. Each user gets a horizontal container with:
        - A profile picture
        - Their name and league
        - Their XP
        The current user is highlighted with a coloured border.
        """
        # Clear any existing leaderboard entries
        self.ids.leaderboard_box.clear_widgets()

        for i, (uid, name, xp, league, profile_pic_url, is_you) in enumerate(self.leaderboard_data, start=1):
            # Display "You" instead of your email for privacy
            display_name = "You" if is_you else name

            # Each user row is a horizontal box layout
            container = BoxLayout(
                orientation='horizontal',
                size_hint_y=None,
                height=dp(60),
                padding=[dp(10), 0],
                spacing=dp(10),
            )

            # Draw a white background for each container
            with container.canvas.before:
                Color(1, 1, 1, 1)
                background = RoundedRectangle(pos=container.pos, size=container.size, radius=[10])

            # Keep background position and size synced to the container
            def bind_bg(widget, bg_rect):
                widget.bind(
                    pos=lambda inst, *a: setattr(bg_rect, 'pos', inst.pos),
                    size=lambda inst, *a: setattr(bg_rect, 'size', inst.size),
                )
            bind_bg(container, background)

            # Highlight the current user with a border (using canvas.after)
            if is_you:
                with container.canvas.after:
                    Color(0.7, 0.6, 0.95, 1)  # Lavender border colour
                    border = Line(width=2)

                # Sync border rectangle with the container's position and size
                def update_border(instance, *args):
                    border.rectangle = (instance.x, instance.y, instance.width, instance.height)
                container.bind(pos=update_border, size=update_border)

            # Create label with user's name and league
            name_label = Label(
                text=f"[b]{i}. {display_name}[/b]\n{league}",
                markup=True,  # Enable bold tags
                font_name="assets/ProximaSoft-Regular",
                font_size='16sp',
                halign='left',
                valign='middle',
                color=(0, 0, 0, 1),
                size_hint_x=0.7,
            )
            name_label.bind(size=name_label.setter('text_size'))

            # Create XP label aligned right
            xp_label = Label(
                text=f"{xp} pts",
                font_name="assets/ProximaSoft-Regular",
                font_size='16sp',
                halign='right',
                valign='middle',
                color=(0, 0, 0, 1),
                size_hint_x=0.3,
            )
            xp_label.bind(size=xp_label.setter('text_size'))

            # Profile image: fallback to default if none is set
            avatar_source = profile_pic_url if profile_pic_url else 'assets/default_avatar.png'
            avatar = RoundedAsyncImage(
                source=avatar_source,
                size=(dp(40), dp(40))
            )
            avatar_container = FloatLayout(size_hint=(None, 1), width=dp(50))
            avatar_container.add_widget(avatar)

            # Assemble the full row and add to the leaderboard display
            container.add_widget(avatar_container)
            container.add_widget(name_label)
            container.add_widget(xp_label)
            self.ids.leaderboard_box.add_widget(container)

    def refresh_leaderboard(self):
        """
        Simulates a dynamic leaderboard update.
        It randomises XP (except for the current user, which fetches real progress),
        then re-sorts the list and re-renders it.
        """
        new_data = []
        for uid, name, xp, league, profile_pic_url, is_you in self.leaderboard_data:
            if is_you:
                # Get actual XP for current user
                updated_xp = progress_manager.get_progress()["xp"]
                new_data.append((uid, name, updated_xp, league, profile_pic_url, is_you))
            else:
                # Simulate XP fluctuation for others
                new_xp = xp + randint(-20, 20)
                new_data.append((uid, name, new_xp, league, profile_pic_url, is_you))

        # Sort the updated data again based on XP
        new_data.sort(key=lambda x: x[2], reverse=True)
        self.leaderboard_data = new_data
        self.display_leaderboard()

    def get_leaderboard_data(self):
        """
        Retrieves all user data from Firebase, including XP, league, and profile pictures.
        Flags the current user for highlighting and returns a sorted list by XP.

        Returns:
            List of tuples: (uid, name, xp, league, profile_pic_url, is_you)
        """
        try:
            users = get_all_users()  # Get all registered users from Firebase
            leaderboard = []

            # Identify the current user by verifying the token
            app = App.get_running_app()
            current_user_info = verify_token(app.token)
            current_uid = current_user_info["uid"]

            # Build structured leaderboard entries
            for user in users:
                uid = user["uid"]
                name = user.get("email", "Anonymous").split("@")[0]
                xp = user.get("xp", 0)
                league = user.get("league", "Bronze")
                profile_picture_url = user.get("profile_picture_url", "")
                is_you = (uid == current_uid)

                leaderboard.append((uid, name, xp, league, profile_picture_url, is_you))

            # Sort leaderboard by XP in descending order
            leaderboard.sort(key=lambda x: x[2], reverse=True)
            return leaderboard

        except Exception as e:
            print(f"Error fetching leaderboard: {e}")
            return []

class RoundedAsyncImage(RelativeLayout):
    """
    Custom Kivy widget that displays a remote image (via AsyncImage) with a circular (rounded) mask.
    This is used for rendering rounded profile pictures on the leaderboard or user profile sections.
    """

    def __init__(self, source, size=(dp(40), dp(40)), **kwargs):
        """
        Constructor to initialise the rounded image.

        Args:
            source (str): URL or local path of the image to load.
            size (tuple): Size of the image (width, height) in density-independent pixels.
        """
        super().__init__(**kwargs)

        # Disable automatic scaling
        self.size_hint = (None, None)
        self.size = size
        self.source = source
        self.padding = [0, 0, 0, 0]

        # === Begin circular masking using Kivy's stencil instructions ===
        with self.canvas.before:
            StencilPush()  # Begin stencil layer
            self.mask = Ellipse(pos=self.pos, size=self.size)  # Define a circular clipping area
            StencilUse()   # Activate the stencil mask

        # === Create and configure the actual image widget ===
        self.image = AsyncImage(allow_stretch=True, keep_ratio=True)
        self.image.source = source
        self.image.size_hint = (None, None)
        self.image.size = self.size
        self.image.pos = self.pos
        self.image.reload()  # Ensure the image refreshes correctly
        self.add_widget(self.image)  # Add the image into the layout

        # === Stop using stencil to allow other elements to draw normally ===
        with self.canvas.after:
            StencilUnUse()
            StencilPop()

        # Keep the mask and image size/position synced with the layout
        self.bind(pos=self.update_mask, size=self.update_mask)

    def update_mask(self, *args):
        """
        Updates the position and size of the mask and image to stay in sync with layout resizing.
        This is triggered automatically by Kivy when the layout size or position changes.
        """
        self.mask.pos = self.pos
        self.mask.size = self.size
        self.image.pos = self.pos
        self.image.size = self.size

class ProfileScreen(Screen):
    """
    This screen allows the user to view and update their profile information,
    including username, email, and profile picture.
    It also supports logging out of the app.
    """

    def on_pre_enter(self):
        """
        Automatically called just before the screen appears.
        This method retrieves and displays the current user's profile details:
        - Email (from the login token)
        - Username and profile picture (from Firestore)
        """
        app = App.get_running_app()
        user_info = verify_token(app.token)  # Decode token to get user UID and email

        uid = user_info["uid"]
        email = user_info.get("email", "Unknown")
        self.ids.email_input.text = email  # Show email (read-only)

        # Import only when needed to reduce load time
        from backend.utils.firebase_utils import get_user_data

        user_data = get_user_data(uid)
        if user_data:
            # Populate the username field
            self.ids.username_input.text = user_data.get("username", "")

            # If user has a profile image saved, download it and display it
            profile_pic_url = user_data.get("profile_picture_url", "")
            if profile_pic_url:
                try:
                    temp_dir = tempfile.gettempdir()
                    filename = os.path.join(temp_dir, f"profile_pic_{uid}.jpg")
                    urllib.request.urlretrieve(profile_pic_url, filename)
                    self.ids.profile_image.source = filename
                except Exception as e:
                    print("Failed to load profile image:", e)

    def upload_profile_picture(self):
        """
        Allows the user to choose and upload a new profile picture.

        Step-by-step:
        1. Open file dialog to select an image
        2. Upload image to Firebase Storage
        3. Save download URL to Firestore
        4. Download image to local temp folder and refresh Kivy view
        """
        file_path = filechooser.open_file(
            title="Pick a profile picture",
            filters=[("Image files", "*.png;*.jpg;*.jpeg")]
        )
        if file_path:
            file_path = file_path[0]  # Take the first selected file
            app = App.get_running_app()
            user_info = verify_token(app.token)
            uid = user_info["uid"]

            # Upload image to Firebase Storage
            uploaded_url = upload_profile_image(uid, file_path)

            if uploaded_url:
                # Save the image URL to Firestore
                update_profile_picture_url(uid, uploaded_url)

                # Download image back to display it immediately in Kivy
                try:
                    temp_dir = tempfile.gettempdir()
                    filename = os.path.join(temp_dir, f"profile_pic_{uid}.jpg")
                    urllib.request.urlretrieve(uploaded_url, filename)
                    self.ids.profile_image.source = filename
                    self.ids.profile_image.reload()  # Force Kivy to refresh the image
                except Exception as e:
                    print("Failed to load updated profile image:", e)

    def save_profile(self):
        """
        Saves the updated username to Firestore.
        Only updates if the input is non-empty.
        """
        new_username = self.ids.username_input.text.strip()
        if not new_username:
            return  # Do not update if field is empty

        app = App.get_running_app()
        user_info = verify_token(app.token)
        uid = user_info["uid"]

        # Update Firestore record for this user
        from backend.utils.firebase_utils import update_username
        update_username(uid, new_username)

    def logout_user(self):
        """
        Logs out the current user by:
        - Clearing the session token
        - Resetting local progress state
        - Clearing input fields
        - Navigating to the login screen
        """
        app = App.get_running_app()
        app.token = None  # Clear login session
        progress_manager.reset_progress()

        # Clear profile fields in the UI
        self.ids.username_input.text = ""
        self.ids.email_input.text = ""
        self.ids.profile_image.source = "assets/default_avatar.png"

        # Go back to the login screen
        app.root.current = "login"

class UploadAvatarScreen(Screen):
    """
    Screen that allows the user to upload a profile picture from their device.
    Once uploaded, the image is saved to Firebase, downloaded locally, and reflected
    both in the upload preview and in the user's profile screen.
    """

    def pick_and_upload_image(self):
        """
        Handles the full avatar upload process:

        1. Opens a file chooser to let the user select an image file.
        2. Uploads the selected image to Firebase Storage.
        3. Stores the download URL in Firestore under the user's profile.
        4. Downloads the image locally and updates both the preview image
           on this screen and the profile picture on the Profile screen.
        """

        # Step 1: Let user choose an image file
        file_path = filechooser.open_file(
            title="Choose a profile image",
            filters=[("Image files", "*.png;*.jpg;*.jpeg")]
        )
        if file_path:
            file_path = file_path[0]  # Take the first selected file

            # Get current user's UID from the token
            app = App.get_running_app()
            user_info = verify_token(app.token)
            uid = user_info["uid"]

            # Step 2: Upload image to Firebase Storage
            url = upload_profile_image(uid, file_path)

            if url:
                # Step 3: Save the download URL to Firestore
                update_profile_picture_url(uid, url)

                try:
                    # Step 4: Download the image and update it locally
                    temp_dir = tempfile.gettempdir()
                    filename = os.path.join(temp_dir, f"profile_pic_{uid}.jpg")
                    urllib.request.urlretrieve(url, filename)

                    # === Update avatar preview on the upload screen ===
                    self.ids.preview_image.source = filename
                    self.ids.preview_image.reload()

                    # === Also update the avatar on the Profile screen ===
                    profile_screen = app.root.get_screen('profile')
                    profile_screen.ids.profile_image.source = filename
                    profile_screen.ids.profile_image.reload()

                    # Navigate back to Profile screen after a short delay
                    Clock.schedule_once(lambda dt: setattr(app.root, 'current', 'profile'), 0.1)

                except Exception as e:
                    print("Failed to load/display uploaded image:", e)

class MicButton(ButtonBehavior, Image):
    """
    A custom UI widget that behaves like a button but displays an image.

    This is used for microphone activation (e.g., push-to-talk).
    Inherits from ButtonBehavior to handle touch/click input,
    and from Image to display an icon (such as a microphone).
    
    This is useful in Kivy when you want an image that functions like a button.
    """
    pass

class IntroContinueScreen(Screen):
    """
    First intro screen shown to the user after launching the app.

    Typically displays the Fluentique logo and a 'Continue' button.
    This screen acts as a brief entry point before onboarding or login.
    All layout and behaviour is handled via KV file.
    """
    pass

class IntroGetStartedScreen(Screen):
    """
    Second intro screen shown when the user presses 'Continue'.

    Usually contains a short description of the app and options to:
    - Get started (new user)
    - Log in (existing user)

    This screen transitions the user into the registration or login flow.
    """
    pass

class LoginScreen(Screen):
    """
    Screen where existing users can log into the Fluentique app.
    This screen handles sending login credentials to the backend API,
    processing the response, managing the session token, and updating the user's XP and streak.
    """

    def login_user(self):
        """
        Handles the full login flow:

        1. Gets email and password from the input fields.
        2. Sends credentials to the FastAPI backend for authentication.
        3. If successful:
           - Stores the returned token for the session.
           - Deletes any old local profile images.
           - Loads the user's XP and streak from Firestore.
           - Navigates to the home screen.
        4. If login fails, shows an error popup.
        """
        email = self.ids.email_input.text
        password = self.ids.password_input.text

        # Step 1: Send login request to backend API
        url = f"{API_BASE_URL}/auth/login"
        data = {"email": email, "password": password}
        response = requests.post(url, json=data, timeout=REQUEST_TIMEOUT)

        if response.status_code == 200:
            # Step 2: Login successful, extract the token
            data = response.json()
            token = data.get("idToken")
            app = App.get_running_app()
            app.store_token(token)  # Store session token in the app

            # Step 3: Clean up old profile pictures from temp directory
            temp_dir = tempfile.gettempdir()
            for f in os.listdir(temp_dir):
                if f.startswith("profile_pic_"):
                    try:
                        os.remove(os.path.join(temp_dir, f))
                    except Exception:
                        pass  # Ignore failures silently

            # Step 4: Load user's XP and streak from Firestore
            from backend.utils.auth_utils import verify_token
            user_info = verify_token(token)  # Decode token to get UID
            uid = user_info["uid"]
            progress_manager.load_user(uid)  # Load XP/streak/progress for user

            # (Repeat cleanup in case of race condition or redundancy)
            temp_dir = tempfile.gettempdir()
            for f in os.listdir(temp_dir):
                if f.startswith("profile_pic_"):
                    try:
                        os.remove(os.path.join(temp_dir, f))
                    except Exception:
                        pass

            # Step 5: Navigate to loading screen briefly, then go to home screen
            app.root.current = 'loading'
            Clock.schedule_once(lambda dt: setattr(app.root, 'current', 'home_incomplete'), 3)

        else:
            # Login failed — show error message to user
            error_detail = response.json().get("detail", "Login failed")
            self.show_popup("Error", str(error_detail))

    def show_popup(self, title, message):
        """
        Displays a simple popup alert with a title and message.
        Used for showing login errors or other notifications.
        """
        popup = Popup(
            title=title,
            content=Label(text=message),
            size_hint=(None, None),
            size=(300, 200)
        )
        popup.open()

class RegistrationScreen(Screen):
    """
    This screen handles the registration of new users.
    It collects user input, validates passwords, sends registration data to the backend API,
    and sets up the session token and progress state upon success.
    """

    def register_user(self):
        """
        Executes the user registration process:

        1. Retrieves and validates email and password fields.
        2. Sends a POST request to the FastAPI backend to register the new user.
        3. If successful:
           - Saves the token
           - Clears any cached profile pictures
           - Loads XP state from Firestore
           - Shows a success popup and redirects to the home screen
        4. If registration fails, an error popup is shown.
        """
        email = self.ids.email_input.text
        password = self.ids.password_input.text
        confirm_password = self.ids.confirm_password_input.text

        # Step 1: Check if passwords match before proceeding
        if password != confirm_password:
            self.show_popup("Error", "Passwords do not match!")
            return

        # Step 2: Prepare data payload for the backend
        payload = {
            "email": email,
            "password": password
        }

        try:
            # Step 3: Send POST request to registration endpoint
            response = requests.post(f"{API_BASE_URL}/auth/register", json=payload, timeout=REQUEST_TIMEOUT)

            if response.status_code == 200:
                # Step 4: Registration successful — extract session token
                data = response.json()
                token = data.get("idToken")
                app = App.get_running_app()
                app.store_token(token)  # Save token for session continuity

                # Step 5: Remove any old cached profile pictures
                temp_dir = tempfile.gettempdir()
                for f in os.listdir(temp_dir):
                    if f.startswith("profile_pic_"):
                        try:
                            os.remove(os.path.join(temp_dir, f))
                        except Exception:
                            pass

                # Step 6: Load the new user's XP and progress from Firestore
                from backend.utils.auth_utils import verify_token
                user_info = verify_token(token)
                uid = user_info["uid"]
                progress_manager.load_user(uid)

                # Step 7: Inform user and redirect
                self.show_popup("Success", "Registration successful!")
                app.root.current = 'loading'
                Clock.schedule_once(lambda dt: setattr(app.root, 'current', 'home_incomplete'), 3)

            else:
                # Step 8: Backend returned an error
                error = response.json().get('detail', 'Registration failed')
                self.show_popup("Error", error)

        except Exception as e:
            # Step 9: Connection or other unexpected error
            self.show_popup("Error", str(e))

    def show_popup(self, title, message):
        """
        Displays a popup window with a given title and message.
        Used for both success and error feedback during registration.
        """
        popup = Popup(
            title=title,
            content=Label(text=message),
            size_hint=(None, None),
            size=(300, 200)
        )
        popup.open()

class LoadingScreen(Screen):
    """
    Transitional loading screen displayed after login or registration.
    It gives the user a visual break while data is being fetched or the UI is prepared.

    This screen is often shown for a few seconds before redirecting to the actual home screen.
    All visual content and transitions are handled in the KV file.
    """
    pass

class HomeIncompleteScreen(Screen):
    """
    This is the main home screen shown when the user has not completed all their tasks.

    It displays the user's XP and streak statistics pulled from the ProgressManager.
    This screen motivates continued engagement by showing partial progress.
    """

    def on_pre_enter(self):
        """
        Automatically called when the screen is about to be shown.
        It updates the XP and streak labels based on the latest user progress.
        """
        stats = progress_manager.get_progress()
        self.ids.xp_label.text = f"XP: {stats.get('xp', 0)}"
        self.ids.streak_label.text = f"Streak: {stats.get('streak', 0)} days"

class HomeCompleteScreen(Screen):
    """
    This is the alternate home screen shown when the user has completed all required exercises.

    It may display a completion badge, rewards, or a motivational message.
    The layout and visuals are managed in the KV file.
    """
    pass

class QuizScreen(Screen):
    """
    This screen handles the full quiz flow for the Fluentique app.

    It includes:
    - Vocabulary question delivery (via backend API)
    - User interaction and answer submission
    - Feedback with definitions and translations
    - XP tracking, main round + challenge round logic
    """

    selected_option = NumericProperty(0)      # Which answer the user has currently selected (1–4)
    question_count = NumericProperty(0)       # Number of questions completed in main quiz
    total_progress = NumericProperty(0)       # Tracks progress across main + challenge rounds

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Initialise quiz state
        self.question_count = 0
        self.main_score = 0
        self.challenge_score = 0
        self.incorrect_words = []
        self.in_challenge_round = False
        self.challenge_questions = []
        self.challenge_index = 0

        # === Load the challenge round prediction model ===
        MODEL_PATH = Path(__file__).resolve().parent.parent / "backend" / "models" / "fluentique_predictor.pkl"
        with open(MODEL_PATH, "rb") as f:
            model_data = pickle.load(f)

        self.predictor = model_data["model"]
        self.le_level = model_data["le_level"]
        self.le_correct = model_data["le_correct"]
        self.le_selected = model_data["le_selected"]

    def show_popup(self, title, message):
        """
        Simple popup used to show errors or alerts.
        """
        popup = Popup(title=title, content=Label(text=message),
                      size_hint=(None, None), size=(300, 200))
        popup.open()

    def on_enter(self):
        """
        Called when this screen is entered.
        Begins the quiz by loading the first question.
        If in challenge mode, it loads a challenge question instead.
        """
        if not self.in_challenge_round:
            self.load_new_quiz_question()
        else:
            self.load_challenge_question()

    def load_new_quiz_question(self):
        """
        Calls the backend API to fetch a new quiz question.
        The correct answer and options are passed to `display_question`.
        """
        app = App.get_running_app()
        headers = {"Authorization": f"Bearer {app.token}"}
        response = requests.get(f"{API_BASE_URL}/quiz/question", headers=headers, timeout=REQUEST_TIMEOUT)

        if response.status_code == 200:
            data = response.json()
            options = data.get("options", [])
            self.correct_answer = data.get("question", "")
            self.display_question(options)
        else:
            self.show_popup("Error", f"Quiz failed: {response.status_code}")

    def load_challenge_question(self):
        """
        Loads the next question in the challenge round from pre-selected local data.
        """
        q = self.challenge_questions[self.challenge_index]
        self.correct_answer = q["correct"]
        self.display_question(q["options"])

    def display_question(self, options):
        """
        Renders the 4 multiple-choice options and resets the feedback bar.

        Args:
            options (list): List of 4 answer options to show.
        """
        self.ids.feedback_bar.clear_widgets()
        self.ids.definition_box.clear_widgets()
        self.ids.feedback_bar.canvas.before.clear()

        # Reset all option buttons
        for i in range(1, 5):
            btn = self.ids[f'option{i}']
            btn.background_color = (246/255, 247/255, 255/255, 1)
            btn.text = options[i-1]

        # Speak the correct word aloud using Amazon Polly
        Clock.schedule_once(
            lambda dt: speak_word_threaded(self.correct_answer),
            0.2,
        )

    def select_option(self, option_num):
        """
        Updates the selected option and highlights it.

        Args:
            option_num (int): The button number clicked by the user (1–4).
        """
        self.selected_option = option_num
        for i in range(1, 5):
            btn = self.ids[f'option{i}']
            btn.background_color = (0.7, 0.6, 0.9, 1) if i == option_num else (246/255, 247/255, 255/255, 1)

    def submit_answer(self):
        """
        Checks the selected answer and provides visual/audio/text feedback.
        Updates the user's score and progresses to the next question.
        """

        # Prevent submission without selecting an option
        if self.selected_option == 0:
            self.show_popup("Select an option", "Please select an answer before submitting.")
            return

        selected_text = self.ids[f'option{self.selected_option}'].text
        correct = selected_text == self.correct_answer

        # Setup feedback bar
        feedback_bar = self.ids.feedback_bar
        feedback_bar.clear_widgets()
        feedback_bar.height = dp(80)
        feedback_bar.canvas.before.clear()

        with feedback_bar.canvas.before:
            Color(0, 1, 0, 0.3) if correct else Color(1, 0, 0, 0.3)
            Rectangle(pos=feedback_bar.pos, size=feedback_bar.size)

        # Text feedback: Good Job or Poor Job
        feedback_bar.add_widget(Label(
            text="Good Job!" if correct else f"Poor Job! Correct: {self.correct_answer}",
            color=(0, 1, 0, 1) if correct else (1, 0, 0, 1),
            font_name="assets/ProximaSoft-Bold",
            halign='center'
        ))

        # === Show Definitions ===
        definition_box = self.ids.definition_box
        definition_box.clear_widgets()

        if correct:
            correct_info = get_word_info(self.correct_answer)
            definition_box.add_widget(Label(
                text=f"{correct_info['definition']}",
                color=(0, 0, 0, 1),
                font_name="assets/ProximaSoft-Regular",
                halign='center'
            ))
        else:
            correct_info = get_word_info(self.correct_answer)
            selected_info = get_word_info(selected_text)
            definition_box.add_widget(Label(
                text=f"'{self.correct_answer}' means: {correct_info['translation']}\n"
                     f"You chose '{selected_text}' which means: {selected_info['translation']}",
                color=(1, 0, 0, 1),
                font_name="assets/ProximaSoft-Regular",
                halign='center'
            ))

        # === XP and Score Logic ===
        if correct:
            if not self.in_challenge_round:
                self.main_score += 1
                progress_manager.increment_quiz()  # +10 XP
            else:
                self.challenge_score += 1
                progress_manager.increment_quiz()  # +10 XP
        else:
            self.incorrect_words.append(self.correct_answer)

        # === Update Progress Tracking ===
        if not self.in_challenge_round:
            self.question_count += 1
            self.total_progress = self.question_count
        else:
            self.total_progress = 20 + self.challenge_index + 1

        # Move to next question after delay
        Clock.schedule_once(lambda dt: self.check_next_step(), 3)

    def check_next_step(self, *args):
        """
        Decides whether to continue the quiz, start the challenge round, or show final results.
        """
        if not self.in_challenge_round:
            if self.question_count >= 20:
                self.prepare_challenge_round()
                progress_manager.update_streak()
            else:
                self.load_new_quiz_question()
        else:
            self.challenge_index += 1
            if self.challenge_index < len(self.challenge_questions):
                self.load_challenge_question()
            else:
                # Quiz complete → show results
                App.get_running_app().root.current = 'results'
                App.get_running_app().root.get_screen('results').display_results(
                    self.main_score, self.challenge_score, self.incorrect_words
                )
                progress_manager.update_streak()

    def prepare_challenge_round(self):
        """
        Prepares and launches the challenge round by:
        - Loading hard questions
        - Using the predictor model to rank difficulty
        - Selecting the top 5 hardest words
        """
        self.in_challenge_round = True
        self.ids.feedback_bar.clear_widgets()
        self.ids.feedback_bar.add_widget(Label(
            text="Entering Challenge Round!",
            color=(1, 0.5, 0, 1),
            font_name="assets/ProximaSoft-Bold"
        ))

        # Load local data
        DATA_PATH = Path(__file__).resolve().parent.parent / "backend" / "data" / "french_quiz_data_aws.json"
        with open(DATA_PATH, "r", encoding="utf-8") as f:
            quiz_data = json.load(f)

        challenge_pool = []
        for q in quiz_data:
            if q["level"] == "hard":
                try:
                    level_encoded = self.le_level.transform([q["level"]])[0]
                    correct_encoded = self.le_correct.transform([q["correct"]])[0]
                except ValueError:
                    print(f"Skipping unseen word: {q['correct']}")
                    continue

                features = pd.DataFrame([{
                    "level_encoded": level_encoded,
                    "correct_encoded": correct_encoded,
                    "selected_encoded": 0,
                    "time_taken": 3.0
                }])
                score_estimate = self.predictor.predict_proba(features)[0][0]
                challenge_pool.append((q, score_estimate))

        # Sort by predicted difficulty and take top 5
        challenge_pool.sort(key=lambda x: x[1], reverse=True)
        self.challenge_questions = [q for q, _ in challenge_pool[:5]]
        self.challenge_index = 0

        # Start challenge after short delay
        Clock.schedule_once(lambda dt: self.load_challenge_question(), 2)

    def replay_word(self):
        """
        Replays the current word’s pronunciation using Amazon Polly.
        """
        speak_word_threaded(self.correct_answer)
    
class ResultsScreen(Screen):
    """
    This screen displays the user's final quiz performance after completing both
    the main and challenge rounds. It shows scores, missed words, and logs results for tracking.
    """

    def display_results(self, main_score, challenge_score, incorrect_words):
        """
        Displays the user's results visually and logs them to text files.

        Args:
            main_score (int): Number of correct answers in the main round (out of 20)
            challenge_score (int): Number of correct answers in the challenge round (out of 5)
            incorrect_words (list): List of words the user answered incorrectly
        """
        total_score = main_score + challenge_score

        # Display formatted score summary
        self.ids.score_label.text = (
            f"Main Score: {main_score}/20\n"
            f"Challenge Bonus: {challenge_score}/5\n"
            f"Total: {total_score}/25"
        )

        # === Clear any previous results shown on the screen ===
        self.ids.feedback_box.clear_widgets()

        # === Show missed words if any ===
        if incorrect_words:
            title = Label(
                text="Words you missed:",
                font_name="assets/ProximaSoft-Bold",
                font_size="18sp",
                color=(0.7, 0, 0, 1),  # Dark red for emphasis
                size_hint_y=None,
                text_size=(Window.width - dp(40), None),
                halign="left",
                valign="top"
            )
            title.bind(texture_size=lambda instance, value: setattr(instance, 'height', value[1] + dp(10)))
            self.ids.feedback_box.add_widget(title)

            # List each incorrect word as a bullet point
            for word in incorrect_words:
                item = Label(
                    text=f"• {word}",
                    font_name="assets/ProximaSoft-Regular",
                    font_size="16sp",
                    color=(0, 0, 0, 1),
                    size_hint_y=None,
                    text_size=(Window.width - dp(40), None),
                    halign="left",
                    valign="top"
                )
                item.bind(texture_size=lambda instance, value: setattr(instance, 'height', value[1] + dp(10)))
                self.ids.feedback_box.add_widget(item)
        else:
            # Show success message if all answers were correct
            success = Label(
                text="You got everything correct!",
                font_name="assets/ProximaSoft-Bold",
                font_size="18sp",
                color=(0, 0.6, 0, 1),  # Green
                size_hint_y=None,
                text_size=(Window.width - dp(40), None),
                halign="center",
                valign="middle"
            )
            success.bind(texture_size=lambda instance, value: setattr(instance, 'height', value[1] + dp(10)))
            self.ids.feedback_box.add_widget(success)

        # === Logging Results to File ===
        LOGS_DIR = BASE_LOG_DIR
        LOGS_DIR.mkdir(parents=True, exist_ok=True)

        # Log the total score to 'score_log.txt'
        score_log_path = LOGS_DIR / "score_log.txt"
        with open(score_log_path, "a", encoding="utf-8") as score_file:
            score_file.write(
                f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - "
                f"Main: {main_score}/20, Challenge: {challenge_score}/5\n"
            )

        # Log mistakes to 'mistakes_log.txt' (if any)
        if incorrect_words:
            mistakes_log_path = LOGS_DIR / "mistakes_log.txt"
            with open(mistakes_log_path, "a", encoding="utf-8") as mistakes_file:
                mistakes_file.write(
                    f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - "
                    f"Missed Words: {', '.join(incorrect_words)}\n"
                )

class GrammarExerciseScreen(Screen):
    """
    Screen that runs the grammar-based sentence-building exercises in Fluentique.

    Users are shown an English sentence and must construct the correct French translation
    by selecting words in the right order from shuffled tiles. After submission, feedback is provided,
    and the user is later guided to speech practice using the same sentence.
    """

    # State tracking variables
    current_prompt = ""                     # The English sentence prompt
    current_correct_sentence = ""          # Correct French sentence from the server
    selected_words = []                    # Words the user selected in the current attempt

    # Progress counters
    progress = NumericProperty(0)
    sentence_progress = NumericProperty(0)  # How many grammar sentences completed (max 5)
    speech_progress = NumericProperty(0)    # How many speech practices completed (max 5)
    audio_practice_done = False             # Whether speech for current sentence was completed

    # Context and feedback preservation
    last_results_data = None
    last_prompt = ""
    last_user_sentence = ""
    last_expected_sentence = ""

    def on_enter(self):
        """
        Called automatically when the screen is shown.
        Begins the first sentence-building exercise unless waiting for feedback.
        """
        if not hasattr(self, 'sentence_progress'):
            self.sentence_progress = 0
        if not hasattr(self, 'speech_progress'):
            self.speech_progress = 0
        self.start_exercise()

    def start_exercise(self):
        """
        Sends a request to the backend to fetch a new grammar sentence.
        Renders shuffled word tiles for the user to build the correct sentence.
        """
        if self.audio_practice_done:
            print("Waiting for user to finish speech feedback — skipping start_exercise.")
            return

        if self.check_completion():
            return

        app = App.get_running_app()
        headers = {"Authorization": f"Bearer {app.token}"}
        response = requests.post(f"{API_BASE_URL}/grammar/exercise/start", headers=headers, timeout=REQUEST_TIMEOUT)

        if response.status_code == 200:
            data = response.json()
            self.current_prompt = data.get("english", "")
            self.current_correct_sentence = data.get("french", "")
            self.ids.prompt_label.text = self.current_prompt
            self.selected_words = []

            # Create and display word buttons in shuffled order
            words = self.current_correct_sentence.split()
            shuffle(words)
            self.ids.word_tiles.clear_widgets()

            for word in words:
                btn = Button(text=word, font_name="assets/ProximaSoft-Regular", size_hint_y=None, height=dp(40))
                btn.bind(on_release=self.select_word)
                self.ids.word_tiles.add_widget(btn)

            self.ids.constructed_sentence.text = ""
            self.ids['submit'].disabled = True
        else:
            self.show_popup("Error", "Failed to start exercise.")

    def select_word(self, instance):
        """
        Handles when a user taps on a word tile.
        Adds the word to the constructed sentence and disables the tile.
        """
        word = instance.text
        self.selected_words.append(word)
        instance.disabled = True
        self.ids.constructed_sentence.text = " ".join(self.selected_words)

        # Enable the Submit button once the full sentence is assembled
        if len(self.selected_words) == len(self.current_correct_sentence.split()):
            self.ids['submit'].disabled = False

        self.submitted = False

    def start_audio_practice(self):
        """
        Launches the speech practice screen using the last submitted sentence.
        Avoids repeating if already completed or queued.
        """
        if self.speech_progress >= 5:
            print("Speech exercises complete — skipping audio practice.")
            return

        if self.audio_practice_done:
            print("Audio already done for this sentence — skipping.")
            return

        app = App.get_running_app()
        speech_screen = app.root.get_screen('speech_practice')

        # Use the most recently submitted correct sentence
        speech_screen.set_target_sentence(self.last_expected_sentence)
        app.root.current = 'speech_practice'

    def submit_translation(self):
        """
        Submits the user's constructed French sentence to the backend for evaluation.
        Also handles logic for freezing state, validation, and feedback routing.
        """
        if getattr(self, 'submitted', False):
            print("Already submitted — skipping duplicate submission.")
            return
        self.submitted = True

        # Prevent submitting sentence before completing speech for previous
        if self.sentence_progress > self.speech_progress:
            print("Waiting for speech feedback — skipping sentence submission.")
            return

        if self.sentence_progress >= 5:
            print("Sentence exercises complete — skipping submission.")
            return

        # Build final user sentence from selected tiles
        user_input = " ".join(self.selected_words).replace(" .", ".").replace(" ,", ",").replace(" !", "!").replace(" ?", "?")
        print(f"SUBMITTING SENTENCE: '{user_input}'")

        # Save current sentence context for later speech feedback
        self.last_prompt = self.current_prompt
        self.last_user_sentence = user_input
        self.last_expected_sentence = self.current_correct_sentence
        self.last_speech_target_sentence = self.current_correct_sentence

        # Send to backend for marking
        self.ids['submit'].disabled = True
        app = App.get_running_app()
        headers = {"Authorization": f"Bearer {app.token}"}
        response = requests.post(
            f"{API_BASE_URL}/grammar/exercise/submit",
            headers=headers,
            json={"user_translation": user_input},
            timeout=REQUEST_TIMEOUT
        )

        if response.status_code == 200:
            self.audio_practice_done = False
            data = response.json()
            self.last_results_data = data
            self.process_feedback(data)

            # Switch to feedback screen
            Clock.schedule_once(lambda dt: setattr(app.root, 'current', 'sentence_result'), 0.5)
        else:
            self.show_popup("Error", "Submission failed.")

    def check_completion(self):
        """
        Checks if both sentence-building and speech practice are finished.
        If so, shows the final results screen.
        """
        print(f"Checking completion: Sentences {self.sentence_progress}, Speech {self.speech_progress}")
        if self.last_results_data is None:
            return False

        if self.sentence_progress >= 5 and self.speech_progress >= 5:
            app = App.get_running_app()
            results_screen = app.root.get_screen('grammar_results')
            results = self.last_results_data.get("results", [])
            results_screen.display_results(results, app.speech_feedback_collected)
            app.root.current = 'grammar_results'
            return True
        return False

    def process_feedback(self, data):
        """
        Processes the feedback response after sentence submission.
        Updates the sentence result screen and awards XP for correct answers.
        """
        app = App.get_running_app()
        all_responses = data.get("responses") or data.get("results")

        if all_responses:
            latest = None
            for res in reversed(all_responses):
                if res.get("expected") == self.last_expected_sentence:
                    latest = res
                    break

            last_response = latest or all_responses[-1]

            # Ensure all required values are available
            last_response.setdefault("prompt", self.last_prompt)
            last_response.setdefault("user", self.last_user_sentence)
            last_response.setdefault("expected", self.last_expected_sentence)

            # Display feedback on the result screen
            sentence_result_screen = app.root.get_screen('sentence_result')
            correct = last_response.get("correct", False)

            if correct:
                sentence_result_screen.ids.result_label.text = "Correct!"
                sentence_result_screen.ids.result_label.color = (0, 0.6, 0, 1)
                sentence_result_screen.bg_color = (0.85, 1, 0.85, 1)
            else:
                sentence_result_screen.ids.result_label.text = "Incorrect!"
                sentence_result_screen.ids.result_label.color = (0.9, 0, 0, 1)
                sentence_result_screen.bg_color = (1, 0.85, 0.85, 1)
                sentence_result_screen.ids.user_sentence_label.text = f"Your answer: {last_response.get('user', '')}"

            sentence_result_screen.ids.correct_sentence_label.text = last_response["expected"]
            self.sentence_progress += 1
            print(f"Sentence progress incremented to: {self.sentence_progress}")

            if correct:
                progress_manager.add_xp(10)

        else:
            print("No valid responses found in feedback.")

    def show_popup(self, title, message):
        """
        Simple popup used for error reporting
        """
        popup = Popup(title=title, content=Label(text=message), size_hint=(None, None), size=(300, 200))
        popup.open()

class GrammarResultsScreen(Screen):
    """
    Displays a comprehensive summary of both the sentence-building
    and speech practice exercises after all tasks are completed.

    It categorises correct/incorrect sentence results and levels of speech feedback,
    showing this visually with colour-coded sections and organised widgets.
    """

    def return_home(self):
        """
        Navigates back to the Home screen after results are viewed.
        Refreshes XP and streak data to reflect recent progress.
        """
        app = App.get_running_app()
        app.root.current = 'home_incomplete'
        home_screen = app.root.get_screen('home_incomplete')
        home_screen.on_pre_enter()

    def display_results(self, sentence_results, speech_feedback):
        """
        Main method that renders the results for both sentence and speech exercises.

        Args:
            sentence_results (list): List of dictionaries with sentence correctness and feedback
            speech_feedback (list): List of speech evaluations including transcriptions and target phrases
        """
        self.ids.sentence_results_box.clear_widgets()
        self.ids.speech_results_box.clear_widgets()

        # === Sentence Puzzle Section ===
        self.ids.sentence_results_box.add_widget(Label(
            text="[b]Sentence Puzzle Results[/b]",
            markup=True,
            font_name="assets/ProximaSoft-Bold",
            color=(0, 0, 0, 1),
            size_hint_y=None,
            height=dp(50),
            halign='center'
        ))

        # Score summary
        correct_count = sum(1 for res in sentence_results if res.get("correct"))
        self.ids.sentence_results_box.add_widget(Label(
            text=f"You got {correct_count}/5 correct in the sentence puzzle!",
            font_name="assets/ProximaSoft-Regular",
            color=(0, 0, 0, 1),
            size_hint_y=None,
            height=dp(30),
            halign='center'
        ))

        # Split results into correct and incorrect groups
        correct_group = [res for res in sentence_results if res.get("correct")]
        incorrect_group = [res for res in sentence_results if not res.get("correct")]

        # Render both groups with colour-coded labels
        self.add_sentence_group("Correct Answers", correct_group, (76/255, 175/255, 80/255, 1))  # Green
        self.add_sentence_group("Incorrect Answers", incorrect_group, (244/255, 67/255, 54/255, 1))  # Red

        # === Speech Feedback Section ===
        self.ids.speech_results_box.add_widget(Label(
            text="[b]Speech Practice Feedback[/b]",
            markup=True,
            font_name="assets/ProximaSoft-Bold",
            color=(0, 0, 0, 1),
            size_hint_y=None,
            height=dp(50),
            halign='center'
        ))

        # Categorise speech transcription results
        right_list, improve_list, wrong_list = [], [], []

        for entry in speech_feedback:
            user_transcript = entry["transcription"]
            target_sentence = entry["target"]
            feedback = entry["feedback"]

            category = self.categorize_transcription(user_transcript, target_sentence)

            if category == 'Right':
                right_list.append((target_sentence, user_transcript, feedback))
            elif category == 'Improve Slightly':
                improve_list.append((target_sentence, user_transcript, feedback))
            else:
                wrong_list.append((target_sentence, user_transcript, feedback))

        # Display each speech category with colour and grouping
        self.add_speech_group("Right", right_list, (76/255, 175/255, 80/255, 1))  # Green
        self.add_speech_group("Improve Slightly", improve_list, (255/255, 152/255, 0, 1))  # Orange
        self.add_speech_group("Wrong", wrong_list, (244/255, 67/255, 54/255, 1))  # Red

    def categorize_transcription(self, user_transcript, target_sentence):
        """
        Compares the user's spoken sentence against the expected sentence.
        Uses a similarity ratio to group as Right, Improve Slightly, or Wrong.

        Returns:
            str: One of 'Right', 'Improve Slightly', or 'Wrong'
        """
        user_clean = user_transcript.strip().lower()
        target_clean = target_sentence.strip().lower()

        if user_clean == target_clean:
            return 'Right'
        else:
            from difflib import SequenceMatcher
            similarity = SequenceMatcher(None, user_clean, target_clean).ratio()
            if similarity >= 0.8:
                return 'Improve Slightly'
            else:
                return 'Wrong'

    def add_sentence_group(self, title, group, color):
        """
        Helper to display a sentence result group (correct/incorrect).

        Args:
            title (str): Group title
            group (list): List of sentence result dicts
            color (tuple): RGBA colour for heading and feedback
        """
        if not group:
            return

        self.ids.sentence_results_box.add_widget(Label(
            text=f"[b]{title}[/b]",
            markup=True,
            font_name="assets/ProximaSoft-Bold",
            color=color,
            size_hint_y=None,
            height=dp(40),
            halign='left',
            valign='middle'
        ))

        for res in group:
            prompt = res.get("prompt")
            user = res.get("user")
            expected = res.get("expected")
            feedback = res.get("feedback", "")

            box = BoxLayout(orientation='vertical', padding=(dp(15), dp(10)), spacing=dp(8),
                            size_hint_x=1, size_hint_y=None)
            box.bind(minimum_height=box.setter('height'))

            # Prompt (English)
            box.add_widget(Label(
                text=f"[b]Prompt:[/b] {prompt}",
                markup=True,
                font_name="assets/ProximaSoft-Bold",
                halign='left',
                valign='top',
                size_hint_y=None,
                text_size=(self.width, None),
                color=(0, 0, 0, 1)
            ))

            # User's submitted translation
            box.add_widget(Label(
                text=f"[b]Your answer:[/b] {user}",
                markup=True,
                font_name="assets/ProximaSoft-Regular",
                halign='left',
                valign='top',
                size_hint_y=None,
                text_size=(self.width, None),
                color=(0, 0, 0, 1)
            ))

            # Show expected + feedback only if incorrect
            if not res.get("correct"):
                box.add_widget(Label(
                    text=f"[b]Expected:[/b] {expected}",
                    markup=True,
                    font_name="assets/ProximaSoft-Regular",
                    halign='left',
                    valign='top',
                    size_hint_y=None,
                    text_size=(self.width, None),
                    color=(0, 0, 0, 1)
                ))
                box.add_widget(Label(
                    text=f"[b]Feedback:[/b] {feedback}",
                    markup=True,
                    font_name="assets/ProximaSoft-Regular",
                    halign='left',
                    valign='top',
                    size_hint_y=None,
                    text_size=(self.width, None),
                    color=color
                ))

            self.ids.sentence_results_box.add_widget(box)

    def add_speech_group(self, title, group, color):
        """
        Helper to display a speech feedback group (Right, Improve Slightly, Wrong).

        Args:
            title (str): Category title
            group (list): List of (target, transcript, feedback) tuples
            color (tuple): RGBA colour used for headings and feedback
        """
        if not group:
            return

        self.ids.speech_results_box.add_widget(Label(
            text=f"[b]{title}[/b]",
            markup=True,
            font_name="assets/ProximaSoft-Bold",
            color=color,
            size_hint_y=None,
            height=dp(40),
            halign='left',
            valign='middle'
        ))

        for target, transcript, feedback in group:
            box = BoxLayout(orientation='vertical', padding=(dp(15), dp(10)), spacing=dp(8),
                            size_hint_x=1, size_hint_y=None)
            box.bind(minimum_height=box.setter('height'))

            box.add_widget(Label(
                text=f"[b]Target:[/b] {target}",
                markup=True,
                font_name="assets/ProximaSoft-Regular",
                halign='left',
                valign='top',
                size_hint_y=None,
                text_size=(self.width, None),
                color=(0, 0, 0, 1)
            ))

            box.add_widget(Label(
                text=f"[b]Your transcription:[/b] {transcript}",
                markup=True,
                font_name="assets/ProximaSoft-Regular",
                halign='left',
                valign='top',
                size_hint_y=None,
                text_size=(self.width, None),
                color=(0, 0, 0, 1)
            ))

            box.add_widget(Label(
                text=f"[b]Feedback:[/b] {feedback}",
                markup=True,
                font_name="assets/ProximaSoft-Regular",
                halign='left',
                valign='top',
                size_hint_y=None,
                text_size=(self.width, None),
                color=color
            ))

            self.ids.speech_results_box.add_widget(box)

class SentenceResultScreen(Screen):
    """
    This screen is shown immediately after the user submits a sentence puzzle.

    It gives instant feedback (Correct/Incorrect) with colour changes and text.
    The user can then continue to the speech practice part of the same sentence.
    """

    # Background colour of the screen, dynamically updated based on correctness
    bg_color = ListProperty([0.965, 0.969, 1, 1])  # Default light blue

    def continue_to_speech(self):
        """
        Called when the user taps 'Continue'.
        This triggers the speech practice screen for the current sentence.
        """
        app = App.get_running_app()

        # Get reference to the grammar exercise screen to carry context
        grammar_screen = app.root.get_screen('grammar_exercise')

        # Start audio practice using the most recent sentence
        grammar_screen.start_audio_practice()

        # Navigate to the speech practice screen
        app.root.current = 'speech_practice'
        
class SpeechPracticeScreen(Screen):
    """
    This screen handles the speech practice component of the exercise.
    The user is asked to say a sentence aloud, which is then evaluated
    by the backend for transcription accuracy and pronunciation quality.
    """

    target_sentence = StringProperty("")  # Sentence the user needs to pronounce

    def on_enter(self):
        """
        Triggered when the screen becomes visible.
        Prepares UI updates, initialises file path for recording, and updates the progress bar.
        """
        print(f"[DEBUG] on_enter — current target_sentence: {self.target_sentence}")
        self.recorded_audio_path = "recorded_audio.wav"

        # Refresh progress bar after short delay
        Clock.schedule_once(self.update_progress, 0.1)

        # Ensure sentence label is updated after UI loads
        Clock.schedule_once(lambda dt: setattr(self.ids.speech_sentence_label, 'text', self.target_sentence), 0.2)

    def update_progress(self, *args):
        """
        Updates the visual progress bar based on how many sentence and speech rounds are done.
        """
        grammar_screen = App.get_running_app().root.get_screen('grammar_exercise')
        progress = grammar_screen.sentence_progress + grammar_screen.speech_progress
        self.ids.progress_bar.value = progress

    def set_target_sentence(self, sentence):
        """
        Sets the sentence the user must read aloud.

        Args:
            sentence (str): The target French sentence for speech practice.
        """
        print(f"[DEBUG] set_target_sentence called with: {sentence}")
        self.target_sentence = sentence

        if hasattr(self.ids, 'speech_sentence_label'):
            self.ids.speech_sentence_label.text = sentence

    def start_recording(self):
        """
        Begins a 3-second audio recording of the user's voice.
        Saves it to a temporary .wav file.
        """
        print("Recording started...")
        self.record_audio('recorded_audio.wav')

    def record_audio(self, filename, duration=3, samplerate=16000):
        """
        Records audio from the microphone and saves it to a file.

        Args:
            filename (str): File name for saving the recording.
            duration (int): Length of the recording in seconds.
            samplerate (int): Audio sample rate (16kHz recommended for STT models).
        """
        print("Recording...")
        recording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1)
        sd.wait()
        sf.write(filename, recording, samplerate)
        print("Recording saved.")
        self.send_audio_to_backend(filename)

    def send_audio_to_backend(self, filepath):
        """
        Sends the recorded audio file to the backend for speech recognition and feedback.

        Args:
            filepath (str): Path to the recorded audio file.
        """
        app = App.get_running_app()
        headers = {"Authorization": f"Bearer {app.token}"}
        target_sentence = self.target_sentence

        data = {'target_sentence': target_sentence}
        print(f"Target sentence sent: {data['target_sentence']}")

        with open(filepath, "rb") as audio_file:
            files = {"file": audio_file}

            response = requests.post(
                f"{API_BASE_URL}/chat/chat_loop",
                headers=headers,
                files=files,
                data=data,
                timeout=REQUEST_TIMEOUT,
            )

        if response.status_code == 200:
            data = response.json()
            self.process_backend_response(data)
        else:
            print("Error submitting audio.", response.status_code)

    def process_backend_response(self, data):
        """
        Processes the backend's feedback after evaluating the user's speech.

        Args:
            data (dict): JSON response containing transcription, feedback text, and optional audio.
        """
        transcription = data.get("transcription", "No transcription")
        feedback = data.get("response", "No feedback")
        audio_file = data.get("audio_file", None)

        print(f"Speech feedback received: {feedback}")

        # Display the results on the speech feedback screen
        feedback_screen = App.get_running_app().root.get_screen('speech_feedback')
        feedback_screen.ids.transcription_label.text = f"Transcription: {transcription}"
        feedback_screen.ids.speech_feedback_label.text = feedback
        App.get_running_app().root.current = 'speech_feedback'

        # Append feedback to list for end-of-exercise summary
        grammar_screen = App.get_running_app().root.get_screen('grammar_exercise')
        if not hasattr(grammar_screen, 'speech_feedback_entries'):
            grammar_screen.speech_feedback_entries = []

        grammar_screen.speech_feedback_entries.append({
            "target": grammar_screen.current_correct_sentence,
            "transcription": transcription,
            "feedback": feedback
        })

        print(f"Collected speech feedback so far: {grammar_screen.speech_feedback_entries}")

        # Mark this speech practice as done to unlock the next sentence
        grammar_screen.audio_practice_done = True
        print("audio_practice_done set to True")

        # Download audio feedback if available and play it
        if audio_file:
            audio_url = f"{API_BASE_URL}/chat/audio/{audio_file}"
            local_audio_path = os.path.join(tempfile.gettempdir(), audio_file)

            try:
                r = requests.get(audio_url, timeout=REQUEST_TIMEOUT)
                with open(local_audio_path, 'wb') as f:
                    f.write(r.content)

                from kivy.core.audio import SoundLoader
                sound = SoundLoader.load(local_audio_path)
                if sound:
                    feedback_screen.current_sound = sound
                    sound.play()
            except Exception as e:
                print(f"Error downloading/playing audio: {e}")

class SpeechFeedbackScreen(Screen):
    """
    This screen displays the result of the user's spoken sentence after backend evaluation.
    It shows the transcription and feedback, plays audio feedback, and allows the user to continue.
    """

    current_sound = None  # Stores the currently playing audio feedback, if any

    def continue_to_next(self):
        """
        Triggered when the user taps 'Continue'.
        Handles cleanup of any playing audio, updates XP and speech progress,
        and navigates to the next appropriate screen based on user progress.
        """
        # Stop audio playback if still playing
        if self.current_sound and self.current_sound.state == 'play':
            self.current_sound.stop()
            self.current_sound = None

        app = App.get_running_app()
        grammar_screen = app.root.get_screen('grammar_exercise')

        print("SpeechFeedbackScreen: continue_to_next called")
        print(f"continue_to_next triggered — audio_practice_done: {grammar_screen.audio_practice_done}")

        if grammar_screen.audio_practice_done:
            # Mark current speech as complete and increment progress
            grammar_screen.audio_practice_done = False
            grammar_screen.speech_progress += 1
            Clock.schedule_once(lambda dt: setattr(grammar_screen, 'speech_progress', grammar_screen.speech_progress))
            print(f"Speech progress incremented to: {grammar_screen.speech_progress}")

            # Award XP for completing speech practice
            progress_manager.add_xp(5)

            # If both sentence and speech rounds are done, go to final summary
            if grammar_screen.sentence_progress >= 5 and grammar_screen.speech_progress >= 5:
                app.root.current = 'final_speech_feedback'
            else:
                # Go back to grammar screen and load the next sentence
                app.root.current = 'grammar_exercise'
                Clock.schedule_once(lambda dt: grammar_screen.start_exercise(), 0.5)
        else:
            print("Skipping increment — waiting for audio practice to complete.")

        # Update the user's daily streak regardless of completion
        progress_manager.update_streak()

class FinalSpeechFeedbackScreen(Screen):
    """
    This screen acts as a checkpoint before the final grammar results are shown.
    It confirms that both sentence and speech exercises are done,
    and gathers all result data to be passed into the results summary screen.
    """

    def on_enter(self):
        """
        Resets local tracking variables whenever this screen is entered.
        This helps ensure a clean state before viewing the final results.
        """
        self.last_results_data = None
        self.sentence_progress = 0
        self.speech_progress = 0
        self.audio_practice_done = False
        self.selected_words = []

    def go_to_results(self):
        """
        Triggered when the user taps the 'View Results' button.
        Collects the grammar and speech feedback from the previous exercises,
        saves them to local logs, and passes them to the grammar results screen.
        """
        app = App.get_running_app()
        grammar_screen = app.root.get_screen('grammar_exercise')
        results_screen = app.root.get_screen('grammar_results')

        # Safety check in case results are missing (e.g. direct screen access)
        if not grammar_screen.last_results_data:
            print("No last_results_data available — cannot show results.")
            self.show_popup("Error", "No results available. Please complete the exercises first.")
            return

        sentence_results = grammar_screen.last_results_data.get("results", [])
        speech_feedback = getattr(grammar_screen, 'speech_feedback_entries', [])

        print(f"Passing sentence results: {sentence_results}")
        print(f"Passing speech feedback: {speech_feedback}")

        # Save session logs before showing results
        self.save_to_logs(sentence_results, speech_feedback)

        # Display final feedback in the results screen
        results_screen.display_results(sentence_results, speech_feedback)
        app.root.current = 'grammar_results'

    def save_to_logs(self, sentence_results, speech_feedback):
        """
        Saves the current grammar and speech session results to timestamped logs.

        Args:
            sentence_results (list): Results of sentence-building exercises.
            speech_feedback (list): Feedback entries from speech practice.
        """
        log_dir = Path(__file__).resolve().parent.parent / "backend" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        # === Sentence Log ===
        sentence_log = log_dir / "sentence_exercise_log.txt"
        with open(sentence_log, "a", encoding="utf-8") as f:
            f.write(f"\n🗓️ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} — Sentence Exercise\n")
            for result in sentence_results:
                target = result.get("expected", "")
                user_input = result.get("user", "")
                is_correct = result.get("correct", False)

                f.write(f"Target: {target}\n")
                f.write(f"User Input: {user_input}\n")
                f.write(f"{'Correct' if is_correct else 'Incorrect'}\n\n")

        # === Speech Log ===
        speech_log = log_dir / "speech_feedback_log.txt"
        with open(speech_log, "a", encoding="utf-8") as f:
            f.write(f"\n{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} — Speech Feedback\n")
            for entry in speech_feedback:
                sentence = entry.get("target", "")
                transcription = entry.get("transcription", "")
                feedback = entry.get("feedback", "")
                f.write(f"Sentence: {sentence}\n")
                f.write(f"User Said: {transcription}\n")
                f.write(f"Feedback: {feedback}\n\n")

class ChatbotTutorScreen(Screen):
    """
    This screen manages the AI-powered tutor chatbot. Users can either type or speak in French,
    and the AI will respond with feedback, guidance, or conversation.
    """

    is_voice_mode = False  # Whether the user is speaking or typing (default: typing)
    has_started = False    # Ensures the bot sends its greeting only once

    def on_enter(self):
        """
        Called when the chatbot screen is first shown.
        Automatically triggers the tutor's greeting message.
        """
        if not self.has_started:
            self.has_started = True
            Clock.schedule_once(lambda dt: self.send_first_message(), 0.5)

    def send_first_message(self):
        """
        Sends an initial message to the backend to receive a welcome/greeting from the AI tutor.
        """
        app = App.get_running_app()
        headers = {"Authorization": f"Bearer {app.token}"}
        payload = {"message": "", "is_first": True}  # Empty message to trigger greeting

        self.set_avatar("thinking")

        def fetch_first(dt):
            try:
                response = requests.post(f"{API_BASE_URL}/chat/tutor", json=payload, headers=headers, timeout=REQUEST_TIMEOUT)
                if response.status_code == 200:
                    data = response.json()
                    reply = data.get("reply", "").strip()
                    audio_file = data.get("audio_file", "")
                    self.add_message(reply, align="left", is_ai=True)
                    self.play_avatar_audio(audio_file)
            except Exception as e:
                self.set_avatar("idle")
                self.add_message(f"{e}", align="left")

        Clock.schedule_once(fetch_first, 0)

    def send_message(self):
        """
        Sends the user's typed message to the backend and displays both the question and AI reply.
        Triggers speech input if the AI expects the user to repeat.
        """
        app = App.get_running_app()
        if not app.token:
            self.add_message("Connecte-toi d'abord.", align="left")
            return

        user_input = self.ids.input_box.text.strip()
        if not user_input:
            return

        user_input = user_input[:1].upper() + user_input[1:]
        self.add_message(user_input, align="right")
        self.ids.input_box.text = ""

        headers = {"Authorization": f"Bearer {app.token}"}
        payload = {"message": user_input}
        self.set_avatar("thinking")

        def fetch_reply(dt):
            try:
                response = requests.post(f"{API_BASE_URL}/chat/tutor", json=payload, headers=headers, timeout=REQUEST_TIMEOUT)
                if response.status_code == 200:
                    data = response.json()
                    reply = data.get("reply", "").strip()
                    audio_file = data.get("audio_file", "")
                    self.add_message(reply, align="left", is_ai=True)

                    # If the bot expects a reply from user (e.g. a speaking task)
                    if "À toi" in reply or "Répète" in reply:
                        Clock.schedule_once(lambda dt: self.record_voice_input(), 1.0)

                    self.play_avatar_audio(audio_file)
                else:
                    self.set_avatar("idle")
                    self.add_message("Error AI", align="left")
            except Exception as e:
                self.set_avatar("idle")
                self.add_message(f"{e}", align="left")

        Clock.schedule_once(fetch_reply, 0)

    def set_avatar(self, state):
        """
        Changes the avatar based on AI's state (idle, thinking, or talking).
        """
        sources = {
            "idle": "assets/avatar_idle.png",
            "thinking": "assets/typing_dots.gif",
            "talking": "assets/avatar_talking.gif"
        }
        self.ids.avatar.source = sources.get(state, sources["idle"])

    def play_avatar_audio(self, audio_file):
        """
        Plays the avatar's spoken response using the returned audio file.
        """
        self.set_avatar("talking")
        audio_url = f"{API_BASE_URL}/chat/audio/{audio_file}"

        try:
            response = requests.get(audio_url, timeout=REQUEST_TIMEOUT)
            if response.status_code == 200:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
                    tmp.write(response.content)
                    tmp_path = tmp.name

                sound = SoundLoader.load(tmp_path)
                if sound:
                    def finish_playback(dt):
                        sound.stop()
                        self.set_avatar("idle")

                    sound.play()
                    duration = sound.length if sound.length and sound.length > 0 else 2.5
                    Clock.schedule_once(finish_playback, duration + 0.3)
                else:
                    self.set_avatar("idle")
        except Exception as e:
            print("Exception in playing avatar audio:", e)
            self.set_avatar("idle")

    def add_message(self, text, align="left", is_ai=False):
        """
        Adds a chat bubble to the UI with optional translation button for AI replies.
        """
        from kivy.uix.button import Button
        from kivy.uix.boxlayout import BoxLayout

        wrapper = BoxLayout(orientation='vertical', size_hint_y=None, spacing=5)
        wrapper.bind(minimum_height=wrapper.setter('height'))

        msg = Label(
            text=text,
            size_hint_y=None,
            halign=align,
            valign="top",
            text_size=(self.ids.chat_box.width - dp(20), None),
            font_size="16sp",
            font_name="assets/ProximaSoft-Regular",
            color=(0, 0, 0, 1)
        )
        msg.bind(texture_size=lambda inst, val: setattr(msg, 'height', val[1] + dp(10)))
        wrapper.add_widget(msg)

        # Add 'Translate to English' button under AI messages
        if is_ai:
            def do_translate(btn):
                if btn.text.startswith("Translate"):
                    self.translate_message(msg.text, wrapper)
                    btn.text = "Translated"
                    btn.disabled = True

            btn = Button(
                text="Translate to English",
                size_hint_y=None,
                height=dp(32),
                font_size="14sp",
                background_color=(0.2, 0.2, 0.2, 1),
                color=(1, 1, 1, 1),
                on_release=do_translate
            )
            wrapper.add_widget(btn)

        self.ids.chat_box.add_widget(wrapper)
        Clock.schedule_once(lambda dt: setattr(self.ids.scroll, 'scroll_y', 0), 0.1)

    def translate_message(self, french_text, container):
        """
        Requests an English translation of a French message from the backend.
        """
        app = App.get_running_app()
        headers = {"Authorization": f"Bearer {app.token}"}
        try:
            response = requests.post(f"{API_BASE_URL}/chat/translate", json={"text": french_text}, headers=headers, timeout=REQUEST_TIMEOUT)
            if response.status_code == 200:
                english = response.json().get("translation", "")
                translated = Label(
                    text=f"🇬🇧 {english}",
                    font_name="assets/ProximaSoft-Regular",
                    font_size="15sp",
                    color=(0.2, 0.2, 0.2, 1),
                    size_hint_y=None,
                    text_size=(self.ids.chat_box.width - dp(20), None)
                )
                translated.bind(texture_size=lambda inst, val: setattr(translated, 'height', val[1] + dp(5)))
                container.add_widget(translated)
        except Exception as e:
            print("Translation error:", e)

    def toggle_input_mode(self):
        """
        Switches between typing and voice input modes.
        """
        self.is_voice_mode = not self.is_voice_mode
        if self.is_voice_mode:
            self.ids.input_box.disabled = True
            self.ids.input_box.hint_text = "Speak Now..."
            self.record_voice_input()
        else:
            self.ids.input_box.disabled = False
            self.ids.input_box.hint_text = "Type your message in French..."

    def record_voice_input(self):
        """
        Records a short voice clip (5 seconds) from the user.
        """
        self.ids.recording_label.text = "Speak now..."
        samplerate = 16000
        duration = 5
        audio = sd.rec(int(samplerate * duration), samplerate=samplerate, channels=1)
        sd.wait()
        self.ids.recording_label.text = ""

        temp_path = os.path.join(tempfile.gettempdir(), "temp_recording.wav")
        sf.write(temp_path, audio, samplerate)
        self.send_voice_to_backend(temp_path)

    def send_voice_to_backend(self, audio_path):
        """
        Converts voice input to text using Wav2Vec2 or Google STT.
        Then sends the transcription to the AI tutor for a response.
        """
        app = App.get_running_app()
        headers = {"Authorization": f"Bearer {app.token}"}
        transcription = ""

        try:
            # Primary method: use local Wav2Vec2 model
            model_dir = (
                PROJECT_ROOT
                / "backend"
                / "models"
                / "fluentique-wav2vec2-fr"
            )
            processor = Wav2Vec2Processor.from_pretrained(str(model_dir))
            model = Wav2Vec2ForCTC.from_pretrained(str(model_dir))

            speech_array, _ = librosa.load(audio_path, sr=16000)
            inputs = processor(speech_array, sampling_rate=16000, return_tensors="pt", padding=True)

            with torch.no_grad():
                logits = model(**inputs).logits
            pred_ids = torch.argmax(logits, dim=-1)
            transcription = processor.batch_decode(pred_ids)[0].lower().strip()

            if len(transcription) < 2:
                raise ValueError("Model working - Wav2Vec2")

        except Exception:
            # Fallback: use Google Speech API
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GOOGLE_CREDS_PATH
            client_google = speech.SpeechClient()

            with io.open(audio_path, "rb") as audio_file:
                content = audio_file.read()

            audio = speech.RecognitionAudio(content=content)
            config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=16000,
                language_code="fr-FR",
            )
            response = client_google.recognize(config=config, audio=audio)
            transcription = response.results[0].alternatives[0].transcript if response.results else ""

        # Format the transcription before sending to AI
        transcription = transcription.strip()
        if transcription and not transcription.endswith("."):
            transcription += "."
        if transcription:
            transcription = transcription[0].upper() + transcription[1:]

        self.add_message(transcription, align="right")
        self.set_avatar("thinking")

        payload = {"message": transcription}
        response = requests.post(f"{API_BASE_URL}/chat/tutor", json=payload, headers=headers, timeout=REQUEST_TIMEOUT)

        if response.status_code == 200:
            data = response.json()
            self.add_message(data.get("reply", ""), align="left", is_ai=True)
            self.play_avatar_audio(data.get("audio_file", ""))
        else:
            self.set_avatar("idle")
            self.add_message("Voice error", align="left")

    def end_chat_session(self):
        """
        Ends the current chat session. Saves the conversation to a local log,
        and retrieves a session summary from the backend.
        """
        from datetime import datetime
        app = App.get_running_app()

        # Extract all user and AI messages from the chat UI
        transcript = []
        for child in self.ids.chat_box.children[::-1]:
            if hasattr(child, 'children'):
                for inner in child.children:
                    if hasattr(inner, 'text') and inner.text.strip():
                        timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M]")
                        direction = "AI" if "🇬🇧" not in inner.text else "User"
                        transcript.append(f"{timestamp} {direction}: {inner.text.strip()}")

        # Save chat transcript locally
        log_folder = BASE_LOG_DIR
        log_folder.mkdir(parents=True, exist_ok=True)
        log_filename = f"chat_transcript_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        log_path = log_folder / log_filename
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("\n".join(transcript))

        # Request summary of the chat session
        headers = {"Authorization": f"Bearer {app.token}"}
        try:
            response = requests.post(f"{API_BASE_URL}/chat/summary", headers=headers, timeout=REQUEST_TIMEOUT)
            if response.status_code == 200:
                app.summary_text = response.json().get("summary", "No summary returned.")
            else:
                app.summary_text = "Could not retrieve summary."
        except Exception as e:
            app.summary_text = f"Summary error: {str(e)}"

        # Go to the summary screen
        self.manager.current = "summary_screen"

class SummaryScreen(Screen):
    """
    This screen displays a summary of the AI tutoring session after the user ends the conversation.
    The summary is generated by the backend and provides a brief reflection or advice.
    """

    def on_enter(self):
        """
        When the screen is entered, it retrieves the stored summary text
        from the application and displays it in the label widget.
        """
        app = App.get_running_app()
        self.ids.summary_label.text = app.summary_text

    def return_home(self):
        """
        Navigates back to the home screen after reviewing the summary.
        """
        self.manager.current = "home_incomplete"

class MistakeLogMenuScreen(Screen):
    """
    This is the navigation menu for viewing different categories of logs,
    such as quiz mistakes, scores, grammar feedback, and AI transcripts.
    Each button leads to a dedicated log screen.
    """
    pass

class QuizMistakesLogsScreen(Screen):
    """
    Displays a visual and textual log of the user's most frequent quiz mistakes.
    It reads from the 'mistakes_log.txt' file and generates a bar chart of the top missed words.
    """

    def on_enter(self):
        """
        Triggered when the user navigates to this screen.
        It loads the mistake log, generates a bar chart of the most missed words,
        and displays both the chart and full log lines in the scrollable container.
        """
        from collections import Counter
        import matplotlib.pyplot as plt
        import tempfile
        from kivy.uix.image import Image

        # Clear previous content
        self.ids.quiz_mistakes_container.clear_widgets()
        log_path = BASE_LOG_DIR / "mistakes_log.txt"

        # Show fallback message if log file does not exist
        if not log_path.exists():
            self.ids.quiz_mistakes_container.add_widget(Label(
                text="No mistake logs found.",
                font_name="assets/ProximaSoft-Regular",
                color=(1, 1, 1, 1),
                size_hint_y=None,
                height=dp(30)
            ))
            return

        # Read all log lines
        with open(log_path, encoding="utf-8") as f:
            lines = f.readlines()

        # Extract missed words from the lines
        all_words = []
        for line in lines:
            if "Missed Words:" in line:
                parts = line.strip().split("Missed Words:")[-1]
                words = [w.strip() for w in parts.split(",") if w.strip()]
                all_words.extend(words)

        # If words are found, generate a bar chart
        if all_words:
            word_counts = Counter(all_words)
            top_words = word_counts.most_common(10)  # Get top 10 most missed
            words, counts = zip(*top_words)

            # Create the bar chart using matplotlib
            fig, ax = plt.subplots(figsize=(6, 3), dpi=100)
            ax.bar(words, counts, color="#37474F")
            ax.set_title("Most Missed Words", fontsize=10)
            ax.set_ylabel("Frequency", fontsize=9)
            ax.set_xticklabels(words, rotation=45, ha='right', fontsize=8)
            plt.tight_layout()

            # Save chart to a temporary file and display in Kivy
            temp_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
            plt.savefig(temp_path)
            plt.close()

            self.ids.quiz_mistakes_container.add_widget(Image(
                source=temp_path,
                size_hint_y=None,
                height=dp(240),
                allow_stretch=True
            ))

        # Display all raw log lines under the chart
        for line in lines:
            label = Label(
                text=line.strip(),
                font_name="assets/ProximaSoft-Regular",
                font_size="15sp",
                color=(1, 1, 1, 1),
                size_hint_y=None,
                text_size=(Window.width - dp(60), None),
                halign="left",
                valign="top",
                padding=(dp(10), dp(10))
            )
            label.bind(texture_size=lambda instance, value: setattr(label, 'height', value[1]))
            self.ids.quiz_mistakes_container.add_widget(label)

class QuizScoresLogsScreen(Screen):
    """
    This screen displays the user's quiz scores over time.
    A line chart visualises progress and consistency, while the full log
    of past quiz scores is also shown for transparency and review.
    """

    def on_enter(self):
        """
        Triggered when the screen becomes active.
        It loads and parses the quiz score log file, generates a score-over-time graph,
        and displays both the visual chart and raw log entries in the interface.
        """
        from collections import defaultdict
        import matplotlib.pyplot as plt
        import tempfile
        from kivy.uix.image import Image

        # Clear any previous widgets in the container
        self.ids.quiz_scores_container.clear_widgets()
        log_path = BASE_LOG_DIR / "score_log.txt"

        score_data = []  # Holds (timestamp, score_string) pairs
        lines = []

        # Load the log file if it exists
        if log_path.exists():
            with open(log_path, encoding="utf-8") as f:
                lines = f.readlines()

        # Parse lines and extract score entries with timestamps
        for line in lines:
            parts = line.strip().split(" - ")
            if len(parts) == 2:
                timestamp, result = parts
                if "Score:" in result or "Main:" in result:
                    score_data.append((timestamp.strip(), result.strip()))

        # Plot the line graph based on parsed score data
        if score_data:
            times = []
            scores = []
            for timestamp, result in score_data:
                try:
                    # Extract numeric score from either format
                    if "Score:" in result:
                        score_val = int(result.split("Score:")[1].split("/")[0])
                    elif "Main:" in result:
                        score_val = int(result.split("Main:")[1].split("/")[0])
                    else:
                        continue
                    times.append(timestamp[-5:])  # Keep only the HH:MM from timestamp
                    scores.append(score_val)
                except:
                    continue  # Skip lines with format issues

            if scores:
                # Generate the line chart using matplotlib
                fig, ax = plt.subplots(figsize=(6, 3), dpi=100)
                ax.plot(times, scores, marker='o', color="#5C6BC0", linestyle='-')
                ax.set_title("Quiz Scores Over Time", fontsize=10)
                ax.set_ylabel("Score", fontsize=9)
                ax.set_xlabel("Time", fontsize=9)
                plt.xticks(rotation=45)
                plt.tight_layout()

                # Save the chart image to a temporary file
                temp_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
                plt.savefig(temp_path)
                plt.close()

                # Display the chart image in the Kivy app
                self.ids.quiz_scores_container.add_widget(Image(
                    source=temp_path,
                    size_hint_y=None,
                    height=dp(240),
                    allow_stretch=True
                ))

        # Below the chart, display the full log for reference
        for line in lines:
            label = Label(
                text=line.strip(),
                font_name="assets/ProximaSoft-Regular",
                font_size="15sp",
                color=(1, 1, 1, 1),
                size_hint_y=None,
                text_size=(Window.width - dp(60), None),
                halign="left",
                valign="top",
                padding=(dp(10), dp(10))
            )
            label.bind(texture_size=lambda instance, value: setattr(label, 'height', value[1]))
            self.ids.quiz_scores_container.add_widget(label)

class SentenceResultsLogsScreen(Screen):
    """
    This screen displays the user's past results from the sentence puzzle exercises.
    It reads from the 'sentence_exercise_log.txt' file and shows each entry in a scrollable list.
    """

    def on_enter(self):
        """
        Called when the screen is entered. It clears the previous content,
        reads the log file, and dynamically displays each logged result as a formatted label.
        """
        # Clear previous widgets from the container
        self.ids.sentence_results_container.clear_widgets()

        log_path = BASE_LOG_DIR / "sentence_exercise_log.txt"

        # Display message if no logs exist
        if not log_path.exists():
            self.ids.sentence_results_container.add_widget(Label(
                text="No sentence puzzle logs found.",
                font_name="assets/ProximaSoft-Regular",
                font_size="15sp",
                color=(1, 1, 1, 1),
                size_hint_y=None,
                height=dp(30)
            ))
            return

        # Read and display each line from the log
        with open(log_path, encoding="utf-8") as f:
            lines = f.readlines()

        for line in lines:
            label = Label(
                text=line.strip(),
                font_name="assets/ProximaSoft-Regular",
                font_size="15sp",
                color=(1, 1, 1, 1),
                text_size=(Window.width - dp(60), None),
                halign="left",
                valign="top",
                size_hint_y=None,
                padding=(dp(10), dp(10))
            )
            label.bind(texture_size=lambda instance, value: setattr(label, 'height', value[1]))
            self.ids.sentence_results_container.add_widget(label)

class SpeechFeedbackLogsScreen(Screen):
    """
    Displays a detailed log and visual analysis of the user's speech feedback history.
    It reads from the 'speech_feedback_log.txt' file, highlights frequent feedback terms,
    and presents both visual insights and full transcript logs.
    """

    def on_enter(self):
        """
        Called when the screen is entered.
        It parses the speech feedback log file, visualises common feedback phrases,
        and displays a scrollable list of raw feedback entries.
        """
        from collections import Counter
        import matplotlib.pyplot as plt
        import tempfile
        from kivy.uix.image import Image

        # Clear previous widgets from the scrollable container
        self.ids.speech_feedback_container.clear_widgets()
        log_path = BASE_LOG_DIR / "speech_feedback_log.txt"

        lines = []
        # Read feedback lines if the file exists
        if log_path.exists():
            with open(log_path, encoding="utf-8") as f:
                lines = f.readlines()

        # Show fallback message if no feedback available
        if not lines:
            self.ids.speech_feedback_container.add_widget(Label(
                text="No speech feedback logs found.",
                font_name="assets/ProximaSoft-Regular",
                color=(1, 1, 1, 1),
                size_hint_y=None,
                height=dp(30)
            ))
            return

        # Extract keywords or terms from lines starting with "- "
        feedback_terms = []
        for line in lines:
            if line.strip().startswith("- "):
                word = line[2:].strip().split()[0].lower()
                feedback_terms.append(word)

        # Create visual chart of common feedback terms
        if feedback_terms:
            fig, ax = plt.subplots(figsize=(6, 3), dpi=100)
            term_counts = Counter(feedback_terms)
            top_terms = term_counts.most_common(10)  # Top 10 most used words
            words, counts = zip(*top_terms)

            ax.bar(words, counts, color="#5C6BC0")
            ax.set_title("Common Speech Feedback Terms", fontsize=10)
            ax.set_ylabel("Frequency", fontsize=9)
            ax.set_xticklabels(words, rotation=45, ha='right', fontsize=8)
            plt.tight_layout()

            # Save chart to a temporary file
            temp_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
            plt.savefig(temp_path)
            plt.close()

            # Display the generated chart in the app
            self.ids.speech_feedback_container.add_widget(Image(
                source=temp_path,
                size_hint_y=None,
                height=dp(240),
                allow_stretch=True
            ))

        # Show the full speech feedback log below the chart
        for line in lines:
            label = Label(
                text=line.strip(),
                font_name="assets/ProximaSoft-Regular",
                font_size="15sp",
                color=(1, 1, 1, 1),
                padding=(dp(10), dp(10)),
                size_hint_y=None,
                text_size=(Window.width - dp(60), None),
                halign="left",
                valign="top"
            )
            label.bind(texture_size=lambda instance, value: setattr(label, 'height', value[1]))
            self.ids.speech_feedback_container.add_widget(label)

class ChatTranscriptsLogsScreen(Screen):
    """
    This screen displays all logged chatbot conversations grouped by date.
    It also supports real-time filtering using a search bar to match keywords across transcripts.
    """

    def on_enter(self):
        """
        Called when the user enters the screen.
        Clears any existing search query and displays all available transcript logs.
        """
        # Reset the search bar if present
        if "transcript_search" in self.ids:
            self.ids.transcript_search.text = ""
        self.display_transcripts()

    def display_transcripts(self, query=""):
        """
        Loads and displays all chatbot conversation logs that match the given search query.
        Groups transcripts by date and lists each conversation log file with its content.
        """
        from collections import defaultdict

        # Clear previous transcript entries
        self.ids.chat_transcripts_container.clear_widgets()

        # Find all transcript files by filename pattern
        transcript_files = sorted(BASE_LOG_DIR.glob("chat_transcript_*.txt"))

        grouped_logs = defaultdict(list)

        for file in transcript_files:
            with open(file, encoding="utf-8") as f:
                content = f.read().strip()

            # Filter by query (if specified) against content or filename
            if query.lower() not in content.lower() and query.lower() not in file.stem.lower():
                continue

            # Extract date from filename (e.g. 'chat_transcript_20250504_143012.txt' → 20250504)
            date = file.stem.split("_")[1][:8]
            grouped_logs[date].append((file.stem, content))

        # Display message if nothing matches the search
        if not grouped_logs:
            self.ids.chat_transcripts_container.add_widget(Label(
                text="No matching transcripts.",
                font_name="assets/ProximaSoft-Regular",
                font_size="15sp",
                color=(1, 1, 1, 1),
                size_hint_y=None,
                height=dp(30)
            ))
            return

        # Display each group by date
        for date, entries in grouped_logs.items():
            # Add a bold date heading
            date_label = Label(
                text=f"[b]{date[:4]}-{date[4:6]}-{date[6:]}[/b]",  # Format: YYYY-MM-DD
                markup=True,
                font_name="assets/ProximaSoft-Bold",
                font_size="16sp",
                color=(1, 1, 1, 1),
                size_hint_y=None,
                height=dp(30)
            )
            self.ids.chat_transcripts_container.add_widget(date_label)

            # Add each transcript log under the date
            for name, content in entries:
                log_label = Label(
                    text=f"[b]{name}[/b]\n{content}",
                    markup=True,
                    font_name="assets/ProximaSoft-Regular",
                    font_size="15sp",
                    color=(1, 1, 1, 1),
                    size_hint_y=None,
                    text_size=(Window.width - dp(60), None),
                    halign="left",
                    valign="top"
                )
                log_label.bind(texture_size=lambda instance, size: setattr(instance, "height", size[1]))
                self.ids.chat_transcripts_container.add_widget(log_label)

    def filter_transcripts(self, query):
        """
        Triggered by the search bar to filter displayed transcripts based on user input.
        """
        self.display_transcripts(query)

class FluentiqueApp(App):
    """
    Root application class for Fluentique.
    Manages screen transitions, global token storage, and high-level user flows such as registration.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.token = None  # Store user authentication token globally

    def store_token(self, token_value):
        """Store the authentication token globally for use across all screens."""
        self.token = token_value

    def build(self):
        """
        Called automatically when the app is launched.
        Loads all `.kv` files and initialises the ScreenManager with all defined screens.
        """
        # Load all UI layouts (defined separately in .kv files)
        Builder.load_file("kv/intro_continue.kv")
        Builder.load_file("kv/intro_get_started.kv")
        Builder.load_file("kv/registration.kv")
        Builder.load_file("kv/login.kv")
        Builder.load_file("kv/loading.kv")
        Builder.load_file("kv/home_completed.kv")
        Builder.load_file("kv/home_incomplete.kv")
        Builder.load_file("kv/quiz.kv")
        Builder.load_file("kv/results.kv")
        Builder.load_file("kv/grammar_exercise.kv")
        Builder.load_file("kv/grammar_results.kv")
        Builder.load_file("kv/sentence_result.kv")
        Builder.load_file("kv/speech_practice.kv")
        Builder.load_file("kv/speech_feedback.kv")
        Builder.load_file("kv/final_speech_feedback.kv")
        Builder.load_file("kv/leaderboard.kv")
        Builder.load_file("kv/profile.kv")
        Builder.load_file("kv/upload_avatar.kv")
        Builder.load_file("kv/chatbot_tutor.kv")
        Builder.load_file("kv/summary.kv")
        Builder.load_file("kv/mistake_log_menu.kv")
        Builder.load_file("kv/mistake_logs.kv")

        # Define and register all screens in the ScreenManager
        sm = ScreenManager()
        sm.add_widget(IntroContinueScreen(name='intro_continue'))
        sm.add_widget(IntroGetStartedScreen(name='intro_get_started'))
        sm.add_widget(LoginScreen(name='login'))
        sm.add_widget(RegistrationScreen(name='registration'))
        sm.add_widget(LoadingScreen(name='loading'))
        sm.add_widget(HomeCompleteScreen(name='home_completed'))
        sm.add_widget(HomeIncompleteScreen(name='home_incomplete'))
        sm.add_widget(QuizScreen(name='quiz'))
        sm.add_widget(ResultsScreen(name='results'))
        sm.add_widget(GrammarExerciseScreen(name='grammar_exercise'))
        sm.add_widget(GrammarResultsScreen(name='grammar_results'))
        sm.add_widget(SentenceResultScreen(name='sentence_result'))
        sm.add_widget(SpeechPracticeScreen(name='speech_practice'))
        sm.add_widget(SpeechFeedbackScreen(name='speech_feedback'))
        sm.add_widget(FinalSpeechFeedbackScreen(name='final_speech_feedback'))
        sm.add_widget(LeaderboardScreen(name='leaderboard'))
        sm.add_widget(ProfileScreen(name='profile'))
        sm.add_widget(UploadAvatarScreen(name='upload_avatar'))
        sm.add_widget(ChatbotTutorScreen(name='chatbot_tutor'))
        sm.add_widget(SummaryScreen(name='summary_screen'))
        sm.add_widget(MistakeLogMenuScreen(name='mistake_log_menu'))
        sm.add_widget(QuizMistakesLogsScreen(name='quiz_mistakes_logs'))
        sm.add_widget(QuizScoresLogsScreen(name='quiz_scores_logs'))
        sm.add_widget(SentenceResultsLogsScreen(name='sentence_results_logs'))
        sm.add_widget(SpeechFeedbackLogsScreen(name='speech_feedback_logs'))
        sm.add_widget(ChatTranscriptsLogsScreen(name='chat_transcripts_logs'))

        # Set initial screen if needed
        # sm.current = 'chatbot_tutor'
        return sm

    def register_user(self):
        """
        Handles user registration using email and password.
        Sends the data to the backend and navigates to the home screen upon success.
        """
        email = self.root.get_screen('registration').ids.email_input.text
        password = self.root.get_screen('registration').ids.password_input.text
        confirm_password = self.root.get_screen('registration').ids.confirm_password_input.text

        if password != confirm_password:
            self.show_popup("Error", "Passwords do not match!")
            return

        payload = {
            "email": email,
            "password": password
        }

        try:
            response = requests.post(f"{API_BASE_URL}/auth/register", json=payload, timeout=REQUEST_TIMEOUT)
            if response.status_code == 200:
                self.show_popup("Success", "Registration successful!")
                app = App.get_running_app()
                app.root.current = 'loading'
                Clock.schedule_once(lambda dt: setattr(app.root, 'current', 'home_incomplete'), 3)
            else:
                error = response.json().get('detail', 'Registration failed')
                self.show_popup("Error", error)
        except Exception as e:
            self.show_popup("Error", str(e))

    def show_popup(self, title, message):
        """
        Display a modal popup with a given title and message.
        Useful for alerts, errors, or confirmations.
        """
        popup = Popup(
            title=title,
            content=Label(text=message),
            size_hint=(None, None),
            size=(300, 200)
        )
        popup.open()

    def go_to_leaderboard(self):
        """
        Navigate to the leaderboard screen and ensure it refreshes with the latest data.
        """
        screen = self.root.get_screen('leaderboard')
        screen.on_pre_enter()  # Ensure data is updated before showing
        self.root.current = 'leaderboard'

# App entry point
if __name__ == '__main__':
    FluentiqueApp().run()

