#!/bin/bash

# Navigate to your Flask app directory
cd app.py

# Check for dependencies and install if they are not installed
# pip install -r requirements.txt

# Start Flask
export FLASK_APP=app.py
export FLASK_ENV=development
flask run

# Optionally open the browser
open http://localhost:5000