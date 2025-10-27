#!/bin/bash

echo "=================================================="
echo "NLQ to SQL Query Generator - Startup Script"
echo "=================================================="
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "Virtual environment created!"
    echo ""
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install/Update dependencies
echo ""
echo "Checking dependencies..."
pip install -q -r requirements.txt

# Check for .env file
if [ ! -f ".env" ]; then
    echo ""
    echo "WARNING: .env file not found!"
    echo "Please create .env from .env.example and add your OpenAI API key"
    echo ""
    exit 1
fi

# Create necessary directories
mkdir -p uploads cache logs

# Start the application
echo ""
echo "=================================================="
echo "Starting NLQ to SQL Generator..."
echo "=================================================="
echo ""
echo "Application will be available at: http://localhost:5001"
echo "Press Ctrl+C to stop the server"
echo ""
echo "=================================================="
echo ""

python3 nlq_sql_app.py
