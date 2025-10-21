import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    print("Installing requirements...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    print("Requirements installed successfully!")

def run_streamlit():
    """Run the Streamlit app"""
    print("Starting Streamlit app...")
    subprocess.run([sys.executable, "-m", "streamlit", "run", "streamlit_app.py"])

if __name__ == "__main__":
    # Check if requirements.txt exists
    if not os.path.exists("requirements.txt"):
        print("Error: requirements.txt not found!")
        sys.exit(1)
    
    # Install requirements
    try:
        install_requirements()
    except subprocess.CalledProcessError:
        print("Error installing requirements. Please install manually:")
        print("pip install -r requirements.txt")
        sys.exit(1)
    
    # Run the app
    run_streamlit()
