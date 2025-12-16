import os
import sys

def main():
    print("Starting NeuroScan AI Local Demo...")
    print("Ensure you have 'streamlit' installed (pip install streamlit).")
    
    # Run Streamlit command
    os.system("streamlit run deploy/app.py")

if __name__ == "__main__":
    main()
