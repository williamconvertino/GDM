import sys
import os
import subprocess
  
# Set the sys path
path = os.path.abspath(os.path.join(os.getcwd(), '../'))
if path not in sys.path:
  sys.path.append(path)
  
# Install the requirements
try:
  subprocess.check_call(["pip", "install", "-r", "../requirements.txt", "--quiet", "pip==21.0.1"]) # Downgrade pip version to suppress warnings
  print("Requirements installed")
except:
  print("Failed to install requirements")