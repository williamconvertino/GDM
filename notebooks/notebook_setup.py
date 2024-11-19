import sys
import os
import subprocess
  
# Set the sys path
path = os.path.abspath(os.path.join(os.getcwd(), '../'))
if path not in sys.path:
  sys.path.append(path)
  
# Install the requirements
try:
  subprocess.check_call(["pip3 install torch>=2.5.0 --index-url https://download.pytorch.org/whl/cu121"])
  print("PyTorch installed")
except:
  print("Failed to install PyTorch")
try:
  subprocess.check_call(["pip", "install", "-r", "../requirements.txt", "--quiet"])
  print("Requirements installed")
except:
  print("Failed to install requirements")