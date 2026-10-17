import sys
# Reconfigure stdout/stderr to UTF-8
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

import os
import glob

# Set environment variables
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Add nvidia DLL directories to path
nvidia_dir = os.path.abspath(os.path.join('.venv', 'Lib', 'site-packages', 'nvidia'))
if os.path.exists(nvidia_dir):
    print("Adding NVIDIA DLL directories...")
    for p in glob.glob(os.path.join(nvidia_dir, '*', 'bin')):
        os.add_dll_directory(p)

# Also add paddle/libs
paddle_libs = os.path.abspath(os.path.join('.venv', 'Lib', 'site-packages', 'paddle', 'libs'))
if os.path.exists(paddle_libs):
    print("Adding paddle/libs:", paddle_libs)
    os.add_dll_directory(paddle_libs)

print("Step 1: Importing paddleocr")
from paddleocr import LayoutDetection
print("Import successful!")

print("Step 2: Initializing LayoutDetection by model_name='PP-DocLayout_plus-L' on CPU")
try:
    layout = LayoutDetection(model_name='PP-DocLayout_plus-L', threshold=0.20, device='cpu')
    print("Step 3: Initialized successfully by model_name on CPU!")
except Exception as e:
    import traceback
    traceback.print_exc()
