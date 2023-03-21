import os


def use_awesamlib():
    print("USING AWESAMLIB")
    global adaptive_maxfilter, maxfilter_kernels, compute_probabilities
    from . import libwrapper

    adaptive_maxfilter = libwrapper.adaptive_maxfilter
    compute_probabilities = libwrapper.compute_probabilities


def use_python_backend():
    print("USING PYTHON-BACKEND")
    global adaptive_maxfilter, maxfilter_kernels, compute_probabilities
    from . import pythonbackend

    adaptive_maxfilter = pythonbackend.adaptive_maxfilter
    compute_probabilities = pythonbackend.compute_probabilities


directory = os.path.dirname(os.path.abspath(__file__))

## COMPILATION

if os.path.isfile(f"{directory}/awesamlib.so"):
    USE_AWESAMLIB = True
else:
    print("Attempting compilation of awesamlib (works only if gcc is available).")
    os.system(
        f"gcc -shared -o {directory}/awesamlib.so -fPIC {directory}/awesamlib.c -O3"
    )
    if not os.path.isfile(f"{directory}/awesamlib.so"):
        print("Compilation Failed")
        print("Using the python backend of AWESAM.")
        USE_AWESAMLIB = False
    else:
        print("Compilation successful")
        USE_AWESAMLIB = True

## CONFIGURATION

adaptive_maxfilter = None
maxfilter_kernels = None
compute_probabilities = None

if USE_AWESAMLIB:
    use_awesamlib()
if not USE_AWESAMLIB:
    use_python_backend()
