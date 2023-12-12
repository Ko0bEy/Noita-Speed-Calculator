# Noita-Speed-Calculator
Finds Combinations of modifiers to use for Speed Based Long Distance Travel.

Requires Pyhton 3.10+.

Some basic Knowledge about Python is recommended, but not required.

### Standalone Version
- extract the zip
- open a console and navigate to the folder
- use `venv\scripts\python` instead of `python` for usage

### Example Usage:


#### New Game PW Travel 
Standard settings

`speed_calc.exe 35840`

#### New Game Plus PW Travel
Favor Budget Builds, shallower search

`speed_calc.exe 32760 -b budget -m shallow`
#### Minimum Terrain duping distance
Favor more accurate solutions and search deeper:

`speed_calc.exe 262144 -b accurate -m deep`

#### Floating Point Limit: 
Search very deep, give very little weight to easiness, but restrict to only some modifiers for a massive speedup

`speed_calc.exe 536870912 -b accurate -m verydeep -w 0.01 -coefs 0.32 0.75 1.68 2.0`

### Installation
- Install python.
- Clone this repository, or download the code as a zip file, and extract it.
- Open a console and navigate to the code directory
- Install the dependencies:`pip install -r requirements.txt`
  - numpy, for faster caluclations
  - tqdm, for a progress bar

From the console, navigate to the code location and run a command as shown below.

