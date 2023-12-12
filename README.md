# Noita-Speed-Calculator
Finds Combinations of modifiers to use for Speed Based Long Distance Travel.

### Standalone Version
- open a console and navigate to the folder of the .exe, then run one of the commands below


### Example Usage:


#### New Game PW Travel 
Standard settings, shallow search depth

`speed_calc.exe 35840`

#### New Game Plus PW Travel
Favor Budget Builds, normal search depth

`speed_calc.exe 32760 -b budget -m normal`
#### Minimum Terrain duping distance
Favor more accurate solutions and search deeper:

`speed_calc.exe 262144 -b accurate -m deep`

#### Floating Point Limit: 
Search very deep, give very little weight to easiness, but restrict to only some modifiers for a massive speedup

`speed_calc.exe 536870912 -b accurate -m verydeep -w 0.01 -coefs 0.32 0.75 1.68 2.0`

### Manual Setup
- Install python.
- Clone this repository, or download the code as a zip file, and extract it.
- Open a console and navigate to the code directory
- Install the dependencies:`pip install -r requirements.txt`

