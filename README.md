# Noita-Speed-Calculator
Finds Combinations of modifiers to use for Speed Based Long Distance Travel.

Example Usage:
Standard settings:
python -u speed_calc.py 35840

More accurate, and a deeper search:
python -u speed_calc.py 262144 -b accurate -m deep

Favor Budget Builds, shallower search
python -u speed_calc.py 32760 -b budget -m shallow

Search very deep, but restrict to only some modifiers for a massive speedup
python -u speed_calc.py 536870912 -b accurate -m verydeep -w 0.01 -coefs 0.32 0.75 1.68 2.0
