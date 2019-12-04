# alpha-dogs
This repo implements AlphaDOGS algorithm to optimize the objective function computed from a time-averaged statistics.

## Prerequisites
This repo has been tested with the following environment:
```
python 3.7
scipy 1.1.0
SNOPT 7
```
SNOPT package is available at [here](https://github.com/snopt/snopt-python).

## How to use
Run the following command in terminal at the root folder.
To test the 1D example:
```
python3 1Dexample.py
```
To test the 2D example:
```
python3 2Dexample.py
```

## Result to expect
All the plots are generated in the folder: ```root/plot```.

1. In your terminal, the optimization code will display the information about point to be sampled at each iteration:
![1](/figures/optimization_info_display.png)

2. The optimization code will plot the values of objective function, discrete search function, and continuous search function in the folder `plot` under the root directory. E.g.:
![2](/figures/plot1D39.png)

3. After the optimization completes, the information of candidate point and distance will be plotted as follow:
![3](/figures/Candidate_point.png)
![4](/figures/Distance.png)

