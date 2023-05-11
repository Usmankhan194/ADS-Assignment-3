import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Data loading in progress
Urban_Population = pd.read_csv('UrbanPopulation.csv', skiprows=4)
Urban_Population.tail(10)

Urban_Population.isnull().sum()

# Omit null observations from data
Urban_Population.drop([ 'Unnamed: 66'], inplace=True , axis=1)

Urban_Population.isnull().sum().sum()

cols = [ '1960', '1961', '1962', '1963', '1964', '1965', '1966', '1967', '1968',
       '1969', '1970', '1971', '1972', '1973', '1974', '1975', '1976', '1977',
       '1978', '1979', '1980', '1981', '1982', '1983', '1984', '1985', '1986',
       '1987', '1988', '1989', '1990', '1991', '1992', '1993', '1994', '1995',
       '1996', '1997', '1998', '1999', '2000', '2001', '2002', '2003', '2004',
       '2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013',
       '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021',]
world_dataa = Urban_Population[cols]
Urban_Population

def exponential_growth_Urban_Population(x, a, b):
    return a * np.exp(b * x)

x = np.array(range(1960, 2022)) 
uk_data = Urban_Population[Urban_Population["Country Name"] == "United Kingdom"]
uk_data
y = (np.array(uk_data[uk_data['Indicator Name']== "Urban population"]))[0][4:76]

popt, pcov = curve_fit(exponential_growth_Urban_Population, x, y)

from scipy import stats
# define the range of years for prediction
prediction_years = np.array(range(2022, 2042))

# use the model for predictions
predicted_values = exponential_growth_Urban_Population(prediction_years, *popt)

# calculate confidence ranges using the err_ranges function
def err_ranges(func, xdata, ydata, popt, pcov, alpha=0.05):
    perr = np.sqrt(np.diag(pcov))
    n = len(ydata)
    dof = max(0, n - len(popt))
    tval = np.abs(stats.t.ppf(alpha / 2, dof))
    ranges = tval * perr
    return ranges

lower_bounds, upper_bounds = err_ranges(exponential_growth_Urban_Population, x, y, popt, pcov)

# plot the best fitting function and the confidence range
plt.plot(x, y, '*', label='data')
plt.plot(x, y, 'g-', label='fit')
plt.fill_between(prediction_years, predicted_values - upper_bounds, predicted_values + lower_bounds, alpha=0.3)
plt.title('Best Fitting Function Vs Confidence Range')
plt.xlabel('X-axis:Years')
plt.ylabel('Y-axis:Urban Population')
plt.legend()
plt.show()
