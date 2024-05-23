# codes_numerical
# NEWTONS FORWARD INTERPOLATION
import numpy as np

# Input number of data points
n = int(input('ENTER THE NUMBER OF DATA POINTS: '))
x = np.zeros(n)
y = np.zeros((n, n))

# Function to calculate u term
def u_cal(u, n):
    temp = u
    for i in range(1, n):
        temp *= (u - i)
    return temp

# Function to calculate factorial
def fact(n):
    f = 1
    for i in range(2, n + 1):
        f *= i
    return f

# Input the values of x and y
print("ENTER THE VALUES OF X AND Y:")
for i in range(n):
    x[i] = float(input(f'x[{i}]= '))
    y[i][0] = float(input(f'y[{i}]= '))

# Calculating the forward difference table
for i in range(1, n):
    for j in range(n - i):
        y[j][i] = y[j + 1][i - 1] - y[j][i - 1]

# Display the forward difference table
print("\nFORWARD DIFFERENCE TABLE:\n")
for i in range(n):
    print(f'{x[i]:0.2f}', end='\t')
    for j in range(n - i):
        print(f'{y[i][j]:0.2f}', end='\t')
    print()

# Value to interpolate
value = 1895
sum = y[0][0]
u = (value - x[0]) / (x[1] - x[0])

# Interpolation process
for i in range(1, n):
    sum += (u_cal(u, i) * y[0][i]) / fact(i)

print(f"\nVALUE OF u IS: {round(u, 6)}")
print(f"VALUE AT {value} IS: {round(sum, 6)}")
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#NEWTON'S BACKWARD INTERPOLATION
import numpy as np
n = int(input("ENTER THE NUMBER OF DATA POINTS: "))
x = np.zeros((n))
y = np.zeros((n, n))

def u_cal(u, n):
    temp = u
    for i in range(1, n):
        temp *= (u + i)
    return temp

def fact(n):
    f = 1
    for i in range(2, n + 1):
        f *= i
    return f

print("ENTER THE VALUES OF X AND Y:")

for i in range(n):
    x[i] = float(input('x[' + str(i) + ']= '))
    y[i][0] = float(input('y[' + str(i) + ']= '))

for i in range(1, n):
    for j in range(n - 1, i - 1, -1):
        y[j][i] = y[j][i - 1] - y[j - 1][i - 1]

print("\nBACKWARD DIFFERENCE TABLE:\n")

for i in range(n):
    print('% 0.2f' % x[i], end=' ')
    for j in range(i+1):
        print('\t%0.2f' % y[i][j], end=' ')
    print()

value = 1925
sum = y[n - 1][0]
u = (value - x[n - 1]) / (x[1] - x[0])

for i in range(1, n):
    sum += (u_cal(u, i) * y[n - 1][i]) / fact(i)
print("\nVALUE OF u IS:", round(u, 6))
print("VALUE AT", value, "IS:", round(sum, 6))
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#EULER METHOD BY USING PYTHON
#FUNCTION TO BE SOLVED
def f(x,y):
  return x+y

#EULER METHOD
def euler(x0,y0,xn,n):
  #CACULATING STEP SIZE.
  h=(xn-x0)/n
  print('\n ----------SOLUTION----------')
  print('--------------------------------')
  print('x0\ty0\tslope\tyn')
  print('--------------------------------')
  for i in range(n):
    slope = f(x0,y0)
    yn = y0+h *slope
    print('%.4f\t%.4f\t%0.4f\t%.4f'%(x0,y0,slope,yn))
    print('--------------------------------')
    y0=yn
    x0=x0+h
  print('\nAt x=%.4f, y=%.4f' %(xn,yn))
#INPUTS
print('ENTER INITIAL CONDITIONS: ')
x0 = float(input('x0 = '))
y0 = float(input('y0 = '))

print('ENTER CALCULATION POINT: ')
xn = float(input('xn = '))

print('ENTER THE NUMBER OF STEPS: ')
step = int(input('NUMBER OF STEPS = '))
#EULER METHOD CALL.
euler(x0,y0,xn,step)
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# RUNGE-KUTTA (RK4) METHOD
def f(x, y):
    return x-y
def runge_kutta(x0, y0, xn, n):
    # CALCULATING STEP SIZE.
    h = (xn - x0) / n
    print('\n ----------SOLUTION----------')
    print('--------------------------------')
    print('x0\ty0\tyn')
    print('--------------------------------')
    for i in range(n):
        k1 = h * (f(x0, y0))
        k2 = h * (f((x0+h/2), (y0+k1/2)))
        k3 = h * (f((x0+h/2), (y0+k2/2)))
        k4 = h * (f((x0+h), (y0+k3)))
        k = (k1 + 2 * k2 + 2 * k3 + k4) / 6
        yn = y0 + k
        print('%.4f\t%.4f\t%.4f'% (x0, y0, yn))
        print('--------------------------------')
        y0 = yn
        x0 = x0 + h

    print('\nAt x=%.4f, y=%.4f' % (xn, yn))

# INPUTS
print('ENTER INITIAL CONDITIONS: ')
x0 = float(input('x0 = '))
y0 = float(input('y0 = '))

print('ENTER CALCULATION POINT: ')
xn = float(input('xn = '))

print('ENTER THE NUMBER OF STEPS: ')
step = int(input('NUMBER OF STEPS = '))

# RUNGE-KUTTA METHOD CALL.
runge_kutta(x0, y0, xn, step)
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#simpson's 1/3rd method
import math
def f(x):
    return 1 / (1 + x ** 2)

def simpsons_one_third(x0, xn, n):
    # Ensure that n is even
    if n % 2 != 0:
        raise ValueError("Number of sub-intervals must be even for Simpson's 1/3rd rule.")
    
    h = (xn - x0) / n
    integration = f(x0) + f(xn)
    
    for i in range(1, n):
        k = x0 + i * h
        if i % 2 == 0:
            integration += 2 * f(k)
        else:
            integration += 4 * f(k)
    
    integration *= h / 3
    return integration

# Input validation loop to ensure valid numerical input
while True:
    try:
        lower_limit = float(input("ENTER THE LOWER LIMIT OF THE INTEGRATION: "))
        upper_limit = float(input("ENTER THE UPPER LIMIT OF THE INTEGRATION: "))
        sub_intervals = int(input("ENTER THE NUMBER OF SUB INTERVALS (must be even): "))
        
        # Check if the number of sub-intervals is even
        if sub_intervals % 2 != 0:
            print("The number of sub-intervals must be even. Please try again.")
            continue
        
        result = simpsons_one_third(lower_limit, upper_limit, sub_intervals)
        print("INTEGRATION RESULT BY SIMPSON'S 1/3rd METHOD IS: %0.6f" % result)
        break  # Exit the loop if input and computation are successful
    except ValueError as e:
        print(f"Invalid input! {str(e)}. Please enter a valid numerical value.")
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# TRAPEZIODAL METHOD.
import math
def f(x):
    return 1 / (1 + x ** 2)

def trapezoidal(x0, xn, n):
    h = (xn - x0) / n
    integration = f(x0) + f(xn)
    for i in range(1, n):
        k = x0 + i * h
        integration += 2 * f(k)
    integration *= h / 2
    return integration

# Input validation loop to ensure valid numerical input
while True:
    try:
        lower_limit = float(input("ENTER THE LOWER LIMIT OF THE INTEGRATION: "))
        upper_limit = float(input("ENTER THE UPPER LIMIT OF THE INTEGRATION: "))
        sub_intervals = int(input("ENTER THE NUMBER OF SUB INTERVALS: "))
        result = trapezoidal(lower_limit, upper_limit, sub_intervals)
        print("INTEGRATION RESULT BY TRAPEZOIDAL METHOD IS: %0.6f" % result)
        break  # Exit the loop if input and computation are successful
    except ValueError:
        print("Invalid input! Please enter a valid numerical value.")


