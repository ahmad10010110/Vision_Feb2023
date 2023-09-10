import math


num1 = 10
num2 = 5

addition = num1 + num2
subtraction = num1 - num2
multiplication = num1 * num2


if num2 == 0:
    raise ValueError("Can not divide by zero!")
division = num1 / num2

power = num1**num2

if num1<=0 or num2 <=0:
    raise ValueError("Cannot take logharithm of non_positive numbers")
logarthim = math.log(num1, num2)


if num1<0:
    raise ValueError("Cannot take square root of negative number!")
square_root = num1 ** 0.5


print('Addition: ', addition)
print('Subtraction: ', subtraction)
print('Multiplicatin: ', multiplication)
print('Division: ', division)
print('Power: ', power)
print('Square Root: ', square_root)
