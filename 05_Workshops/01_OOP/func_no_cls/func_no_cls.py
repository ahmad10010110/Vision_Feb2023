import math

def add(num1, num2):
    return num1 + num2


def subtract(num1, num2):
    return num1 - num2

def multiply(num1, num2):
    return num1 * num2

def divide(num1, num2):
    if num2 == 0:
        raise ValueError('Cannot divide by zero!')
    return num1 / num2

def power(num1, num2):
    return num1**num2

def square_root(num1):
    if num1 < 0:
        raise ValueError('Cannot take square root of negative number!')
    return num1 ** 0.5

def logarithm(num1, num2):
    if num1 <= 0 or num2 <= 0:
        raise ValueError('Cannot take alghorithm of non positive numbers')
    return math.log(num1, num2)