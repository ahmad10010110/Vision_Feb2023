import math

from BasicCalculator import BasicCalculator

class ScientificCalculator(BasicCalculator):
    def __init__( self, num1, num2):
        self.num1 = num1
        self.num2 = num2
        
    
    
    def power(self):
        return self.num1**self.num2

    def square_root(self):
        if self.num1 < 0:
            raise ValueError('Cannot take square root of negative number!')
        return self.num1 ** 0.5

    def logarithm(self):
        if self.num1 <= 0 or self.num2 <= 0:
            raise ValueError('Cannot take alghorithm of non positive numbers')
        return math.log(self.num1, self.num2)