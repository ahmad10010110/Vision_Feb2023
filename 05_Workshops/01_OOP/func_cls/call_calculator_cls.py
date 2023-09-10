from BasicCalculator import BasicCalculator
from ScientificCalculator import ScientificCalculator

bsc_calculator = BasicCalculator(8, 9)
sci_calculator = ScientificCalculator(10, 5)

print('Addition: ', bsc_calculator.add() )
print('Subtraction: ', bsc_calculator.subtract() )

print('Power: ', sci_calculator.power())
print('Squar_root: ', sci_calculator.square_root())

