#Function
def greet():
    """
    the function prints Hi!
    """
    print('Hi')


greet()
#greet()
#greet()

def greet(name):
    print(f'Hi {name}')



greet('Asghar')

def greet(first_name, last_name):
    return f'Hello {first_name} {last_name}'

g = greet('Ali', 'Alavi')
out = g + ' welcome!'
#print(type(g))
print(out)


# open the camera! --> func_1
# take a quick photo! --> func_2
# recognize the guest! --> func_3
# say hello to the guest! --> func_4

#def welcome():
    #func_1
    #func_2
    #....
    
# Class!
# utils!
## calculate_area!
    
def multiplication(a, b, c=2):
    return a * b * c

multiplication(1, 2)


#lambda!
multiply = lambda a, b, c=2 : a * b * c
print(multiply(3, 4))


def func_one():
    return 2

def func_two(func):
    return func * 2


a = func_two(func_one())
print(a)
    
    
#String
#str.format!

name = 'John'
s_1 = f'Hello, {name}'
print(s_1)

s_2 = f'Hello, {name.upper()}'
print(s_2)

num = 200
s_3 = f'{num: 06}'
print(s_3)


num_2 = 9.98564
s_4 = f'{num_2: 0.2f}'
print(s_4)


num_3 = 80000000000000000000
s_5 = f'{num_3 : ,}'
print(s_5)

num_4 = 0.89654587
s_6 = f'{num_4: 0.2%}'
print(s_6)



#Backslash
print('Hello,\n Adam')


print('Hello,\t Adam')

print(len('\\n'))

txt = '"python\'s awesome"she said'
print(txt)

txt_2 = "we are \"IRANIAN!\" "
print(txt_2)

PATH = 'lang\tv\nPython\t3'
print(PATH)


PATH_2 = 'lang\\tv\\nPython\\t3'
print(PATH_2)

PATH_3 = r'lang\tv\nPython\t3'
print(PATH_3)


print(PATH_2 == PATH_3)














