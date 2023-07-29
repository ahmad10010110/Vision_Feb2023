def division_func(a, b):
    if b != 0:
        c = a / b
    else:
        c = 'Error, Please input b != 0'
    return c

print(division_func(4, 0))

#try , except!
try:
    #Code that may cause error!
    pass
except:
    #handle the error !
    pass

# General Exception
def division_func_2(a, b):
        c = a / b
        return c
try:
    numerator = float(input('enter the numerator:'))
    denominator = float(input('enter the denominator: '))
    d = division_func_2(numerator, denominator)
except:
    print('Error occured!')
    
 
    
# Specific Exception
def division_func_3(a, b):
        c = a / b
        return c
try:
    numerator = float(input('enter the numerator:'))
    denominator = float(input('enter the denominator: '))
    d = division_func_3(numerator, denominator)
    print(d)
except ValueError as error:
    print(error)
    print('Please enter number!')


# several specific exceptions!

try:
    pass
except ValueError:
    pass
except ZeroDivisionError:
    pass
except Exception as error: #for all exceptions!
    print(error)


def division_func_4(a, b):
        c = a / b
        return c
try:
    numerator = float(input('enter the numerator:'))
    denominator = float(input('enter the denominator: '))
    d = division_func_4(numerator, denominator)
    print(d)
except ValueError as error:
    print(error)
    print('Please enter number!')
except ZeroDivisionError as error:
    print('Please enter the number more than 0')
except Exception as error:
    print(error)
else:
    print('the function executed without any error!')
finally:
    print('finished!')
    


#Recursion!

#multiply a and b, using iteration!

def multiply(a, b):
    result = 0
    for i in range(b):
        result += a
    return result

print(multiply(3, 12))


def mult(a, b):
    if b == 1:
        return a 
    else:
        print(a + mult(a, b-1))
        return a + mult(a, b-1)
 
mult(3, 12)


def fact_iter(n):
    j = 1
    for i in range(1, n+1):
        j *= i
    return j

print(fact_iter(5))

def fact_recur(n):
    if n == 1:
        return 1
    else:
        return n * fact_recur(n-1)
        
print(fact_recur(5))

    
























