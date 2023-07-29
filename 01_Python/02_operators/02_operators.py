#Arithmetic Ops. 
a = 20
b = 15

##Addition
print( "a + b: ", a + b)

##Subtraction
print("a - b: ", a - b)

##Multiplication
print("a x b: ",  a * b )

## Division
print("a / b: ", a / b)

## Remainder
print("a % b:", a % b)

## Power
print( " a ** b: ", a ** b)

## Floor Division
print("a // b: ", a//b)


#Assignment Ops. 
c = 10

##Assign value
d = c

print(id(d))
print(id(c))

print( d is c)
print( d == c)


d += c 
print(d)

d -= c
print(d)

d *= c 
print(d)


print( d is c)

#Comparision Ops. 
e = 8
f = 12


##equal
print(e == f)

##Not equal
print( e != f)

##Greater than
print(e > f)

## Less than 
print(e < f)

## Greater than or equal to 
print( e >= f)

## Less than or equal to 
print(e  <= f)


#Logical Ops. 


g = True
h = False

##And
print( g and h)

##Or
print(g or h )

##Not
print( not g)


#Membership Ops.
i = 10
j = 20

lst = [10, 24, 18]

print( i in lst)
print(j in lst)

print( j not in lst)

#Identity Ops. 
k= 20
l= 20

print(id(k))
print(id(l))

print( k is l)
print(k is not l)

v = 20
print(id(v))


#String 
std_name = 'Ali'
mark = 18

print( std_name, mark)

std_name_0 = "Ali_0"
txt = """ 
txt 
txt 
txt
"""

#print
text_sample = 'this is about computer vision!'
num = 18
mark_2 = 19


print('Marks of Ali is {0}, {1} '.format(num, mark_2))



#Input
age = input('enter your age: ')

print(type(age))

age_2 = float(age) + 15
print(age_2) 

###
r = 20
p = 20
print(id(r))
print(id(p))

p = p + 1

print(id(p))








