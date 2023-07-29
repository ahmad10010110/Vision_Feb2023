#Control flow & iteration

#if
age = int(input('enter your age: '))

money_unit = 'tomans'

if age <= 5:
    ticket_price = 8000
elif age <= 10:
    ticket_price = 10000
else:
    ticket_price = 20000

print(' you should pay {:0.1f} {}'.format(ticket_price, money_unit))


#Ternary Ops. 
age = 42
ticket_price = 8000 if age <= 5 else 20000
print(ticket_price)



#while 
max_inp_frm_per_sec = 5
counter = 0

while counter < max_inp_frm_per_sec:
    print(counter)
    counter += 1
else:
    print('loop finished!')


#for range!

for element in range(2, 12, 2):
    print(element)


letters = ['a', 'b', 'c', 'd']
for letter in letters:
    print(letter)
else:
    print('finally completed!')

#we prefer to use Numpy array  instead of in built python list.



#break
#100 / [1, 8, 12, ..., 458, 0]

n = 5
while n > 0:
    n -= 1
    if n == 2:
        break 
    print(n)
    
print('loop ended!')

#continue

n = 5
while n > 0:
    n -= 1
    if n == 2:
        continue
    print(n)
    
print('loop ended!')

























#

