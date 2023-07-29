#List
mt_lst = []
print(len(mt_lst))


lst = [1, 2, 3, 4, 12, 18]
lst.append(181)
print(lst)


#print(len(lst))


#print(lst[3])


print(lst[-1])
print(lst[-2])


#mutate!
lst[3] = 888
print(lst)

lst_sec = [ 45, 46, 47, 48]
lst_sec.insert(2, 123)
print(lst_sec)


#remove particular element using index!
del lst_sec[0]
print(lst_sec)


#remove particular element using value!
lst_sec.remove(47)
print(lst_sec)


#remove particular index while return the value!
lst_thd = [2, 4, 6, 8, 10]
elmt = lst_thd.pop(1)
elmt_2 = elmt + 12

print(lst_thd)
print(elmt)

#Tuple
rgb = ('red', 'green', 'blue')
print(rgb[0])

rgb[1] = 'yellow'

flower = ('Tulip', )
print(type(flower))


marks_one = ('A', 'B', 'C', 'D')
marks_two = ('w', 'x', 'y', 'z')

#print(marks)

all_marks = marks_one + marks_two

print(all_marks)


#Dictionary
#Key: Value!
#Keys are unique!
mt_dictionary = {}
print(type(mt_dictionary))


user ={'first_name':'Ali', 
       'last_name': 'Farajnia', 
       'age': 25}


#active!
user['active'] = True

print(user)

print(user['first_name'])
print(user['last_name'])


age = user.get('age')
print(age)


del user['active']
print(user)


for key in user:
    print(key)



for value in user.values():
    print(value)


#Set
mt_set = {}
print(type(mt_set))

mt_set_sec = set()
courses_set = set(['py','py', 'dip', 'deep', 'vision'])
print(courses_set)
print(type(courses_set))


size_set = len(courses_set)
print(size_set)

for element in courses_set:
    print(element)

#remove
#discard
#pop

#courses_set.remove('farsi')
courses_set.discard('farsi')


print(courses_set)
print(courses_set.pop())
print(courses_set)


