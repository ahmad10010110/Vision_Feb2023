class Person:
    def __init__(self, name, birth_date):
        self.name = name
        self.birth_date = birth_date



class Employee(Person):
    def __init__(self, name, birth_date, position):
        super().__init__(name, birth_date)
        self.position = position




emp_01 = Employee('ali', '1998-08-09', 'project manager')


print(emp_01.name)
print(emp_01.birth_date)
print(emp_01.position)
