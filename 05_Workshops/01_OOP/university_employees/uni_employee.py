#for univesity employees!

#1. Faculty
## first_name, last_name, enter_marks, study

#2. Student
## first_name, last_name, take_exam, study

#3.Staff
## first_name, last_name, register_students, study


class Employee:
    def __init__(self, first_name, last_name):
        self.first_name = first_name
        self.last_name = last_name
    
    
    def study(self):
        print('{} {} you have to study more'.format(self.first_name, self.last_name))

    

class Faculty(Employee):
    def __init__(self, first_name, last_name):
        super().__init__(first_name, last_name)
    
    def enter_marks(self):
        print('mr/miss {} {}, you have to enter the marks as soon as possible!'.format(self.first_name, self.last_name))
        
    def study(self):
        print('{} {} if study more, it will be better!'.format(self.first_name, self.last_name))

    
    
#deadline of entering the marks - current date < 0
faculty_1 = Faculty('Hassan', 'ShamaeiZadeh')
faculty_1.enter_marks()

faculty_2 = Faculty('Ali', 'Khosravi')
faculty_2.study()


class Student(Employee):
    def __init__(self, first_name, last_name):
        super().__init__(first_name, last_name)
    

    def take_exam(self):
        print('mr/miss {} {}, you have to be ready for the exam as soon as possible!'.format(self.first_name, self.last_name))
    
    def study(self):
        print('{} {} you have to study more, if not, you will be fired!'.format(self.first_name, self.last_name))

    
    



student_1 = Student('John', 'Smith')
student_2 = Student('Allen', 'Gordon')

student_1.take_exam()
student_2.study()



class Staff(Employee):
    def __init__(self, first_name, last_name):
        super().__init__(first_name, last_name)
    
    
    def register_students(self):
        print('mr/miss {} {}, register the students from tomorrow!'.format(self.first_name, self.last_name))
    



staff_1 = Staff('Zahra', 'Jamshidy')
staff_2 = Staff('Leila', 'Ahmadi')

staff_1.register_students()
staff_2.study()







