class Parent():
    def __init__(self):
        self._initial = 5

    @property
    def initial(self):
        return self._initial

    @initial.setter
    def initial(self, value):
        self._initial = value

class Child(Parent):
    def __init__(self):
        Parent.__init__(self)
        self._goal = 10
        Parent.initial.fset(self, 4)
#        super(Child, self.__class__).initial.fset(self, 4)
#    @property
#    def initial(self):
#        return super().initial

#    @initial.setter    
#    def initial(self, value):
#        super(Child, self.__class__).initial.fset(self, value)

c = Child()
print(c._initial)



