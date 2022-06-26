class Test:
    # TEMP = "1"

    @property
    def temp(self):
        return "22"

    @temp.setter
    def temp(self, value):
        print("Try to set temp")
        raise AttributeError

    # TEMP = property(get_temp, set_temp)

t = Test()
# t.TEMP = "2"
print(t.temp)
