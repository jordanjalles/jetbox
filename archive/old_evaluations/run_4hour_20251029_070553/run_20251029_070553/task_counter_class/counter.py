class Counter:
    def __init__(self, initial=0):
        self._value = initial

    def increment(self, amount=1):
        self._value += amount
        return self._value

    def decrement(self, amount=1):
        self._value -= amount
        return self._value

    def reset(self, value=0):
        self._value = value
        return self._value

    def get_value(self):
        return self._value
