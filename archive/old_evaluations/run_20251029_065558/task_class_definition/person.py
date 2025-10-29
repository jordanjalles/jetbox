class Person:
    def __init__(self, name: str, age: int, email: str):
        self.name = name
        self.age = age
        self.email = email

    def __str__(self):
        return f"Person(name={self.name!r}, age={self.age}, email={self.email!r})"

    def is_adult(self) -> bool:
        return self.age >= 18
