from dataclasses import dataclass


@dataclass
class Class1:
    name: str = "foo"


@dataclass(slots=True)
class Class2:
    name: str = "bar"


if __name__ == '__main__':
    c1 = Class1("n1")
    c1.x = 1
    print(c1.x) # 여기선 뭐라 지적하지만 잘 동작된다.

    c2 = class2()
    c2.x = 1
    print(c2.x) # 여긴 안된다.
