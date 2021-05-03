x = 1


def f():
    y, z = 2, 3

    def g():
        print(z)
        # z = 3

    return g


g = f()

g()
