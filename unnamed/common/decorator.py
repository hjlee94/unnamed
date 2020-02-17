
def abstract(func):
    def wrapper(*args):
        print('%s() is abstract. implement it first'%func.__name__)
        func(*args)

    return wrapper