from time import perf_counter

def timer(func):

    def wrapper(*args, **kwargs):
        init = perf_counter()
        result = func(*args, **kwargs)
        end  = perf_counter()
        print(f'The method {func.__name__} took {(end - init):.4e} seconds to run')
        return result
    return wrapper