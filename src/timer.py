from time import perf_counter
from ansi import ANSI

def timer(func):    

    def wrapper(*args, **kwargs):
        init = perf_counter()
        result = func(*args, **kwargs)
        end  = perf_counter()
        
        a = ANSI()
        print (
            f'The method {a.b}{a.cyan} {func.__name__}{a.res}  '
            f'took {a.b}{a.green}{(end - init):.4e}{a.res} seconds to run'
        )
        return result
    return wrapper