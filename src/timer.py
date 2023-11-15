from time import perf_counter
from ansi import BOLD, RESET, CYAN, GREEN

def timer(func):    

    def wrapper(*args, **kwargs):
        init = perf_counter()
        result = func(*args, **kwargs)
        end  = perf_counter()
        print (
            f'The method {BOLD}{CYAN} {func.__name__}{RESET}  '
            f'took {BOLD}{GREEN}{(end - init):.4e}{RESET} seconds to run'
        )
        return result
    return wrapper