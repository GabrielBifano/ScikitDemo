from time import perf_counter

def timer(func):

    B = "\033[1m"
    R = "\033[0m"
    CYAN = "\033[36m"
    GREEN = "\033[92m"

    def wrapper(*args, **kwargs):
        init = perf_counter()
        result = func(*args, **kwargs)
        end  = perf_counter()
        print (
            f'The method {B}{CYAN} {func.__name__}{R}  '
            f'took {B}{GREEN}{(end - init):.4e}{R} seconds to run'
        )
        return result
    return wrapper