import time


def time_it(label, fn):
    """
    Wrap a function with a timer

    label: string label or method that creates a label from the result of calling fn
    fn: the function to wrap
    """

    def wrapper(*args):
        start_time = time.perf_counter()
        result = fn(*args)
        end_time = time.perf_counter()
        label_str = str(label(result)) if callable(label) else str(label)

        print("[Timer]: {} took {:.3f}s".format(label_str, end_time - start_time))
        return result

    return wrapper