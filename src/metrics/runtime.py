import time

class Runtime:

    @staticmethod
    def measure(function,*args,**kwargs):
        start=time.perf_counter()
        output=function(*args,**kwargs)
        end=time.perf_counter()
        return output, end-start