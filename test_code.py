from py_vetor import PyVector
ryan = [0.1, 1.1, 2.1, 3, 4.1, 5.1, 6.1, 7.1, 8.1, 9.1]
travis = [0.1, 1.1, 2.1, 3.0, 90, 5.1, 6.1, 7.1, 8.1, 9.1]
scott = [2, 3, 4]

r = PyVector(ryan, typesafe=True)
r[r == travis]

long_r = [x - 80.75 for x in range(100)]
t = PyVector(long_r, typesafe=True)

# import numpy as np
# vvv = np.array(long_r)



# addition
4.0 + t 
t + 4.0 
ryan + r 
r + travis 
r + r 


class Fake():
	""" Iterable vector with optional type safety """
	_dtype = None
	_underlying = None

	def __init__(self, initial=[], dtype=None):
		""" Create a new PyVector from an initial list """
		self._underlying = (x for x in initial)
		self._dtype = dtype


from py_vetor import PyVector
a = PyVector([1, 2, 0, 1], name = 'a')
b = PyVector([0, -1, -1, 1], name = 'b')
c = PyVector([3, 1, -2, 2], name = 'c')
big_a = PyVector([a, b, c])

d = PyVector([1, 2, 4])
e = PyVector([2, 3, 2])
f = PyVector([1, 1, 2])
big_b = PyVector([d, e, f])
big_a @ big_b


big_c = PyVector([
	PyVector([1, 0, 1]),
	PyVector([2, 1, 1]),
	PyVector([0, 1, 1]),
	PyVector([1, 1, 2])])
from py_vetor import PyTable


c = [1, 2, 3, 4, 5]
c[-7:-4] = [3, 2, 1]



from py_vetor import PyVector, _PyFloat
import timeit
import numpy as np
import math
import random
import statistics

rng = [x for x in range(15000)]
rand_t = tuple(random.gauss(3, 2) for x in rng)
rand_l = [x for x in rand_t]
rand_p = PyVector(rand_t)
rand_r = _PyFloat(rand_t)
rand_n = np.array([x for x in rand_t])


print(f"rand_t: {timeit.timeit(lambda: sum(rand_t), number = 1000)}")
print(f"rand_l: {timeit.timeit(lambda: sum(rand_l), number = 1000)}")
print(f"rand_p: {timeit.timeit(lambda: sum(rand_p), number = 1000)}")
print(f"rand_r: {timeit.timeit(lambda: sum(rand_r), number = 1000)}")
print(f"rand_n: {timeit.timeit(lambda: sum(rand_n), number = 1000)}")
print(f"vector rand_n: {timeit.timeit(lambda: rand_n.sum(), number = 1000)}")


print(f"numpy mean: {timeit.timeit(lambda: rand_n.mean(), number = 1000)}")
print(f"statistics.mean: {timeit.timeit(lambda: statistics.mean(rand_r), number = 1000)}")
print(f"sum / len: {timeit.timeit(lambda: sum(rand_r) / len(rand_r), number = 1000)}")

print(f"numpy std: {timeit.timeit(lambda: rand_n.std(), number = 1000)}")
print(f"statistics.stdev: {timeit.timeit(lambda: statistics.stdev(rand_r), number = 1000)}")
print(f"explcit: {timeit.timeit(lambda: (sum((rand_r - (sum(rand_r) / len(rand_r)))**2.0))**0.5, number = 1000)}")