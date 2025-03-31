from py_vetor import PyVector
d = PyVector([1, 2, 3])
e = PyVector([4, 5, 6])
f = PyVector([7, 6, 9])
big_b = PyVector([d, e, f])

norm_mtx = [sum(v**2.)**0.5 for v in big_b._underlying]
mtx_p = PyVector([x / n for n, x in zip(norm_mtx, big_b._underlying, strict=True)])
mtx_q = PyVector([x / n for n, x in zip(norm_mtx, big_b._underlying, strict=True)])





def some_stuff(mtx):
    norm_mtx = [sum(v**2.)**0.5 for v in mtx._underlying]

    mtx_p = PyVector([x / n for n, x in zip(norm_mtx, mtx._underlying, strict=True)])
    mtx_q = PyVector([x / n for n, x in zip(norm_mtx, mtx._underlying, strict=True)])
    mtx_q = PyVector(tuple(v for v in sorted(mtx_q._underlying, key=lambda x: norm(x))))
    mtx_q = orthogonalize(mtx_q)

    tmp = mtx_p @ mtx_q
    mtx_q = mtx_p @ mtx_q
    mtx_q = PyVector(tuple(v for v in sorted(mtx_q._underlying, key=lambda x: norm(x))))
    mtx_q = orthogonalize(mtx_q)
    mtx_q - tmp

from py_vetor import PyVector

def norm(v):
    return sum(v**2.)**.5

def orthogonalize(mtx):
    # n, k = size(mtx)
    # U = zeros(n,k);
    # U(:,1) = V(:,1) / norm(V(:,1)); 
    gs = [x for x in mtx._underlying]
    for ii in range(1, len(gs)):
        for jj in range(ii):
            gs[ii] -= sum(gs[jj] * gs[ii]) * gs[jj]
        gs[ii] = gs[ii]/norm(gs[ii])

    return PyVector(gs)

d = PyVector([1, 2, 3])
e = PyVector([4, 5, 6])
f = PyVector([7, 6, 9])
mtx = PyVector([d, e, f])

norm_mtx = [sum(v**2.)**0.5 for v in mtx._underlying]
mtx_p = PyVector([x / n for n, x in zip(norm_mtx, mtx._underlying, strict=True)])
mtx_q = PyVector([x / n for n, x in zip(norm_mtx, mtx._underlying, strict=True)])
#mtx_q = PyVector(tuple(v for v in sorted(mtx_q._underlying, key=lambda x: norm(x))))
mtx_j = orthogonalize(mtx_q)

a = mtx_j._underlying[0]
b = mtx_j._underlying[1]
c = mtx_j._underlying[2]
print(sum(a*b))
print(sum(a*c))
print(sum(b*c))


for ii in range(20):
	mtx_q = mtx_p @ mtx_j
	a = mtx_q._underlying[0]
	b = mtx_q._underlying[1]
	c = mtx_q._underlying[2]
	print(f"{norm(a)}, {norm(b)} {norm(c)}")
	# normalize q
	mtx_q = PyVector([x * (-1 if sum(x) < 0.0 else 1)/ norm(x) for x in mtx_q._underlying])
	# mtx_q = PyVector(tuple(v/norm(v) for v in sorted(mtx_q._underlying, key=lambda x: norm(x))))
	# orthogonalize q
	mtx_j = orthogonalize(mtx_q)


def argsort(seq):
  """
  Returns the indices that would sort a sequence.

  Args:
    seq: A list or tuple of comparable elements.

  Returns:
    A list of integers representing the sorted indices of the input sequence.
  """
  return [i for i, _ in sorted(enumerate(seq), key=lambda x: x[1])]