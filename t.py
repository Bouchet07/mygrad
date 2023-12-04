import mygrad

a = mygrad.Tensor(2, requires_grad=True)
b = mygrad.Tensor(3, requires_grad=True)
c = a*b
d = c*a
e = c*d

e.backward()