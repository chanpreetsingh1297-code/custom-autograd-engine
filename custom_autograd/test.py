
# testing the scaler autograd(basic operations testing)
a = Value(2.0)
b = Value(5.0)

c = a + b
print(f"c={c}")
print(f"parents={c._prev}")

d = a + 2.0
print(f"d={d}")

e = 10.0 + b
print(f"e={e}")

print("Simulating the backward pass")
c.grad = 1.0
c.backward()

print(f"Gradient of a={a.grad}")
print(f"fGradient of b={b.grad}")
