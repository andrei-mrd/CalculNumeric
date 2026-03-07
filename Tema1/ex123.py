#ex1
print("ex1 : ")
m = 1
while 1.0 + 10**(-m) != 1.0:
    m += 1
m = m - 1
print(m)

#ex2
print("ex 2 : ")
#verificare asociativitate adunare
print("verificare asociativitate adunare :")
x = 1.0
y = (10**(-m))/10
z = (10**(-m))/10

if (x + y) + z == x + (y + z):
    print("True")
else:
    print("False")

#gasire x,y,z pt care Xc e neasociativa
print("gasire x,y,z pt care Xc e neasociativa :")

x = 1e-200
y = 1e-200
z = 1e200

if (x * y) * z == x * (y * z):
    print("True")
else:
    print("False")


print(x, y, z)
print((x * y) * z)
print(x * (y * z))

#ex3
#aproximare lentz
print("ex3 : ")
import math
import random
import time

def my_tan_lentz(x, eps=1e-12):
    mic = 1e-12

    b0 = 0.0
    f = b0
    if f == 0:
        f = mic
    C = f
    D = 0.0
    j = 1

    while True:

        if j == 1:
            a = x
            b = 1
        else:
            a = -x * x
            b = 2 * j - 1

        D = b + a * D
        if D == 0:
            D = mic

        C = b + a / C
        if C == 0:
            C = mic

        D = 1.0 / D
        delta = C * D
        f = delta * f

        if abs(delta - 1) < eps:
            break

        j += 1

    return f

xs = [random.uniform(-math.pi/2, math.pi/2) for _ in range(10000)]
start = time.time()

error_lentz = []

for x in xs:
    real = math.tan(x)
    approx = my_tan_lentz(x)

    error_lentz.append(abs(real - approx))

time_lentz = time.time() - start
print("Eroare medie Lentz:", sum(error_lentz) / len(error_lentz))
print("Eroare maxima Lentz:", max(error_lentz))
print("Timp executie Lentz:", time_lentz)

#aproximare folosind polinoame
def my_tan_polinom(x):
    c1 = 0.33333333333333333
    c2 = 0.13333333333333333
    c3 = 0.053968253968254
    c4 = 0.0218694885361552

    if x < 0:
        return -my_tan_polinom(-x)

    if x > math.pi / 4:
        return 1 / my_tan_polinom(math.pi / 2 - x)

    x2 = x * x
    x3 = x2 * x
    x5 = x3 * x2
    x7 = x5 * x2
    x9 = x7 * x2

    return x + c1 * x3 + c2 * x5 + c3 * x7 + c4 * x9

xs1 = [random.uniform(-math.pi/2 + 1e-6, math.pi/2 - 1e-6) for _ in range(10000)]
start = time.time()

error_poly = []

for x in xs1:
    real = math.tan(x)
    approx = my_tan_polinom(x)

    error_poly.append(abs(real - approx))

time_poly = time.time() - start
print("Eroare medie Polinom:", sum(error_poly)/len(error_poly))
print("Eroare maxima Polinom:", max(error_poly))
print("Timp executie Polinom:", time_poly)