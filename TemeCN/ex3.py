import math
import random
import time

def verify_tg(a):
    if a > math.pi / 2 or a < -math.pi / 2:
        k = round(a / math.pi)

        a_redus = a - k * math.pi

        if math.isclose(abs(a_redus), math.pi / 2, abs_tol=1e-9):
            raise ValueError(f"Tangenta nu e definita pt {a} (multiplu de pi/2)")

        return a_redus
    else:
        return a

def my_tan_lentz(x, eps=1e-12):
    x = verify_tg(x)

    if x < 0:
        return -my_tan_lentz(-x, eps)

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

xs = [random.uniform(-math.pi/2 + 1e-6, math.pi/2 - 1e-6) for _ in range(10000)]
start = time.time()

error_lentz = []

for x in xs:
    real = math.tan(x)
    approx = my_tan_lentz(x)

    error_lentz.append(abs(real - approx))

time_lentz = time.time() - start
print("Eroare medie Lentz:", sum(error_lentz) / len(error_lentz))
print("Eroare maxima Lentz:", max (error_lentz))
print("Timp executie Lentz:", time_lentz)

#aproximare folosind polinoame
def my_tan_polinom(x):
    x = verify_tg(x)
    c1 = 0.33333333333333333 # 1/3
    c2 = 0.13333333333333333 # 2/15
    c3 = 0.053968253968254 # 17/315
    c4 = 0.0218694885361552 # 62/2835

    if x < 0:
        return -my_tan_polinom(-x)

    if x > math.pi / 4:
        return 1 / my_tan_polinom(math.pi / 2 - x)

    x_2 = x * x # x ^ 2
    x_3 = x_2 * x # x ^ 3
    x_4 = x_2 * x_2 # x ^ 4
    x_6 = x_4 * x_2 # x ^ 6

    return x + x_3 * (c1 + c2 * x_2 + c3 * x_4 + c4 * x_6)

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

print(my_tan_lentz(math.pi / 4))
print(my_tan_polinom(math.pi / 4))
print(math.tan(math.pi / 4))