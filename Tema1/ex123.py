# 40% generat cu ChatGPT
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
