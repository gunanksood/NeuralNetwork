x_old = 0
x_new = 6
count = 0
gamma = 0.01

precision = 0.00001

def df(x):
    y = 4 * x ** 3 - 9 * x ** 2
    return y

while abs(x_new - x_old) > precision:
    x_old = x_new
    count = count + 1
    x_new += -gamma *df(x_old)
    print ("x_old = " + str(x_old) + " x_new = " + str(x_new))

print ("Local minima is " + str(x_new) + " with count = " + str(count))

