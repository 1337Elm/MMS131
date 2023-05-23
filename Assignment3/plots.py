#3.1 a) plots

import matplotlib.pyplot as plt

x1 = [0,1]
y1 = [0,1]

x2 = [0,1]
y2 = [1,0]
fig1 = plt.figure()
plt.scatter(x1,y1,marker  = "o", s = 30, color = "b")
plt.scatter(x2,y2,marker  = "o", s = 30, color = "r")
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.title("Linearly inseparable boolean function")

fig2 = plt.figure()
plt.scatter(x1,y1,marker  = "o", s = 30, color = "r")
plt.scatter(x2,y2,marker  = "o", s = 30, color = "b")
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.title("Linearly inseparable boolean function")
plt.show()