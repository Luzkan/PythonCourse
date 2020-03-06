from sympy import sympify
from sympy.plotting import textplot
f = sympify(input('Podaj funkcje f(x) = '))
a = sympify(input('Podaj początek przedziału a = '))
b = sympify(input('Podaj koniec przedziału b = '))
textplot(f, a, b)   # ex: sin(x), -pi, pi

# Wcześniej (poza próbami z numpy, gnuplotlib itd.)
# import: Symbol, S, symbols, pi, sin 
# x = Symbol('x')
# pi = Symbol('pi')
# textplot(sin(x), -S.Pi, S.Pi)