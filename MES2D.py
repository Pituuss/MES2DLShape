import numpy as np
import matplotlib.pyplot as plt
import math
from fractions import Fraction as Fract
import mpl_toolkits.mplot3d


def fi1(x1, x2, b1, b2, a):
    return (1 - (x1 - b1) / a) * (1 - (x2 - b2) / a)


def fi2(x1, x2, b1, b2, a):
    return ((x1 - b1) / a) * (1 - (x2 - b2) / a)


def fi3(x1, x2, b1, b2, a):
    return ((x1 - b1) / a) * ((x2 - b2) / a)


def fi4(x1, x2, b1, b2, a):
    return (1 - (x1 - b1) / a) * ((x2 - b2) / a)


def check_if_neumann(points):
    is_neumann = True
    for point in points:
        xc, yc = point
        check_elem = (xc == -1 and 0 <= yc <= 1) or (xc == 1 and -1 <= yc <= 1) or (
                -1 <= xc <= 1 and (yc == 1 or yc == -1))
        is_neumann = is_neumann and check_elem
    return is_neumann


def check_if_dirichlet(point):
    xc, yc = point
    check_elem = (-1 <= xc <= 0 and yc == 0) or (xc == 0 and -1 <= yc <= 0)
    return check_elem


def get_next_cords(current_pos, ind, a):
    xp, yp = current_pos
    return {
        0: (xp + a, yp),
        1: (xp, yp + a),
        2: (xp - a, yp),
        3: (xp, yp - a)
    }.get(ind)


def cords(pos, a):
    coordinates = [pos]
    tmp_pos = pos
    for i in range(3):
        tmp_pos = get_next_cords(tmp_pos, i, a)
        coordinates.append(tmp_pos)
    return coordinates


def is_in_area(x, y):
    return (0 <= y <= 1 and -1 <= x < 1) or (-1 <= y <= 0 <= x < 1)


def create_mesh_mapping_nodes(a):
    x0 = -1
    y0 = 1
    count = 0
    mapping = {}
    elements = []
    nodes = []
    while (not math.isclose(x0, 1, rel_tol=1e-5) and not math.isclose(y0, -1, rel_tol=1e-5)) or (
            math.isclose(x0, 1, rel_tol=1e-5) and not math.isclose(y0, -1, rel_tol=1e-5)) or (
            not math.isclose(x0, 1, rel_tol=1e-5) and math.isclose(y0, -1, rel_tol=1e-5)):
        if 0 < y0 <= 1:
            if x0 <= 1:
                if is_in_area(x0, y0 - a):
                    elements.append(Elem(Fract(x0), y0 - a, a))
                mapping[(x0, y0)] = count
                nodes.append((x0, y0))
                count += 1
                x0 += a
            else:
                x0 = -1
                y0 -= a
        else:
            if x0 <= 1:
                if is_in_area(x0, y0 - a):
                    elements.append(Elem(Fract(x0), y0 - a, a))
                mapping[(x0, y0)] = count
                nodes.append((x0, y0))
                count += 1
                x0 += a
            else:
                x0 = 0
                y0 -= a

    mapping[1, -1] = count
    return mapping, elements, nodes


class Elem(object):

    def __init__(self, b1, b2, a):
        self.b1 = b1
        self.b2 = b2
        self.a = a
        self.cords = cords((self.b1, self.b2), self.a)

    def __repr__(self):
        return f"(x: {self.b1} y: {self.b2})"

    def get_pos(self):
        return self.b1, self.b2

    def get_cords(self):
        return self.cords

    def get_top_pos(self):
        return self.cords[2]


dx1 = [lambda a1: -1 / (2 * a1), lambda a1: 1 / (2 * a1), lambda a1: 1 / (2 * a1), lambda a1: -1 / (2 * a1)]
dx2 = [lambda a2: -1 / (2 * a2), lambda a2: -1 / (2 * a2), lambda a2: 1 / (2 * a2), lambda a2: 1 / (2 * a2)]
fis = [fi1, fi2, fi3, fi4]


def b(i, j, a):
    return dx1[i](a) * dx1[j](a) * a * a + dx2[i](a) * dx2[j](a) * a * a


def g(r, phi):
    val = r ** 2 * np.sin(phi + (np.pi / 4)) ** 2
    return val ** (1 / 3)


def l_fun(xl, yl, fi_fun, b1, b2, a):
    r, phi = cart_to_polar(xl, yl)
    return g(r, phi) * fi_fun(xl, yl, b1, b2, a)


def cart_to_polar(xc, yc):
    rho = np.sqrt(xc ** 2 + yc ** 2)
    phi = np.arctan2(yc, xc)
    return rho, phi


# noinspection PyPep8Naming,PyPep8Naming,PyPep8Naming,PyPep8Naming,PyPep8Naming,PyPep8Naming
def solver():
    a1 = int(input('enter size(1/a)> '))
    a = Fract(1, a1)
    mapping, elements, nodes = create_mesh_mapping_nodes(a)
    nodes_number = int(3 * a1 * a1 + 2 * (a1 + a1) + 1)
    B = np.zeros((nodes_number, nodes_number))
    L = np.zeros((nodes_number, 1))

    for elem in elements:
        position = elem.get_cords()
        (x, y) = elem.get_pos()
        for index in range(4):
            pos_1 = position[index]
            pos_2 = position[(index + 1) % 4]
            (x0, y0) = pos_1
            (xn, yn) = pos_2
            if check_if_neumann([pos_1, pos_2]):
                xh = float(xn / 2 + x0 / 2)
                yh = float(yn / 2 + y0 / 2)
                L[mapping[position[0]]] += l_fun(xh, yh, fis[0], x, y, a)
                L[mapping[position[1]]] += l_fun(xh, yh, fis[1], x, y, a)
                L[mapping[position[2]]] += l_fun(xh, yh, fis[2], x, y, a)
                L[mapping[position[3]]] += l_fun(xh, yh, fis[3], x, y, a)
            for index2 in range(4):
                B[mapping.get(position[index])][mapping.get(position[index2])] += b(index, index2, a)

    for node in mapping:
        if check_if_dirichlet(node):
            B[mapping[node]] = 0
            L[mapping[node]] = 0
            B[mapping[node]][mapping[node]] = 1

    A = np.linalg.solve(B, L)

    n = 30
    xs = np.linspace(-1, 1, n)
    ys = np.linspace(-1, 1, n)
    Z = np.zeros((len(xs), len(ys)))

    for (i, x) in enumerate(xs):
        for (j, y) in enumerate(ys):
            match_elem = None
            for elem in elements:
                (x0, y0) = elem.get_pos()
                (x1, y1) = elem.get_top_pos()
                if (x0 <= Fract.from_float(x) <= x1) and (y0 <= Fract.from_float(y) <= y1):
                    match_elem = elem
                    break
            if match_elem is None:
                continue
            (bx, by) = match_elem.get_pos()
            position = match_elem.get_cords()
            Z[i, j] += A[mapping.get(position[0])] * fis[0](x, y, bx, by, a)  # phi0
            Z[i, j] += A[mapping.get(position[1])] * fis[1](x, y, bx, by, a)  # phi1
            Z[i, j] += A[mapping.get(position[2])] * fis[2](x, y, bx, by, a)  # phi2
            Z[i, j] += A[mapping.get(position[3])] * fis[3](x, y, bx, by, a)  # phi3
    fig = plt.figure(figsize=(15, 10))
    ax = fig.gca(projection='3d')

    X, Y = np.meshgrid(xs, ys)
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=plt.get_cmap('jet'))
    ax.view_init(45, 240)

    plt.show()


solver()
