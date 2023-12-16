from gen_programming_algo import Funcs, GenExprTree, TreeLinker

FUNCS = {
    '+': Funcs.ADD,
    '-': Funcs.SUBTRACT,
    'P': Funcs.POWER,
    'U': Funcs.UNARY_SUBTRACTION,
    '*': Funcs.MULTIPLY,
    '/': Funcs.DIVIDE,
}

TERMINALS = ['a']

a_y = [
    [[3.4], 2.64],
    [[5.4], 65.04],
    [[6.7], 122.76],
    [[8.2], 206.16],
    [[9.12], 266.2176],
    [[10.25], 349.25],
    [[12.34], 529.7424],
    [[21.43], 1721.26],
    [[23.76], 2133.11],
    [[25.32], 2433.13]
]

gen_expr_tree = GenExprTree(FUNCS,
                            TERMINALS,
                            16,
                            2,
                            2,
                            TreeLinker.SUM,
                            0.6,
                            3,
                            0.1,
                            0.1
                            )

epochs = 1100
best_ge = gen_expr_tree.train_(a_y, 50, epochs)

points = [10, 20, 30]
for val in points:
    print(f"F({val})={gen_expr_tree.eval_expr(best_ge, {'a': val})}")

expression = gen_expr_tree.restore(best_ge)
print(expression)
print(best_ge)
