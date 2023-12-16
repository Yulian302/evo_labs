import random
from collections import deque
from gen_funcs import *
import numpy as np


class GenExprTree:
    def __init__(self,
                 funcs: dict[str, Func],
                 terminal_symbols_: list[str],
                 head_len: int,
                 arity: int,
                 n_genes: int,
                 linker_: TreeLinker,
                 mut_prob: float,
                 move_max_len: int,
                 move_is_elements_prob: float,
                 move_ris_elements_prob: float):
        self.functions = funcs
        self.num_genes = n_genes
        self.mutation_prob = mut_prob
        self.terminals = set(terminal_symbols_)
        self.head_len = head_len
        self.move_max_len = move_max_len
        self.tail_len = head_len * (arity - 1) + 1
        self.gene_len = self.head_len + self.tail_len
        self._terminals = terminal_symbols_
        self.move_is_elements_prob = move_is_elements_prob
        self.move_ris_elements_prob = move_ris_elements_prob
        self.linker = linker_
        self._functions = list(funcs.keys())
        self._functions_terminals = self._functions + self._terminals
        self._validation_terminal_values = {self._terminals[i]: 0 for i in range(0, len(self._terminals))}

    def _eval_k_expression(self, k_expression: str, terminal_values: dict[str, float]) -> float:
        try:
            k_expr_valid = self.cycle_evaluation(k_expression)
            queue = deque()
            for prim in list(k_expr_valid):
                if prim in self.terminals:
                    queue.append(terminal_values[prim])
                if prim in self.functions.keys():
                    num_args = self.functions[prim].get_args_num
                    operands = []
                    for _ in range(0, num_args):
                        operands.append(queue.popleft())
                    operands.reverse()
                    queue.append(self.functions[prim].evaluate(operands))
            return queue.pop()
        except:
            return 0

    def eval_expr(self, gen_expr, terminals_):
        if self.linker == TreeLinker.SUM:
            return sum([self._eval_k_expression(k_expr, terminals_) for k_expr in gen_expr])
        if self.linker == TreeLinker.MAX:
            return max([self._eval_k_expression(k_expr, terminals_) for k_expr in gen_expr])
        if self.linker == TreeLinker.MIN:
            return max([self._eval_k_expression(k_expr, terminals_) for k_expr in gen_expr])

    def _rnd_gen_expr(self):
        return [random.choice(list(self._functions))
                + ''.join(random.choice(self._functions_terminals) for _ in range(1, self.head_len))
                + ''.join(random.choice(self._terminals) for _ in range(0, self.tail_len))
                for _ in range(0, self.num_genes)]

    def generate_genexpr(self):
        while True:
            expr = self._rnd_gen_expr()
            try:
                self.eval_expr(expr, self._validation_terminal_values)
                return expr
            except:
                continue

    def evaluate_fitness(self, genetic_expression: list[str], value_pairs):
        score = 0
        for value_pair in value_pairs:
            terminal_values = {self._terminals[i]: value_pair[0][i] for i in range(0, len(value_pair[0]))}
            score += abs(self.eval_expr(genetic_expression, terminal_values) - value_pair[1])
        return 1 / score if score != 0 else 0

    def eval_fitness_scores(self,
                            genetic_expressions: list[list[str]],
                            value_pairs):
        return [self.evaluate_fitness(ge, value_pairs)
                for ge in genetic_expressions]

    def cycle_evaluation(self, k_expression):
        k_expr_valid = deque()
        i, num_args = 0, 0
        while i < len(k_expression):
            prim = k_expression[i]
            if prim in self.functions.keys():
                num_args += self.functions[prim].get_args_num
            k_expr_valid.appendleft(prim)
            if num_args == 0:
                break
            else:
                num_args -= 1
            i += 1
        return k_expr_valid

    # mutation
    def _mutation(self,
                  gen_expr: list[str]):
        for i in range(0, self.num_genes):
            gen = gen_expr[i]
            index = random.randint(0, self.gene_len - 1)
            if index == 0:
                gen_expr[i] = random.choice(self._functions) + gen[index + 1:]
            elif index < self.head_len:
                gen_expr[i] = gen[:index] + random.choice(self._functions_terminals) + gen[index + 1:]
            else:
                gen_expr[i] = gen[:index] + random.choice(self._terminals) + gen[index + 1:]

    # IS moving (ІС переміщення)
    def _is_element_moving(self,
                           genetic_expression: list[str]):
        f_gen_idx = random.randint(0, self.num_genes - 1)
        s_gen_idx = random.randint(0, self.num_genes - 1)
        while f_gen_idx == s_gen_idx:
            s_gen_idx = random.randint(0, self.num_genes - 1)
        f_gene = genetic_expression[f_gen_idx]
        s_gene = genetic_expression[s_gen_idx]
        sequence_len = random.randint(1, self.move_max_len)
        move_from = random.randint(1, self.gene_len - 1 - sequence_len)
        move_to = random.randint(1, self.head_len - 1 - sequence_len)
        s_gene = s_gene[:move_to] + f_gene[move_from:move_from + sequence_len] + s_gene[move_to + sequence_len:]
        genetic_expression[s_gen_idx] = s_gene

    # selecting gens due to their fitness
    @staticmethod
    def _selection(
            gen_expr: list[list[str]],
            scores: list[float],
            n_ges):
        return [random.choices(gen_expr, np.divide(scores, sum(scores)))[0]
                for _ in range(0, n_ges)]

    # RIS moving (RIS переміщення)
    def _ris_element_moving(self,
                            genetic_expression: list[str]):
        f_gen_idx = random.randint(0, self.num_genes - 1)
        s_gen_idx = random.randint(0, self.num_genes - 1)
        while f_gen_idx == s_gen_idx:
            s_gen_idx = random.randint(0, len(genetic_expression) - 1)
        f_gen = genetic_expression[f_gen_idx]
        s_gen = genetic_expression[s_gen_idx]
        sequence_len = random.randint(1, self.move_max_len)
        move_from = random.randint(1, self.head_len - 1 - sequence_len)
        while f_gen[move_from] not in self.functions:
            move_from = random.randint(1, self.head_len - 1 - sequence_len)
        s_k_expr = f_gen[move_from:move_from + sequence_len] + s_gen[sequence_len:]
        genetic_expression[s_gen_idx] = s_k_expr

    # one point recombination
    def _one_point_recomb(self,
                          f_ge: list[str],
                          s_ge: list[str]):
        _r_f_ge, _r_s_ge = [], []
        f_ge_str = ''.join(f_ge)
        s_ge_str = ''.join(s_ge)
        idx = random.randint(0, self.num_genes * self.gene_len - 1)
        r_f_ge_str = f_ge_str[:idx] + s_ge_str[idx:]
        r_s_ge_str = s_ge_str[:idx] + f_ge_str[idx:]
        i = self.gene_len
        while i < self.num_genes * self.gene_len + 1:
            _r_f_ge.append(r_f_ge_str[i - self.gene_len:i])
            _r_s_ge.append(r_s_ge_str[i - self.gene_len:i])
            i += self.gene_len
        return [_r_f_ge, _r_s_ge]

    # two point recombination
    def _two_points_recomb(self,
                           f_ge: list[str],
                           s_ge: list[str]):
        r_f_ge, r_s_ge = [], []
        f_ge_str = ''.join(f_ge)
        s_ge_str = ''.join(s_ge)
        f_index = random.randint(0, self.num_genes * self.gene_len - 2)
        s_index = random.randint(f_index + 1, self.num_genes * self.gene_len - 1)
        r_f_ge_str = f_ge_str[:f_index] + s_ge_str[f_index:s_index] + f_ge_str[s_index:]
        r_s_ge_str = s_ge_str[:f_index] + f_ge_str[f_index:s_index] + s_ge_str[s_index:]
        i = self.gene_len
        while i < self.num_genes * self.gene_len + 1:
            r_f_ge.append(r_f_ge_str[i - self.gene_len:i])
            r_s_ge.append(r_s_ge_str[i - self.gene_len:i])
            i += self.gene_len
        return [r_f_ge, r_s_ge]

    # full recombination (or gen recombination)
    def _recomb(self,
                f_ge: list[str],
                s_ge: list[str]):
        gene_index = random.randint(0, self.num_genes - 1)
        r_f_ge = f_ge.copy()
        r_s_ge = s_ge.copy()
        r_f_ge[gene_index] = s_ge[gene_index]
        r_s_ge[gene_index] = f_ge[gene_index]
        return [r_f_ge, r_s_ge]

    # training
    def train_(self, a_F, n_gen_expr, epochs):
        gen_expressions = [self.generate_genexpr() for _ in range(0, n_gen_expr)]
        fitness_scores = self.eval_fitness_scores(gen_expressions, a_F)
        best = gen_expressions[np.argmax(fitness_scores)]
        for e in range(0, epochs):
            if random.random() < self.move_ris_elements_prob:
                self._ris_element_moving(random.choice(gen_expressions))
            if random.random() < self.mutation_prob:
                self._mutation(random.choice(gen_expressions))
            if random.random() < self.move_is_elements_prob:
                self._is_element_moving(random.choice(gen_expressions))
            parents = random.sample(gen_expressions, 2)
            offsprings_ = self._one_point_recomb(parents[0], parents[1])
            gen_expressions.extend(offsprings_)
            parents = random.sample(gen_expressions, 2)
            offsprings_ = self._recomb(parents[0], parents[1])
            gen_expressions.extend(offsprings_)
            parents = random.sample(gen_expressions, 2)
            offsprings_ = self._two_points_recomb(parents[0], parents[1])
            gen_expressions.extend(offsprings_)
            fitness_scores = self.eval_fitness_scores(gen_expressions, a_F)
            gen_expressions = GenExprTree._selection(gen_expressions, fitness_scores, n_gen_expr)
            best_result = max(gen_expressions, key=lambda ge: self.evaluate_fitness(ge, a_F))
            if self.evaluate_fitness(best_result, a_F) > self.evaluate_fitness(best, a_F):
                best = best_result
                error = 0
                for vp in a_F:
                    prediction = self.eval_expr(best, {'a': vp[0][0]})
                    error += abs(vp[1] - prediction)
                print(f'Error on epoch {e}: {error / len(a_F)}')
        return best

    def restore(self,
                genetic_expression: list[str]):
        restored = []
        for k_expression in genetic_expression:
            k_expression_valid = self.cycle_evaluation(k_expression)
            restored.append(''.join(list(reversed([*k_expression_valid]))))
        return {'expression': restored, 'linker': self.linker}
