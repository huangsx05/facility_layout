from typing import Tuple, List

import pulp
from pulp import PULP_CBC_CMD, GUROBI_CMD

from model.abstract_model import AbstractModel


class SetCover(AbstractModel):
    def __init__(self, grid: Tuple[int, int], shapes: List[Tuple[int, int]],
                 upper_bound):
        self.grid = grid
        self.shapes = shapes
        self.upper_bound = upper_bound
        self.m = pulp.LpProblem('layout', pulp.LpMaximize)
        return

    def _set_iterables(self):
        # 每一个单元格
        self.cells = [(i, j) for i in range(self.grid[0])
                      for j in range(self.grid[1])]
        # candidate shape的最大集合，对应变量x
        self.extended_shapes = [(s, n) for s, _ in enumerate(self.shapes)
                                for n in range(self.upper_bound)] 
        # 每一个s的左上角点的所有可能位置
        self.cell_shapes = {
            s: [(i, j) for i in range(self.grid[0] - shape[0] + 1)
                for j in range(self.grid[1] - shape[1] + 1)]
            for s, shape in enumerate(self.shapes)
        }
        # 每一个(s, n)的可能位置(i, j)，对应变量y
        self.cell_shape_list = [(s, n, i, j) for (s, n) in self.extended_shapes
                                for (i, j) in self.cell_shapes[s]]
        # 形状s的参考点放置在单元格(k,l)时，会覆盖单元格(i,j)
        self.cell_neighbors = {(i, j): [(s, k, l)
                                        for s, shape in enumerate(self.shapes)
                                        for (k, l) in self.cell_shapes[s]
                                        if _is_valid(i, j, k, l, shape)]
                               for (i, j) in self.cells}

        return

    def _set_variables(self):
        # 是否被排上
        self.x = pulp.LpVariable.dicts('x',
                                       self.extended_shapes,
                                       cat=pulp.LpBinary)
        # 是否被排在某位置
        self.y = pulp.LpVariable.dicts('y',
                                       self.cell_shape_list,
                                       cat=pulp.LpBinary)
        return

    def _set_objective(self):
        # 数量最多
        self.m += pulp.lpSum(self.x[s, n] for (s, n) in self.extended_shapes)
        return

    def _set_constraints(self):
        # shape selection：保证某个形状使用的话，一定需要放置在矩形中
        # 如果(s, n)被排上，在它所有可能的放置位置中有且只有一个位置被选中
        # 如果没有排上，没有位置
        for (s, n) in self.extended_shapes:
            self.m += (self.x[s,
                              n] == pulp.lpSum(self.y[s, n, i, j]
                                               for (i,
                                                    j) in self.cell_shapes[s]),
                       f'x-y-{s}-{n}')
        # no conflict：保证任意单元格不会被两个形状重复覆盖
        # 对于每一个单元格(i,j), 覆盖单元格的形状最多只能有一个
        for (i, j), item in self.cell_neighbors.items():
            self.m += (pulp.lpSum(self.y[s, n, k, l] for s, k, l in item
                                  for n in range(self.upper_bound)) <= 1,
                       f'cover-{i}-{j}')
        # simple symmetry breaking on x_s
        # 如果n+1的s放置上去了，那么n的s也必然放置
        for s, _ in enumerate(self.shapes):
            for n in range(self.upper_bound - 1):
                self.m += (self.x[s, n] <= self.x[s, n + 1],
                           f'symmetry-{s}-{n}')
        return

    def _optimize(self):
        time_limit_in_seconds = 1 * 60 * 60
        # It takes CBC 20 minutes to solve the problem, take Gurobi 2 minutes
        self.m.solve(PULP_CBC_CMD(timeLimit=time_limit_in_seconds,
                                  gapRel=0.01))
        # self.m.solve(GUROBI_CMD(timeLimit=time_limit_in_seconds,
        #                         gapRel=0.01))
        return

    def _is_feasible(self):
        return True

    def _process_infeasible_case(self):
        return list(), list()

    def _post_process(self):
        blocks = list()
        for (s, n, i, j) in self.cell_shape_list:
            if self.y[s, n, i, j].value() > 0.9:
                shape = self.shapes[s]
                blocks.append([[i, j], [i + shape[0], j],
                               [i + shape[0], j + shape[1]], [i, j + shape[1]],
                               [i, j]])
        return blocks, list()


def _is_valid(i, j, k, l, shape):
    return (0 <= i - k <= shape[0] - 1) and (0 <= j - l <= shape[1] - 1)
