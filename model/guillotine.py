import itertools
from typing import Tuple, List
from dataclasses import dataclass
from queue import SimpleQueue

import pulp
from pulp import PULP_CBC_CMD, GUROBI_CMD

from model.abstract_model import AbstractModel


@dataclass(eq=True, frozen=True, unsafe_hash=True)
class Rect:
    width: int
    height: int


class Guillotine(AbstractModel):
    """
    Please refer to Furini et al. (2016) for the detailed model and the plate and variable enumeration methods.

    References:
        Fabio Furini, Enrico Malaguti, Dimitri Thomopulos (2016) Modeling Two-Dimensional Guillotine Cutting Problems via Integer
        Programming. INFORMS Journal on Computing 28(4):736-751. https://doi.org/10.1287/ijoc.2016.0710
    """
    def __init__(self, grid: Tuple[int, int], shapes: List[Tuple[int, int]],
                 upper_bound):
        self.grid = grid
        self.shapes = shapes
        self.upper_bound = upper_bound
        self.m = pulp.LpProblem('guillotine', pulp.LpMaximize)
        self.orients = {'h', 'v'}  # 这里指的是切割的方向，而不是item旋转方向
        return

    def _set_iterables(self):
        # indices and shapes of all items and plates
        # Key = plate_idx, Value = Rect
        self.cap_j = dict()  # 所有可能的plates。item是plate的特殊情况
        # indices of all plates
        self.cap_j_plates = set()  # 所有plate的不同indices，不包括items
        # indices of all items
        self.cap_j_items = set()  # 所有items的不同indices
        # Given a plate and an orientation, store all possible cut positions
        # Key = [plate, orientation], Value = List[positions]
        self.cut_pos_dict = dict()
        # Given a plate, a cut position and an orientation, store the two items/plates generated
        # Key = [plate, cut_pos, orientation], Value = List[plate or item in cap_j]
        self.plate_cut_dict = dict()
        # Given an item/plate j and an orientation, store all plates
        # that can generate j with the corresponding cut position
        # Key = [item, orientation], Value = List[Tuple[plate, cut_pos]]
        self.inverse_plate_cut_dict = dict()
        # Store all the valid cuts, each is stored as Tuple[position, plate_idx, orientation]
        # Use this to define x variables
        self.cut_set = set()  # plate_idx指的是一种形状，而不是id

        (self.cap_j, self.cap_j_items, self.cut_set, self.plate_cut_dict,
         self.inverse_plate_cut_dict,
         self.cut_pos_dict) = self._plate_variable_enumeration()
        self.cap_j_plates = set(self.cap_j.keys()).difference(self.cap_j_items)
        return

    def _set_variables(self):
        # 每一个item的数量
        self.y = pulp.LpVariable.dicts('y',
                                       self.cap_j_items,  # 对应每一个item
                                       cat=pulp.LpInteger,
                                       lowBound=0,
                                       upBound=self.upper_bound)
        # 每一条切线被切的次数
        self.x = pulp.LpVariable.dicts('x',
                                       self.cut_set,
                                       cat=pulp.LpInteger,
                                       lowBound=0)
        return

    def _set_objective(self):
        self.m += pulp.lpSum(self.y[j] for j in self.cap_j_items)  # item数量最多
        return

    def _set_constraints(self):
        for j in self.cap_j_items:  # j是shape index，而不是具体的id
            # 能切出j的切线实际被切的总次数 >= 被切的形状j的数量 + 形状j被留作item的数量
            # 不是等于号是因为切完之后不一定得到形状j，也可能是其他比j大的形状
            self.m += (
                pulp.lpSum(self.x[q, k, o] for o in self.orients
                           for (q, k) in self.inverse_plate_cut_dict.get(
                               (j, o), list())) -
                pulp.lpSum(self.x[q, j, o] for o in self.orients
                           for q in self.cut_pos_dict.get((j, o), list())) -
                self.y[j] >= 0, f'shape_{j}')
        for j in self.cap_j_plates:
            # 同上，只是不属于item，减少了y变量的数量
            if j == 0:
                continue
            self.m += (
                pulp.lpSum(self.x[q, k, o] for o in self.orients
                           for (q, k) in self.inverse_plate_cut_dict.get(
                               (j, o), list())) -
                pulp.lpSum(self.x[q, j, o] for o in self.orients
                           for q in self.cut_pos_dict.get((j, o), list())) >=
                0, f'plate_{j}')
        # 原来的板只能被切一次
        self.m += (pulp.lpSum(self.x[q, 0, o] for o in self.orients
                              for q in self.cut_pos_dict.get((0, o), list()))
                   <= 1, 'panel')
        return

    def _optimize(self):
        time_limit_in_seconds = 1 * 60 * 60
        self.m.solve(PULP_CBC_CMD(timeLimit=time_limit_in_seconds,
                                  gapRel=0.01))
        # self.m.solve(GUROBI_CMD(timeLimit=time_limit_in_seconds, gapRel=0.01))
        return

    def _is_feasible(self):
        return True

    def _process_infeasible_case(self):
        return list(), list(), list(), list()

    def _post_process(self):
        result_y = dict()
        result_x = dict()
        for j in self.cap_j_items:
            if self.y[j].value() > 0.9:
                result_y[j] = self.y[j].value()
        for (q, k, o) in self.cut_set:
            if self.x[q, k, o].value() > 0.9:
                result_x[k] = ((q, o), self.x[q, k, o].value())
        result_blocks, result_cuts = self._get_final_results(
            result_y, result_x)
        return result_blocks, result_cuts

    def _plate_variable_enumeration(self):
        """
        遍历整个切割的过程，获得所有可能的切线和切割得到的plates。这些plates的其中一些会被保留为items。
        """
        cut_pos_dict = dict()   # return. cut_pos_dict[plate_idx, orientation] = positions  一个plate在一个方向上所有可能的切割位置
        shape_j = set()         # return, shape index set, len = 2
        cap_j = dict()          # return, {index: Rect}
        inverse_cap_j = dict()  # {Rect: index}
        cap_j_set = set()       # Rect set
        un_processed = set()    # plate_index set

        # add gird
        grid = Rect(self.grid[0], self.grid[1])
        cap_j[0] = grid
        inverse_cap_j[grid] = 0
        cap_j_set.add(grid)
        un_processed.add(0)

        # add shapes
        for shape in self.shapes:
            rect_shape = Rect(shape[0], shape[1])
            count = len(cap_j)
            cap_j[count] = rect_shape
            inverse_cap_j[rect_shape] = count
            cap_j_set.add(rect_shape)
            shape_j.add(count)

        x_set = list()  # return. 对应变量x. 每一个元素是(position, plate_idx, otientation), 对应一条切线。
        # Key = [plate, cut_pos, orientation], Value = List[plate or item in cap_j]
        cut_dict = dict()  # return. 切之前的一块板和切之后的两块板之间基于index的对应关系。cut_dict[(plate_idx, position, orientation)] = [plate1_idx, plate2_idx]
        # Key = [item, orientation], Value = List[Tuple[cut_pos, plate]]
        inverse_cut_dict = dict()  # return. item和可以切出item的板之间基于index的对应关系. inverse_cut_dict[(plate1_idx)] = (position, plate_idx)

        while un_processed:  # 一次处理一个plate
            plate_idx = un_processed.pop()
            plate = cap_j[plate_idx]  # Rect
            for o in self.orients:
                positions = self._compute_possible_cut_positions(plate, o)  # 根据shapes得到所有可能切割的位置
                cut_pos_dict[plate_idx, o] = positions
                for position in positions:  # 遍历每一个切割位置得到切割结果
                    plate1, plate2 = _cut(plate, o, position)  # plate1和plate2都只是形状，不包含index
                    if plate1 not in cap_j_set:
                        count = len(cap_j)
                        cap_j[count] = plate1
                        inverse_cap_j[plate1] = count
                        cap_j_set.add(plate1)
                        un_processed.add(count)
                    if plate2 not in cap_j_set:
                        count = len(cap_j)
                        cap_j[count] = plate2
                        inverse_cap_j[plate2] = count
                        cap_j_set.add(plate2)
                        un_processed.add(count)
                    x_set.append((position, plate_idx, o))
                    plate1_idx = inverse_cap_j[plate1]
                    plate2_idx = inverse_cap_j[plate2]
                    cut_dict[(plate_idx, position,
                              o)] = [plate1_idx, plate2_idx]
                    inverse_cut_dict.setdefault(
                        (plate1_idx, o), list()).append((position, plate_idx))
                    inverse_cut_dict.setdefault(
                        (plate2_idx, o), list()).append((position, plate_idx))

        return cap_j, shape_j, x_set, cut_dict, inverse_cut_dict, cut_pos_dict

    def _compute_possible_cut_positions(self, plate, o):
        positions = list()
        if o == 'h':
            available_shapes = [
                s for s in self.shapes
                if s[0] <= plate.width and s[1] < plate.height
            ]
            if not available_shapes:
                return positions
            max_pieces = [plate.height // s[1] for s in available_shapes]
            max_pieces_iter = [range(max_piece + 1) for max_piece in max_pieces]
            for pieces in itertools.product(*max_pieces_iter):
                cut_height = sum(piece * s[1]
                                 for piece, s in zip(pieces, available_shapes))
                if 0 < cut_height < plate.height:
                    positions.append(cut_height)
        elif o == 'v':
            available_shapes = [
                s for s in self.shapes
                if s[0] < plate.width and s[1] <= plate.height
            ]
            if not available_shapes:
                return positions
            max_pieces = [plate.width // s[0] for s in available_shapes]
            max_pieces_iter = [range(max_piece + 1) for max_piece in max_pieces]
            for pieces in itertools.product(*max_pieces_iter):
                cut_width = sum(piece * s[0]
                                for piece, s in zip(pieces, available_shapes))
                if 0 < cut_width < plate.width:
                    positions.append(cut_width)
        else:
            raise ValueError(f'Unknown orientation {o}')
        return positions

    def _get_final_results(self, result_y, result_x):
        # Only leaf nodes are recorded (and should be a shape)
        result_blocks = list()
        result_cuts = list()
        unprocessed = SimpleQueue()
        plate = 0
        unprocessed.put((plate, 0, 0))
        while not unprocessed.empty():
            plate, start_x, start_y = unprocessed.get()
            if plate in result_x:
                pos, orient = result_x[plate][0]
            else:
                min_x = start_x
                min_y = start_y
                max_x = start_x + self.cap_j[plate].width
                max_y = start_y + self.cap_j[plate].height
                # Only leaf nodes which are items are added to result_blocks
                if plate in self.cap_j_items:
                    result_blocks.append([(min_x, min_y), (max_x, min_y),
                                          (max_x, max_y), (min_x, max_y),
                                          (min_x, min_y)])
                continue
            plate1, plate2 = self.plate_cut_dict[plate, pos, orient]
            unprocessed.put((plate1, start_x, start_y))
            if orient == 'h':
                min_x = start_x
                min_y = start_y + pos
                max_x = start_x + self.cap_j[plate2].width
                max_y = start_y + pos + self.cap_j[plate2].height
                unprocessed.put((plate2, min_x, min_y))
                result_cuts.append([(min_x, min_y), (max_x, min_y)])
            else:
                min_x = start_x + pos
                min_y = start_y
                max_x = start_x + pos + self.cap_j[plate2].width
                max_y = start_y + self.cap_j[plate2].height
                unprocessed.put((plate2, min_x, min_y))
                result_cuts.append([(min_x, min_y), (min_x, max_y)])

        return result_blocks, result_cuts


def _cut(plate, o, pos):
    if o == 'v':
        plate1 = Rect(pos, plate.height)
        plate2 = Rect(plate.width - pos, plate.height)
        return plate1, plate2
    elif o == 'h':
        plate1 = Rect(plate.width, pos)
        plate2 = Rect(plate.width, plate.height - pos)
        return plate1, plate2
    else:
        raise ValueError(f'Unknown orientation {o}')
