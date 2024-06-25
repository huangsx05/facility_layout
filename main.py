from output_handler.drawer import draw_solution
from model.guillotine import Guillotine
from model.set_covering import SetCover


def main():
    grid = (53, 32)
    shapes = [(5, 7), (7, 5)]
    upper_bound = 46
    # mode = 'set_covering'
    mode = 'guillotine'
    if mode == 'set_covering':
        model = SetCover(grid, shapes, upper_bound)  # 像素级别求解，只有binary决策变量
    elif mode == 'guillotine':
        model = Guillotine(grid, shapes, upper_bound)
    else:
        raise ValueError(f'Unknown mode: {mode}')
    blocks, cuts = model.solve()
    draw_solution(grid, blocks, cuts, f'data/output/{mode}')
    return


if __name__ == '__main__':
    main()

# 可都是根据遍历的思想，针对所有的可能性创建变量