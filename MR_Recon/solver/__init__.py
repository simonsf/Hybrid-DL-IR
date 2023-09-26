# create by wangerxiao on 20221013
from .acs_solver import ACSSolver


def create_solver(opt):
    if opt['mode'] == 'sr':
        solver = ACSSolver(opt)
    else:
        raise NotImplementedError

    return solver