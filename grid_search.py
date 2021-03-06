#!coding=utf-8
"""
Search best parameters for the custom score functions.
"""
from __future__ import print_function, absolute_import

import numpy as np
from os.path import isfile
from itertools import product
from joblib import Parallel, delayed
from tournament import Agent, play_round
from sample_players import RandomPlayer
from sample_players import open_move_score, center_score, improved_score
from game_agent import MinimaxPlayer, AlphaBetaPlayer
from game_agent import custom_score, custom_score_2, custom_score_3
from argparse import ArgumentParser
from functools import partial

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


def get_cpu_agents():
    """
    Return a list of cpu agents.

    Returns
    -------
    agents : List[Agent]
        A list of opponent agents to test.

    """
    return [
        Agent(RandomPlayer(), "Random"),
        Agent(MinimaxPlayer(score_fn=open_move_score), "MM_Open"),
        Agent(MinimaxPlayer(score_fn=center_score), "MM_Center"),
        Agent(MinimaxPlayer(score_fn=improved_score), "MM_Improved"),
        Agent(AlphaBetaPlayer(score_fn=open_move_score), "AB_Open"),
        Agent(AlphaBetaPlayer(score_fn=center_score), "AB_Center"),
        Agent(AlphaBetaPlayer(score_fn=improved_score), "AB_Improved")
    ]


def load_grids(filename):
    """
    Load grid points from a npz file.

    Parameters
    ----------
    filename : str
        The file to load.

    Returns
    -------
    grids : array_like
        An array of grids to search.

    """
    ar = np.load(filename)
    return ar["points"]


def _eval_with_params(cpu_agents, num_matches, score_fn, **kwargs):
    """
    Evaluation the given parameter set for the score function.

    Parameters
    ----------
    cpu_agents : List[Agent]
        A list of opponent agents to play with.
    num_matches : int
        The number of matches against each opponent.
    fn : Callable
        A score function.
    kwargs : dict
        The parameters to evaluate

    Returns
    -------
    total_wins : int
        The total number of wins for this parameter set.

    """
    fn = partial(score_fn, **kwargs)
    test_agent = Agent(AlphaBetaPlayer(score_fn=fn), "Eval")
    total_wins = 0
    for cpu_agent in cpu_agents:
        wins = {test_agent.player: 0,
                cpu_agent.player: 0}
        play_round(cpu_agent, [test_agent], wins, num_matches=num_matches)
        total_wins += wins[test_agent.player]
    return total_wins


def grid_search_custom_fn3_ab(n_jobs=-1, num_matches=20, grid_file=None):
    """
    Grid search of the best a,b for `custom_score_3`.

    Parameters
    ----------
    n_jobs : int
        The maximum number of concurrently running jobs. If -1 all CPUs are
        used. If 1 is given, no parallel computing code is used at all, which is
        useful for debugging. For n_jobs below -1, (n_cpus + 1 + n_jobs) are
        used. Thus for n_jobs = -2, all CPUs but one are used.

    num_matches : int
        The number of matches against each opponent.

    grid_file : str
        The npz file which contains the grids to search. If given, only grid
        points in this file will be tested.

    """
    cpu_agents = get_cpu_agents()
    if grid_file and isfile(grid_file):
        ab = load_grids(grid_file)
    else:
        ab = list(product(range(1, 10), range(1, 10)))
    total_matches = num_matches * len(cpu_agents) * 2
    total_wins = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(_eval_with_params)(
            cpu_agents, num_matches, custom_score_3, a=a, b=b
        )
        for a, b in ab
    )
    for i, (a, b) in enumerate(ab):
        print("a = {}, b = {}, win: {} / {}".format(
            a, b, total_wins[i], total_matches))
    best = np.argmax(total_wins)
    print("-------------------------------")
    print("The best result: a = {}, b = {}".format(*ab[best]))
    print("-------------------------------")


def grid_search_custom_fn2_abc(n_jobs=-1, num_matches=20, grid_file=None):
    """
    Grid search of the best a,b,c for `custom_score_2`.

    Parameters
    ----------
    n_jobs : int
        The maximum number of concurrently running jobs. If -1 all CPUs are
        used. If 1 is given, no parallel computing code is used at all, which is
        useful for debugging. For n_jobs below -1, (n_cpus + 1 + n_jobs) are
        used. Thus for n_jobs = -2, all CPUs but one are used.

    num_matches : int
        The number of matches against each opponent.

    grid_file : str
        The npz file which contains the grids to search. If given, only grid
        points in this file will be tested.

    """
    cpu_agents = get_cpu_agents()
    if grid_file and isfile(grid_file):
        abc = load_grids(grid_file)
    else:
        abc = list(product(range(1, 10), range(1, 10), range(1, 5)))
    total_matches = num_matches * len(cpu_agents) * 2
    total_wins = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(_eval_with_params)(
            cpu_agents, num_matches, custom_score_2, a=a, b=b, c=c
        )
        for a, b, c in abc
    )
    for i, (a, b, c) in enumerate(abc):
        print("a = {}, b = {}, c = {}, win: {} / {}".format(
            a, b, c, total_wins[i], total_matches))
    best = np.argmax(total_wins)
    print("---------------------------------------")
    print("The best result: a = {}, b = {}, c = {}".format(*abc[best]))
    print("---------------------------------------")


def grid_search_custom_fn1_abcd(n_jobs=-1, num_matches=20, grid_file=None):
    """
    Grid search of the best a,b,c,d for `custom_score`.

    Parameters
    ----------
    n_jobs : int
        The maximum number of concurrently running jobs. If -1 all CPUs are
        used. If 1 is given, no parallel computing code is used at all, which is
        useful for debugging. For n_jobs below -1, (n_cpus + 1 + n_jobs) are
        used. Thus for n_jobs = -2, all CPUs but one are used.

    num_matches : int
        The number of matches against each opponent.

    grid_file : str
        The npz file which contains the grids to search. If given, only grid
        points in this file will be tested.

    """
    cpu_agents = get_cpu_agents()
    if grid_file and isfile(grid_file):
        abcd = load_grids(grid_file)
    else:
        abcd = list(product(range(1, 10), range(1, 10),
                            range(1, 5), range(1, 5)))
    total_matches = num_matches * len(cpu_agents) * 2
    total_wins = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(_eval_with_params)(
            cpu_agents, num_matches, custom_score, a=a, b=b, c=c, d=d
        )
        for a, b, c, d in abcd
    )
    for i, (a, b, c, d) in enumerate(abcd):
        print("a = {}, b = {}, c = {}, d = {}, win: {} / {}".format(
            a, b, c, d, total_wins[i], total_matches))
    best = np.argmax(total_wins)
    print("-----------------------------------------------")
    print("The best result: a = {}, b = {}, c = {}, d = {}".format(*abcd[best]))
    print("-----------------------------------------------")


def custom_match(num_matchs=20):
    """
    Matches between the custom functions.

    Parameters
    ----------
    num_matchs : int
        The number of matches against each opponent.

    """
    agents = [
        Agent(AlphaBetaPlayer(score_fn=custom_score), "AB_Custom_1"),
        Agent(AlphaBetaPlayer(score_fn=custom_score_2), "AB_Custom_2"),
        Agent(AlphaBetaPlayer(score_fn=custom_score_3), "AB_Custom_3"),
    ]
    for i in range(1, 3):
        for j in range(i + 1, len(agents)):
            agent_i = agents[i]
            agent_j = agents[j]
            wins = {agent_i.player: 0, agent_j.player: 0}
            play_round(agent_i, [agent_j], wins, num_matches=num_matchs)
            print('{} vs {} = {} : {}'.format(
                agent_i.name,
                agent_j.name,
                wins[agent_i.player],
                wins[agent_j.player])
            )


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument(
        "score_fn",
        choices=["fn1", "fn2", "fn3", "versus"],
        type=str,
        help="Choose the score function to tune."
    )
    parser.add_argument(
        "--num_matches",
        default=10,
        type=int,
        help="The number of matches against each opponent."
    )
    parser.add_argument(
        "--num_jobs",
        default=-1,
        type=int,
        help="The maximum number of concurrently running jobs, such as the "
             "number of Python worker processes when backend=”multiprocessing” "
             "or the size of the thread-pool when backend=”threading”. If -1 "
             "all CPUs are used. If 1 is given, no parallel computing code is "
             "used at all, which is useful for debugging. For n_jobs below -1, "
             "(n_cpus + 1 + n_jobs) are used. Thus for n_jobs = -2, all CPUs "
             "but one are used."
    )
    parser.add_argument(
        "--grid_file",
        default=None,
        type=str,
        help="The npz file which contains the grids to search. If given, only "
             "grid points in this file will be tested."
    )

    args = parser.parse_args()
    params = {"num_matches": args.num_matches,
              "n_jobs": args.num_jobs,
              "grid_file": args.grid_file}
    if args.score_fn == "versus":
        custom_match(num_matchs=args.num_matches)
    elif args.score_fn == "fn1":
        grid_search_custom_fn1_abcd(**params)
    elif args.score_fn == "fn2":
        grid_search_custom_fn2_abc(**params)
    else:
        grid_search_custom_fn3_ab(**params)
