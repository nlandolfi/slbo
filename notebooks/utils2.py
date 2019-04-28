import json
import os
import typing    

import numpy as np
import matplotlib.pyplot as plt


class SLBOIter(object):
    n_iters: int
    n_policy_iters: int
    n_evaluate_iters: int

    trpo_means: np.array
    trpo_stds: np.array
    trpo_n_episodes: np.array
    trpo_times: np.array

    real_mean_evals: np.array
    virt_mean_evals: np.array
    real_stds_evals: np.array
    virt_stds_evals: np.array
    real_eval_times: np.array
    virt_eval_times: np.array

    def __init__(self, n_iters: int, n_policy_iters: int, n_evaluate_iters):
        self.n_iters = n_iters
        self.n_policy_iters = n_policy_iters
        self.n_evaluate_iters = n_evaluate_iters

        self.trpo_means = np.array([])
        self.trpo_stds = np.array([])
        self.trpo_n_episodes = np.array([])
        self.trpo_times = np.array([])

        self.real_mean_evals: np.array([])
        self.virt_mean_evals: np.array([])
        self.real_stds_evals: np.array([])
        self.real_virt_evals: np.array([])

    def append_trpo(self, eps: int, mean: float, std: float, time: int):
        np.append(self.trpo_n_episodes, eps)
        np.append(self.trpo_means, mean)
        np.append(self.trpo_stds, std)
        np.append(self.trpo_times, time)

    def append_real(self, mean: float, std: float):
        np.append(self.real_mean_evals, mean)
        np.append(self.real_stds_evals, std)

    def append_virt(self, mean: float, std: float):
        np.append(self.virt_mean_evals, mean)
        np.append(self.virt_stds_evals, std)

    def verify(self):
        assert len(self.trpo_means) == len(self.trpo_stds)
        assert len(self.trpo_stds) == len(self.trpo_n_episodes)
        assert len(self.trpo_means) == self.n_iters * self.n_policy_iters

        assert len(self.real_mean_evals) == len(self.real_stds_evals)
        assert len(self.real_stds_evals) == len(self.virt_mean_evals)
        assert len(self.virt_mean_evals) == len(self.virt_stds_evals)
        assert len(self.real_mean_evals) == self.n_evaluate_iters


class SLBO(object):
    stages: typing.List[SLBOIter]

    def __init__(self):
        self.stages = []

    def append_stage(self, s: SLBOIter):
        self.stages.append(s)

class Run(object):
    tasks: typing.List[float]

    # task number, TRPO iteration.
    warms: typing.List[SLBOIter]

    # task number, SLBO iteration, TRPO iteration.
    slbos: typing.List[SLBO]

    n_slbo_iters: int

    n_expected_tasks: int

    def __init__(self, n_expected_tasks=None):
        self.n_expected_tasks = n_expected_tasks
        self.tasks = []
        self.warms = []
        self.slbos = []

    def append_task(self, t: float): 
        self.tasks.append(t)

    def append_warmup(self, w: SLBOIter):
        self.warms.append(w)

    def append_slbo(self, s: SLBO):
        self.slbos.append(s)

    def plot_tasks(self):
        plt.stem(self.tasks)
        plt.title(f"Tasks ({len(self.tasks)} of them)")
        plt.xlabel("task #")
        plt.ylabel("goal velocity (speed, really)")

    def plot_warmup(self, i: int, min=None, max=None, avg=None):
        if i == 0:
            print("No warmup for first task")

        fig, ax = plt.subplots(figsize=(15, 5))

        warm = self.warms[9]

        ax.plot(warm.trpo_means)

        eval_spots = [i * warm.n_policy_iters*warm.n_evaluate_iters
                        for i in range(int(warm.n_iters/warm.n_evaluate_iters))]
        ax.errorbar(
            eval_spots,
            warm.real_mean_evals,
            warm.real_stds_evals,
            color='r',
            fmt='o'
        )

        ax.errorbar(
            eval_spots,
            warm.virt_mean_evals,
            warm.virt_stds_evals,
            color='g',
            fmt='o'
        )

        if min is not None:
            ax.axhline(y=min, color="yellow", label="min")
        if max is not None:
            ax.axhline(y=max, color="orange", label="max")
        if avg is not None:
            ax.axhline(y=avg, color="green", label="avg")

        ax.set_ylim(-300, 0)
        ax.set_title(f"Warmup #{i}, Task is {self.tasks[i]}")
        ax.legend()
        ax.y





    if n == 0:
        return
    
    fig, ax = plt.subplots(figsize=(15, 5))

    plot_stage(ax, warms[jobidx][n], 80, 40)
    
    length = min(len(real_warms[jobidx][n]), len(virt_warms[jobidx][n]))

    ax.scatter([(i+1)*40 for i in range(0, length)], real_warms[jobidx][n], color="r")
    ax.scatter([(i+1)*40 for i in range(0, length)], virt_warms[jobidx][n], color="y")
    
    taskvel = tasks[jobidx][n]
    ax.axhline(y=est(mins, taskvel), color="yellow", label="min")
    ax.axhline(y=est(maxs, taskvel), color="orange", label="max")
    ax.axhline(y=est(avgs, taskvel), color="green", label="avg")
    ax.set_ylim((-300, 0))
    ax.set_title(f"WARMUP: Task {n}, Velocity: {tasks[jobidx][n]} WARMUPS")
    ax.legend()


# read a structured log file with lines of JSON to a list of dictionaries.
def read_log(path):
    lines = []
    with open(path) as f:
        for line in f:
            lines.append(json.loads(line))
    return lines

def from_lines(lines) -> Run:
    task_num = 0

    r = Run()
    warm = None
    slbo = None
    stage = None

    for l in lines:
        fmt = l['fmt']
        args = l['args']
        time = l['time']
    
        if "STARTING TASK" in fmt:
            task_num += 1
            warm = SLBOIter()
            slbo = SLBO()
            r.append_warmup(warm)
            r.append_slbo(slbo)

        if "Task Sampled" in fmt or "Task Fixed" in fmt:
            r.append_task(args[0]["goal_velocity"])

        if "Starting Stage" in fmt:
            stage = SLBOIter()
            slbo.append_stage(stage)

        if args[0] == "pre-warm-up":
            in_warmup = True
        if args[0] == "post-warmup":
            in_warmup = False

        if "[TRPO]" in fmt:
            n_episodes = args[1]
            r_mean = args[2]
            r_std = args[3]

            if in_warmup:
                warm.append_trpo(n_episodes, r_mean, r_std)
            else:
                warm.append_trpo(n_episodes, r_mean, r_std)

        if "iteration" == args[0]:
            if in_warmup:
                if args[1] == "Real Env":
                    warm.append_real(args[3], args[4])
                if args[1] == "Virt Env":
                    warm.append_virt(args[3], args[4])

    return r

def copy(remotepath: str, localpath: str):
    command = f"scp lando@sc.stanford.edu:{remotepath} {localpath}"
    rcode = os.system(command)
    if rcode != 0:
        print(command)
        raise Exception("Bad Status code")
    return True

def remote(job: int):
    return f"/tiger/u/lando/cmeta/{job}/log.json" 

def local(job) -> str:
    return  f"/tmp/log{job}.json" 

def load(job: int, reload=True, verbose=True) -> Run:
    if reload:
        if verbose:
            print("Loading log file")
        copy(remote(job), local(job))
        if verbose:
            print("Loaded log file")
    else:
        if verbose:
            print("Skipping Load")

    if verbose:
        print("Reading lines")
    lines = read_log(local(job))
    if verbose:
        print("Read lines")
    return from_lines(lines)