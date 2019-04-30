import datetime
import math
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

        self.real_mean_evals = np.array([])
        self.virt_mean_evals = np.array([])
        self.real_stds_evals = np.array([])
        self.virt_stds_evals = np.array([])
        self.real_eval_times = np.array([])
        self.virt_eval_times = np.array([])

    def append_trpo(self, eps: int, mean: float, std: float, time: int):
        self.trpo_n_episodes = np.append(self.trpo_n_episodes, eps)
        self.trpo_means = np.append(self.trpo_means, mean)
        self.trpo_stds = np.append(self.trpo_stds, std)
        self.trpo_times = np.append(self.trpo_times, time)

    def append_real(self, mean: float, std: float, time: int):
        self.real_mean_evals = np.append(self.real_mean_evals, mean)
        self.real_stds_evals = np.append(self.real_stds_evals, std)
        self.real_eval_times = np.append(self.real_eval_times, time)

    def append_virt(self, mean: float, std: float, time: int):
        self.virt_mean_evals = np.append(self.virt_mean_evals, mean)
        self.virt_stds_evals = np.append(self.virt_stds_evals, std)
        self.virt_eval_times = np.append(self.virt_eval_times, time)

    def is_complete(self) -> bool:
        return len(self.trpo_means) == self.n_iters * self.n_policy_iters and len(self.real_mean_evals) == self.n_evaluate_iters
         
    def verify(self):
        assert len(self.trpo_means) == len(self.trpo_stds)
        assert len(self.trpo_stds) == len(self.trpo_n_episodes)

        assert len(self.real_mean_evals) == len(self.real_stds_evals)
        assert len(self.real_stds_evals) == len(self.virt_mean_evals)
        assert len(self.virt_mean_evals) == len(self.virt_stds_evals)

    def duration_in_seconds(self):
        if len(self.trpo_times) == 0:
            return 0
        return self.trpo_times[-1] - self.trpo_times[0]

    def plot(self, minrew=None, maxrew=None, avgrew=None, title=None, ylim=None):
        try:
            self.verify()
        except AssertionError as e:
            print(f"Verify Failed {e}")
            return

        fig, ax = plt.subplots(figsize=(15, 5))

        mt = min(np.min(self.trpo_times), np.min(self.real_eval_times), np.min(self.virt_eval_times))

        ax.plot(self.trpo_times - mt, self.trpo_means, label="trpo means")


        ax.errorbar(
            self.real_eval_times - mt,
            self.real_mean_evals,
            self.real_stds_evals,
            color='red',
            fmt='o',
            label="real evals"
        )

        ax.errorbar(
            self.virt_eval_times - mt,
            self.virt_mean_evals,
            self.virt_stds_evals,
            color='blue',
            fmt='o',
            label='virtual evals'
        )

        # if minrew is not None:
        #     ax.axhline(y=min, color="yellow", label="min")
        # if maxrew is not None:
        #     ax.axhline(y=max, color="orange", label="max")
        # if avgrew is not None:
        #     ax.axhline(y=avg, color="green", label="avg")

        # ax.set_ylim(-300, 0)
        if title is not None:
            ax.set_title(title)
        ax.legend()
        ax.set_ylabel("reward")

        if ylim is not None:
            ax.set_ylim(ylim)



        #return fig


class SLBO(object):
    stages: typing.List[SLBOIter]

    def __init__(self):
        self.stages = []

    def append_stage(self, s: SLBOIter):
        self.stages.append(s)

class Run(object):
    start_time: int
    
    tasks: typing.List[float]

    post_slbo_real_eval_means: np.array
    post_slbo_real_eval_stds: np.array
    post_slbo_real_eval_times: np.array

    # task number, TRPO iteration.
    warms: typing.List[SLBOIter]

    # task number, SLBO iteration, TRPO iteration.
    slbos: typing.List[SLBO]

    n_slbo_iters: int

    n_expected_tasks: int

    def __init__(self, start_time: int, n_expected_tasks=None):
        self.start_time = start_time
        self.n_expected_tasks = n_expected_tasks
        self.tasks = []
        self.warms = []
        self.slbos = []
        self.post_slbo_real_eval_means = np.array([])
        self.post_slbo_real_eval_stds = np.array([])
        self.post_slbo_real_eval_times = np.array([])

    def append_task(self, t: float): 
        self.tasks.append(t)

    def append_warmup(self, w: SLBOIter):
        self.warms.append(w)

    def append_slbo(self, s: SLBO):
        self.slbos.append(s)

    def append_post_slbo_real_eval(self, mean: float, std: float, time: int):
        self.post_slbo_real_eval_means = np.append(self.post_slbo_real_eval_means, mean)
        self.post_slbo_real_eval_stds = np.append(self.post_slbo_real_eval_stds, std)
        self.post_slbo_real_eval_times = np.append(self.post_slbo_real_eval_times, time)

    def plot_tasks(self):
        plt.stem(self.tasks)
        plt.title(f"Tasks ({len(self.tasks)} of them)")
        plt.xlabel("task #")
        plt.ylabel("goal velocity (speed, really)")

    def details(self):
        print(f"Run:\n\t{len(self.tasks)} Tasks so far")

    def plot_post_slbo_evals(self, ylim=None):
        fig, ax = plt.subplots(figsize=(15, 5))
        ax.errorbar(
            np.arange(len(self.post_slbo_real_eval_means)),
            self.post_slbo_real_eval_means,
            self.post_slbo_real_eval_stds,
            color='red',
            fmt='o',
            label="real evals"
        )
        if ylim is not None:
            ax.set_ylim(ylim)

    def plot_timing(self):
        offset = 0
        fig, ax = plt.subplots(figsize=(15, 5))
    
        for i in range(len(self.tasks)):
            inoff = 0
            # plot warmup

            ntrpos = len(self.warms[i].trpo_times)

            ax.plot(offset + inoff + np.arange(ntrpos), self.warms[i].trpo_times - self.start_time, color="blue", label="warmup")

            inoff += ntrpos

            for j in range(len(self.slbos[i].stages)):
                ininoff = 0
                stage: SLBOIter= self.slbos[i].stages[j]
                ntrpos = len(stage.trpo_times)

                ax.plot(offset + inoff + ininoff + np.arange(ntrpos), stage.trpo_times - self.start_time, color="green", label="stage")
                ininoff += ntrpos

                inoff += ininoff

            offset += inoff       




# read a structured log file with lines of JSON to a list of dictionaries.
def read_log(path):
    lines = []
    with open(path) as f:
        for line in f:
            lines.append(json.loads(line))
    return lines

def timefrom(s):
    return datetime.datetime.strptime(s, '%Y-%m-%dT%H:%M:%S.%f')

def from_lines(lines) -> Run:
    task_num = 0

    r = Run(timefrom(lines[0]['time']).timestamp())
    warm = None
    slbo = None
    stage = None

    for l in lines:
        fmt = l['fmt']
        args = l['args']
        time = timefrom(l["time"])
    
        if "STARTING TASK" in fmt:
            task_num += 1
            warm = SLBOIter(40, 40, 1)
            slbo = SLBO()
            r.append_warmup(warm)
            r.append_slbo(slbo)

        if "Task Sampled" in fmt or "Task Fixed" in fmt:
            if type(args[0]) is str:
                if "BACKWARD" in args[0]:
                    r.append_task(-math.inf)
                if "FORWARD" in args[0]:
                    r.append_task(math.inf)
            else:
                r.append_task(args[0]["goal_velocity"])

        if "Starting Stage" in fmt:
            stage = SLBOIter(20, 40, 10)
            slbo.append_stage(stage)

        if args[0] == "pre-warm-up":
            in_warmup = True
        if args[0] == "post-warm-up":
            in_warmup = False

        if "[TRPO]" in fmt:
            n_episodes = args[1]
            r_mean = args[2]
            r_std = args[3]
            if in_warmup:
                warm.append_trpo(n_episodes, r_mean, r_std, time.timestamp())
            else:
                stage.append_trpo(n_episodes, r_mean, r_std, time.timestamp())

        if "# Iter" in fmt:
            # pull out the model information
            pass

        if "iteration" == args[0]:
            if in_warmup:
                if args[1] == "Real Env":
                    warm.append_real(args[3], args[4], time.timestamp())
                if args[1] == "Virt Env":
                    warm.append_virt(args[3], args[4], time.timestamp())
            else:
                if args[1] == "Real Env":
                    stage.append_real(args[3], args[4], time.timestamp())
                if args[1] == "Virt Env":
                    stage.append_virt(args[3], args[4], time.timestamp())

        if "episode" == args[0]:
            if in_warmup:
                raise Exception("shouldn't have episode eval in warmup!")

        if "post-slbo" == args[0]:
            if args[1] == "Real Env":
                r.append_post_slbo_real_eval(args[3], args[4], time.timestamp())

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

def load(job: int, reload=True, verbose=True):
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
    return (from_lines(lines), lines)