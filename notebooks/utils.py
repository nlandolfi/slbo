import json
import numpy as np
import matplotlib.pyplot as plt
import datetime

def read_log(path):
    lines = []
    with open(path) as f:
        for line in f:
            lines.append(json.loads(line))
    return lines

def extract_rewards(lines, tasks=True):
    in_warmup = False
    warmup = []
    warmup_task = None
    slbo = []
    slbo_task = None
    slbo_stage = None
    if not tasks:
        slbo_task = []
        slbo.append(slbo_task)

    for line in lines:
        if "STARTING TASK" in line["fmt"]:
            slbo_task = []
            slbo.append(slbo_task)
            warmup_task = []
            warmup.append(warmup_task)
        if "Starting Stage" in line["fmt"]:
            slbo_stage = []
            slbo_task.append(slbo_stage)

        if len(line["args"]) == 0:
            continue

        if line["args"][0] == "pre-warm-up":
            in_warmup = True
            if line["args"][1] == "Real Env":
                pre_warmup_real_reward = line["args"][3]
        if line["args"][0] == "post-warm-up":
            in_warmup = False
            if line["args"][1] == "Real Env":
                post_warmup_real_reward = line["args"][3]

        if "[TRPO]" in line["fmt"]:
            if in_warmup:
                warmup_task.append(line["args"][2])
            else:
                slbo_stage.append(line["args"][2])

    return (warmup, slbo)

def plot_stage(ax, rewards, iters, policy_iters, title=None, ylim=None, iterlines=True):
    if iterlines:
        for i in range(iters):
            ax.axvline(x=i*policy_iters, alpha=0.5)
    ax.plot(rewards)
    if title:
        ax.set_title(title)
    if ylim:
        ax.set_ylim(ylim)

def plot_stages(task, iters, p_iters, ylim=None):
    num = len(task)
    fig, ax = plt.subplots(num, 1, figsize=(14, num*5 +1))
    if num == 1:
        plot_stage(ax, task[0], iters, p_iters, title=f"Stage 0", ylim=ylim)
    else:
        for (i, stage) in enumerate(task):
            plot_stage(ax[i], stage, iters, p_iters, title=f"Stage {i}", ylim=ylim)
    return ax

def plot_log_stages(task, iters, p_iters):
    num = len(task)
    fig, ax = plt.subplots(num, 1, figsize=(14, num*5 +1))
    for (i, stage) in enumerate(task):
        plot_stage(ax[i], [np.log(r) for r in stage], iters, p_iters, title=f"Stage {i}")

def get_warmup(lines, shadow=False):
    real_warm = []
    virt_warm = []
    virt_std = []
    shad_warm = []
    warm = False
    already = False
    for line in lines:
        if len(line["args"]) == 0:
            continue

        if "pre-warm-up" == line["args"][0]:
            warm = True
            if already:
                break
        if "post-warm-up" == line["args"][0]:
            warm = False
            already = True
        if "iteration" == line["args"][0]:
            if warm:
                if line["args"][1] == "Real Env":
                    real_warm.append(line["args"][3])
                if line["args"][1] == "Virt Env":
                    virt_warm.append(line["args"][3])
                    virt_std.append(line["args"][4])
                if line["args"][1] == "Shadow Env":
                    shad_warm.append(line["args"][3])
    if shadow: #incredibly gross, but to not break old notebooks
        return real_warm, virt_warm, shad_warm
    else:
        return real_warm, virt_warm

def get_warmups(lines, n_shadow):
    if n_shadow > 10:
        raise Exception("not robust to larger than 10, see line below")
    real_warms = []
    virt_warms = []
    virt_stds = []
    shad_warms = []
    warm = False
    real_warm = None
    virt_warm = None
    virt_std = None
    shad_warm = None
    for line in lines:
        if len(line["args"]) == 0:
            continue

        if "pre-warm-up" == line["args"][0]:
            if not warm:
                warm = True
                real_warm = []
                virt_warm = []
                virt_std = []
                shad_warm = [[] for i in range(n_shadow)]
                real_warms.append(real_warm)
                virt_warms.append(virt_warm)
                virt_stds.append(virt_std)
                shad_warms.append(shad_warm)
        if "post-warm-up" == line["args"][0]:
            warm = False

        if "post-warm-up" == line["args"][0]:
            warm = False
            already = True
        if "iteration" == line["args"][0]:
            if warm:
                if line["args"][1] == "Real Env":
                    real_warm.append(line["args"][3])
                if line["args"][1] == "Virt Env":
                    virt_warm.append(line["args"][3])
                    virt_std.append(line["args"][4])
                if "Shadow Env" in line["args"][1] and n_shadow > 0:
                    # TODO this is not robust
                    n = int(line["args"][1][-1])
                    shad_warm[n].append(line["args"][3])

    return real_warms, virt_warms, shad_warms

def get_warmups2(lines, n_shadow):
    if n_shadow > 10:
        raise Exception("not robust to larger than 10, see line below")
    real_warms = []
    virt_warms = []
    virt_stds = []
    shad_warms = []
    train_losses = []
    dev_losses = []
    grad_norms = []
    timings = []
    warm = False
    real_warm = None
    virt_warm = None
    virt_std = None
    shad_warm = None
    train_loss = None
    dev_loss = None
    grad_norm = None
    timing = None
    for line in lines:
        if len(line["args"]) == 0:
            continue

        if "pre-warm-up" == line["args"][0]:
            if not warm:
                warm = True
                real_warm = []
                virt_warm = []
                virt_std = []
                shad_warm = [[] for i in range(n_shadow)]
                train_loss = []
                dev_loss = []
                grad_norm = []
                timing = []
                real_warms.append(real_warm)
                virt_warms.append(virt_warm)
                virt_stds.append(virt_std)
                shad_warms.append(shad_warm)
                train_losses.append(train_loss)
                dev_losses.append(dev_loss)
                grad_norms.append(grad_norm)
                timings.append(timing)
        if "post-warm-up" == line["args"][0]:
            warm = False

        if "post-warm-up" == line["args"][0]:
            warm = False
            already = True
        if "# Iter" in line["fmt"]:
            if warm:
                train_loss.append(line["args"][1])
                dev_loss.append(line["args"][2])
                grad_norm.append(line["args"][4])
        if "[TRPO]" in line["fmt"]:
            if warm:
                t = datetime.datetime.strptime(line["time"], '%Y-%m-%dT%H:%M:%S.%f')
                timing.append(t)
        if "iteration" == line["args"][0]:
            if warm:
                if line["args"][1] == "Real Env":
                    real_warm.append(line["args"][3])
                if line["args"][1] == "Virt Env":
                    virt_warm.append(line["args"][3])
                    virt_std.append(line["args"][4])
                if "Shadow Env" in line["args"][1] and n_shadow > 0:
                    # TODO this is not robust
                    n = int(line["args"][1][-1])
                    shad_warm[n].append(line["args"][3])

    return real_warms, virt_warms, shad_warms, train_losses, dev_losses, grad_norms, timings

def get_checkpoints(lines):
    real_rewards = []
    virt_rewards = []
    for line in lines:
        if len(line["args"]) == 0:
            continue
        if line["args"][0] == "episode":
            if line["args"][1] == "Real Env":
                real_rewards.append(line["args"][3])
            if line["args"][1] == "Virt Env":
                virt_rewards.append(line["args"][3])

    return (real_rewards, virt_rewards)

def get_multi_checkpoints(lines, includepost=False):
    real_rewards_tasks = []
    virt_rewards_tasks = []
    real_rewards = None
    virt_rewards = None
    warm = False
    for line in lines:
        if len(line["args"]) == 0:
            continue

        if "pre-warm-up" == line["args"][0]:
            if not warm:
                warm = True
                real_rewards = []
                virt_rewards = []
                real_rewards_tasks.append(real_rewards)
                virt_rewards_tasks.append(virt_rewards)
        if "post-warm-up" == line["args"][0]:
            warm = False

        if line["args"][0] == "episode":
            if line["args"][1] == "Real Env":
                real_rewards.append(line["args"][3])
            if line["args"][1] == "Virt Env":
                virt_rewards.append(line["args"][3])

        if line["args"][0] == "post-slbo" and includepost:
            if line["args"][1] == "Real Env":
                real_rewards.append(line["args"][3])
            if line["args"][1] == "Virt Env":
                virt_rewards.append(line["args"][3])

    return (real_rewards_tasks, virt_rewards_tasks)

def split_lines_by_tasks(lines):
    stages = []
    stage = None

    for line in lines:
        if "STARTING TASK" in line["fmt"]:
            stage = []
            stages.append(stage)

        if stage is not None:
            stage.append(line)

    return stages
