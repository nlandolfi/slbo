# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import pickle
from collections import deque
import tensorflow as tf
import numpy as np
import lunzi.nn as nn
from lunzi.Logger import logger
from slbo.utils.average_meter import AverageMeter
from slbo.utils.flags import FLAGS
from slbo.utils.dataset import Dataset, gen_dtype
from slbo.utils.OU_noise import OUNoise
from slbo.utils.normalizer import Normalizers
from slbo.utils.tf_utils import get_tf_config
from slbo.utils.runner import Runner
from slbo.policies.gaussian_mlp_policy import GaussianMLPPolicy
from slbo.envs.virtual_env import VirtualEnv
from slbo.dynamics_model import DynamicsModel
from slbo.v_function.mlp_v_function import MLPVFunction
from slbo.partial_envs import make_env, make_task
from slbo.loss.multi_step_loss import MultiStepLoss
from slbo.algos.TRPO import TRPO


def evaluate(settings, tag):
    for runner, policy, name in settings:
        runner.reset()
        _, ep_infos = runner.run(policy, FLAGS.rollout.n_test_samples)
        returns = np.array([ep_info['return'] for ep_info in ep_infos])
        logger.info('Tag = %s, Reward on %s (%d episodes): mean = %.6f, std = %.6f', tag, name,
                    len(returns), np.mean(returns), np.std(returns))


def add_multi_step(src: Dataset, dst: Dataset):
    n_envs = 1
    dst.extend(src[:-n_envs])

    ending = src[-n_envs:].copy()
    ending.timeout = True
    dst.extend(ending)


def make_real_runner(n_envs, task_config=None):
    from slbo.envs.batched_env import BatchedEnv
    batched_env = BatchedEnv([make_env(FLAGS.env.id, task_config=task_config) for _ in range(n_envs)])
    return Runner(batched_env, rescale_action=True, **FLAGS.runner.as_dict())


def main():
    FLAGS.set_seed()
    FLAGS.freeze()

    task = make_task(FLAGS.env.id)
    env = make_env(FLAGS.env.id, task_config=task)
    dim_state = int(np.prod(env.observation_space.shape))
    dim_action = int(np.prod(env.action_space.shape))

    env.verify()

    normalizers = Normalizers(dim_action=dim_action, dim_state=dim_state)

    dtype = gen_dtype(env, 'state action next_state reward done timeout')
    train_set = Dataset(dtype, FLAGS.rollout.max_buf_size)
    dev_set = Dataset(dtype, FLAGS.rollout.max_buf_size)

    policy = GaussianMLPPolicy(dim_state, dim_action, normalizer=normalizers.state, **FLAGS.policy.as_dict())
    # batched noises
    noise = OUNoise(env.action_space, theta=FLAGS.OUNoise.theta, sigma=FLAGS.OUNoise.sigma, shape=(1, dim_action))
    vfn = MLPVFunction(dim_state, [64, 64], normalizers.state)
    model = DynamicsModel(dim_state, dim_action, normalizers, FLAGS.model.hidden_sizes)
    shadow_models = [DynamicsModel(dim_state, dim_action, normalizers, FLAGS.model.hidden_sizes) for n in range(FLAGS.warmup.n_shadow_models)]

    virt_env = VirtualEnv(model, make_env(FLAGS.env.id, task_config=task), FLAGS.plan.n_envs, opt_model=FLAGS.slbo.opt_model)
    virt_runner = Runner(virt_env, **{**FLAGS.runner.as_dict(), 'max_steps': FLAGS.plan.max_steps})

    shadow_envs = [VirtualEnv(shadow_model, make_env(FLAGS.env.id, task_config=task), FLAGS.plan.n_envs, opt_model=FLAGS.slbo.opt_model) for shadow_model in shadow_models]
    shadow_runners = [Runner(shadow_env, **{**FLAGS.runner.as_dict(), 'max_steps': FLAGS.plan.max_steps}) for shadow_env in shadow_envs]

    criterion_map = {
        'L1': nn.L1Loss(),
        'L2': nn.L2Loss(),
        'MSE': nn.MSELoss(),
    }
    criterion = criterion_map[FLAGS.model.loss]
    loss_mod = MultiStepLoss(model, normalizers, dim_state, dim_action, criterion, FLAGS.model.multi_step)
    loss_mod.build_backward(FLAGS.model.lr, FLAGS.model.weight_decay)
    shadow_loss_mods = [MultiStepLoss(shadow_model, normalizers, dim_state, dim_action, criterion, FLAGS.model.multi_step) for shadow_model in shadow_models]
    for shadow_loss_mod in shadow_loss_mods:
        shadow_loss_mod.build_backward(FLAGS.model.lr, FLAGS.model.weight_decay)
    algo = TRPO(vfn=vfn, policy=policy, dim_state=dim_state, dim_action=dim_action, **FLAGS.TRPO.as_dict())

    tf.get_default_session().run(tf.global_variables_initializer())

    assert FLAGS.algorithm != 'MF', "don't support model free for now"

    runners = {
        'test': make_real_runner(4, task_config=task),
        'collect': make_real_runner(1, task_config=task),
        'dev': make_real_runner(1, task_config=task),
        'train': make_real_runner(FLAGS.plan.n_envs, task_config=task) if FLAGS.algorithm == 'MF' else virt_runner,
    }
    settings = [(runners['test'], policy, 'Real Env'), (runners['train'], policy, 'Virt Env')]
    for (i, runner) in enumerate(shadow_runners):
        settings.append((runner, policy, f'Shadow Env-{i}'))

    saver = nn.ModuleDict({'policy': policy, 'model': model, 'vfn': vfn})
    print(saver)

    if FLAGS.ckpt.model_load:
        saver.load_state_dict(np.load(FLAGS.ckpt.model_load)[()])
        logger.warning('Load model from %s', FLAGS.ckpt.model_load)

    if FLAGS.ckpt.buf_load:
        n_samples = 0
        for i in range(FLAGS.ckpt.buf_load_index):
            data = pickle.load(open(f'{FLAGS.ckpt.buf_load}/stage-{i}.inc-buf.pkl', 'rb'))
            add_multi_step(data, train_set)
            n_samples += len(data)
        logger.warning('Loading %d samples from %s', n_samples, FLAGS.ckpt.buf_load)

    max_ent_coef = FLAGS.TRPO.ent_coef

    if FLAGS.ckpt.buf_load:
        for (model, loss_mod) in zip(shadow_models, shadow_loss_mods):
            losses = deque(maxlen=FLAGS.warmup.n_shadow_model_iters)
            grad_norm_meter = AverageMeter()
            n_model_iters = FLAGS.warmup.n_shadow_model_iters
            for _ in range(n_model_iters):
                samples = train_set.sample_multi_step(FLAGS.model.train_batch_size, 1, FLAGS.model.multi_step)
                _, train_loss, grad_norm = loss_mod.get_loss(
                    samples.state, samples.next_state, samples.action, ~samples.done & ~samples.timeout,
                    fetch='train loss grad_norm')
                losses.append(train_loss.mean())
                grad_norm_meter.update(grad_norm)
                # ideally, we should define an Optimizer class, which takes parameters as inputs.
                # The `update` method of `Optimizer` will invalidate all parameters during updates.
                for param in model.parameters():
                    param.invalidate()


    skip_metrics = []
    TASK_NUM = 0

    while TASK_NUM < FLAGS.task.n_iters:
        logger.info("### STARTING TASK %d ###", TASK_NUM)
        if FLAGS.task.method == 'random':
            task.sample()
            if np.all(np.abs(task.goal_velocity) < 10):
                logger.info('Task Sampled: %s', task)
            else:
                logger.info('Task Sampled: %s', task.__str__())
        elif FLAGS.task.method == 'fixed':
            if FLAGS.task.skip_policy == 'none':
                assert len(FLAGS.task.fixed_velocities) == FLAGS.task.n_iters, f"{len(FLAGS.task.fixed_velocities)} given velocities, but task.n_iters = {FLAGS.task.n_iters}"
            task.goal_velocity = FLAGS.task.fixed_velocities[TASK_NUM]
            if np.all(np.abs(task.goal_velocity) < 10):
                logger.info('Task Fixed: %s', task)
            else:
                logger.info('Task Fixed: %s', task.__str__())

        if FLAGS.task.reset_policy:
            logger.info("Resetting Policy")
            #logger.info(policy.parameters())
            tf.get_default_session().run(tf.variables_initializer(policy.parameters()))


        last_end = None
        drops = []

        logger.info("Training Shadow Models")
        if TASK_NUM > 0 or FLAGS.ckpt.buf_load:
            for (model, loss_mod) in zip(shadow_models, shadow_loss_mods):
                losses = deque(maxlen=FLAGS.warmup.n_shadow_model_iters)
                grad_norm_meter = AverageMeter()
                n_model_iters = FLAGS.warmup.n_shadow_model_iters
                for _ in range(n_model_iters):
                    samples = train_set.sample_multi_step(FLAGS.model.train_batch_size, 1, FLAGS.model.multi_step)
                    _, train_loss, grad_norm = loss_mod.get_loss(
                        samples.state, samples.next_state, samples.action, ~samples.done & ~samples.timeout,
                        fetch='train loss grad_norm')
                    losses.append(train_loss.mean())
                    grad_norm_meter.update(grad_norm)
                    # ideally, we should define an Optimizer class, which takes parameters as inputs.
                    # The `update` method of `Optimizer` will invalidate all parameters during updates.
                    for param in model.parameters():
                        param.invalidate()

        evaluate(settings, 'pre-warm-up')

        for i in range(FLAGS.warmup.n_iters):
            if TASK_NUM == 0:
                if not FLAGS.ckpt.model_load:
                    break

            if i % FLAGS.warmup.n_evaluate_iters == 0 and i != 0:
                # cur_actions = policy.eval('actions_mean actions_std', states=recent_states)
                # kl_old_new = gaussian_kl(*ref_actions, *cur_actions).sum(axis=1).mean()
                # logger.info('KL(old || cur) = %.6f', kl_old_new)
                evaluate(settings, 'iteration')

            losses = deque(maxlen=FLAGS.warmup.n_model_iters)
            grad_norm_meter = AverageMeter()
            n_model_iters = FLAGS.warmup.n_model_iters
            for _ in range(n_model_iters):
                samples = train_set.sample_multi_step(FLAGS.model.train_batch_size, 1, FLAGS.model.multi_step)
                _, train_loss, grad_norm = loss_mod.get_loss(
                    samples.state, samples.next_state, samples.action, ~samples.done & ~samples.timeout,
                    fetch='train loss grad_norm')
                losses.append(train_loss.mean())
                grad_norm_meter.update(grad_norm)
                # ideally, we should define an Optimizer class, which takes parameters as inputs.
                # The `update` method of `Optimizer` will invalidate all parameters during updates.
                for param in model.parameters():
                    param.invalidate()

            if i % FLAGS.model.validation_freq == 0:
                samples = train_set.sample_multi_step(
                    FLAGS.model.train_batch_size, 1, FLAGS.model.multi_step)
                loss = loss_mod.get_loss(
                    samples.state, samples.next_state, samples.action, ~samples.done & ~samples.timeout)
                loss = loss.mean()
                if np.isnan(loss) or np.isnan(np.mean(losses)):
                    logger.info('nan! %s %s', np.isnan(loss), np.isnan(np.mean(losses)))
                logger.info('# Iter %3d: Loss = [train = %.3f, dev = %.3f], after %d steps, grad_norm = %.6f',
                            i, np.mean(losses), loss, n_model_iters, grad_norm_meter.get())

            # losses = deque(maxlen=FLAGS.warmup.n_model_iters)
            # grad_norm_meter = AverageMeter()
            # n_model_iters = FLAGS.warmup.n_model_iters
            # for _ in range(n_model_iters):
            #     samples = train_set.sample_multi_step(FLAGS.model.train_batch_size, 1, FLAGS.model.multi_step)
            #     _, train_loss, grad_norm = shadow_loss_mod.get_loss(
            #         samples.state, samples.next_state, samples.action, ~samples.done & ~samples.timeout,
            #         fetch='train loss grad_norm')
            #     losses.append(train_loss.mean())
            #     grad_norm_meter.update(grad_norm)
            #     # ideally, we should define an Optimizer class, which takes parameters as inputs.
            #     # The `update` method of `Optimizer` will invalidate all parameters during updates.
            #     for param in shadow_model.parameters():
            #         param.invalidate()

            for n_updates in range(FLAGS.warmup.n_policy_iters):
                if FLAGS.algorithm != 'MF' and FLAGS.warmup.start == 'buffer':
                    runners['train'].set_state(train_set.sample(FLAGS.plan.n_envs).state)
                else:
                    runners['train'].reset()

                data, ep_infos = runners['train'].run(policy, FLAGS.plan.n_trpo_samples)
                advantages, values = runners['train'].compute_advantage(vfn, data)
                dist_mean, dist_std, vf_loss = algo.train(max_ent_coef, data, advantages, values)
                returns = [info['return'] for info in ep_infos]
                if n_updates == 0:
                    if last_end is not None:
                        logger.info("DROP: %.10f", last_end - np.mean(returns))
                        drops.append(last_end - np.mean(returns))
                last_end = np.mean(returns)
                logger.info('[TRPO] # %d: n_episodes = %d, returns: {mean = %.0f, std = %.0f}, '
                            'dist std = %.10f, dist mean = %.10f, vf_loss = %.3f',
                            n_updates, len(returns), np.mean(returns), np.std(returns) / np.sqrt(len(returns)),
                            dist_std, dist_mean, vf_loss)

        evaluate(settings, 'post-warm-up')

        logger.info("Task skip policy %s", FLAGS.task.skip_policy)
        if TASK_NUM > 0 or FLAGS.ckpt.model_load: # i.e., we did the warmup
            if FLAGS.task.skip_policy == 'none':
                pass
            elif FLAGS.task.skip_policy == 'drop-mean' or FLAGS.task.skip_policy == 'drop-variance':
                assert len(drops) > 10
                if FLAGS.task.skip_policy == 'drop-mean':
                    logger.info("DROP MEAN %.10f", np.mean(drops[-10]))
                    skip_metrics.append(np.mean(drops[-10]))
                elif FLAGS.task.skip_policy == 'drop-variance':
                    logger.info("DROP STD %.10f", np.std(drops[-10]))
                    skip_metrics.append(np.std(drops[-10]))
                else:
                    raise Exception(f"unknown skip policy {FLAGS.task.skip_policy}")
            elif FLAGS.task.skip_policy == 'shadow-reward-variance':
                shadow_returns = []
                for runner in shadow_runners:
                    data, ep_infos = runner.run(policy, FLAGS.rollout.n_test_samples)
                    returns = [info['return'] for info in ep_infos]
                    shadow_returns.append(np.mean(returns))
                logger.info("Shadow Returns %s, mean=%.10f, std=%.10f", shadow_returns, np.mean(shadow_returns), np.std(shadow_returns))
                skip_metrics.append(np.std(shadow_returns))
            elif FLAGS.task.skip_policy == 'shadow-state-error' or FLAGS.task.skip_policy == 'shadow-state-variance':
                assert FLAGS.warmup.n_shadow_models >= 2

                r0, m1 = shadow_runners[0], shadow_models[1]
                data, ep_infos = r0.run(policy, FLAGS.rollout.n_test_samples)
                state_errs = np.linalg.norm(data['next_state'] - m1.forward(data['state'], data['action']), axis=1)

                if FLAGS.task.skip_policy == 'shadow-state-error':
                    logger.info("SHADOW STATE ERRORS MEAN %.10f", np.mean(state_errs))
                    skip_metrics.append(np.mean(state_errs))
                elif FLAGS.task.skip_policy == 'shadow-state-variance':
                    logger.info("SHADOW STATE ERRORS STD %.10f", np.std(state_errs))
                    skip_metrics.append(np.std(state_errs))
                else:
                    raise Exception(f"unknown skip policy {FLAGS.task.skip_policy}")
            elif FLAGS.task.skip_policy == 'real-reward-diff' or FLAGS.task.skip_policy == 'real-state-err' or FLAGS.task.skip_policy == 'real-state-variance':
                real_data, ep_infos = runners['test'].run(policy, FLAGS.rollout.n_test_samples)
                real_returns = [info['return'] for info in ep_infos]
                _, ep_infos = runners['train'].run(policy, FLAGS.rollout.n_test_samples)
                virt_returns = [info['return'] for info in ep_infos]
                errs = np.linalg.norm(real_data['next_state'] - model.forward(real_data['state'], real_data['action']), axis=1)

                if FLAGS.task.skip_policy == 'real-reward-diff':
                    logger.info("REAL REWARD DIFF %.10f", np.mean(virt_returns) - np.mean(real_returns))
                    skip_metrics.append(np.mean(virt_returns) - np.mean(real_returns))
                elif FLAGS.task.skip_policy == 'real-state-err':
                    logger.info("REAL STATE ERRORS MEAN %.10f", np.mean(errs))
                    skip_metrics.append(np.mean(errs))
                elif FLAGS.task.skip_policy == 'real-state-variance':
                    logger.info("REAL STATE ERRORS STD %.10f", np.std(errs))
                    skip_metrics.append(np.std(errs))
                else:
                    raise Exception(f"unknown skip policy {FLAGS.task.skip_policy}")
            else:
                raise Exception(f"unknown skip policy {FLAGS.task.skip_policy}")
        
        if len(skip_metrics) > 0:
            logger.info("SKIP METRIC %.10f", skip_metrics[-1])

        if len(skip_metrics) >= 11:
            if skip_metrics[-1] < np.median(skip_metrics[-11:-1]):
                logger.info("SKIPPING TASK (%s) %d < %d", FLAGS.task.skip_policy, skip_metrics[-1], np.median(skip_metrics[-11:-1]))
                continue
            else:
                logger.info("PERFORMING TASK (%s) %d > %d", FLAGS.task.skip_policy, skip_metrics[-1], np.median(skip_metrics[-11:-1]))
        
        TASK_NUM += 1
        for T in range(FLAGS.slbo.n_stages):
            logger.info('------ Starting Stage %d --------', T)
            evaluate(settings, 'episode')

            if not FLAGS.use_prev:
                train_set.clear()
                dev_set.clear()

            # collect data
            recent_train_set, ep_infos = runners['collect'].run(noise.make(policy), FLAGS.rollout.n_train_samples)
            add_multi_step(recent_train_set, train_set)
            add_multi_step(
                runners['dev'].run(noise.make(policy), FLAGS.rollout.n_dev_samples)[0],
                dev_set,
            )

            returns = np.array([ep_info['return'] for ep_info in ep_infos])
            if len(returns) > 0:
                logger.info("episode: %s", np.mean(returns))

            if T == 0:  # check
                samples = train_set.sample_multi_step(100, 1, FLAGS.model.multi_step)
                for i in range(FLAGS.model.multi_step - 1):
                    masks = 1 - (samples.done[i] | samples.timeout[i])[..., np.newaxis]
                    assert np.allclose(samples.state[i + 1] * masks, samples.next_state[i] * masks)

            # recent_states = obsvs
            # ref_actions = policy.eval('actions_mean actions_std', states=recent_states)
            if FLAGS.rollout.normalizer == 'policy' or FLAGS.rollout.normalizer == 'uniform' and T == 0:
                normalizers.state.update(recent_train_set.state)
                normalizers.action.update(recent_train_set.action)
                normalizers.diff.update(recent_train_set.next_state - recent_train_set.state)

            if T == 50:
                max_ent_coef = 0.

            for i in range(FLAGS.slbo.n_iters):
                if i % FLAGS.slbo.n_evaluate_iters == 0:# and i != 0:
                    # cur_actions = policy.eval('actions_mean actions_std', states=recent_states)
                    # kl_old_new = gaussian_kl(*ref_actions, *cur_actions).sum(axis=1).mean()
                    # logger.info('KL(old || cur) = %.6f', kl_old_new)
                    evaluate(settings, 'iteration')

                losses = deque(maxlen=FLAGS.slbo.n_model_iters)
                grad_norm_meter = AverageMeter()
                n_model_iters = FLAGS.slbo.n_model_iters
                for _ in range(n_model_iters):
                    samples = train_set.sample_multi_step(FLAGS.model.train_batch_size, 1, FLAGS.model.multi_step)
                    _, train_loss, grad_norm = loss_mod.get_loss(
                        samples.state, samples.next_state, samples.action, ~samples.done & ~samples.timeout,
                        fetch='train loss grad_norm')
                    losses.append(train_loss.mean())
                    grad_norm_meter.update(grad_norm)
                    # ideally, we should define an Optimizer class, which takes parameters as inputs.
                    # The `update` method of `Optimizer` will invalidate all parameters during updates.
                    for param in model.parameters():
                        param.invalidate()

                if i % FLAGS.model.validation_freq == 0:
                    samples = train_set.sample_multi_step(
                        FLAGS.model.train_batch_size, 1, FLAGS.model.multi_step)
                    loss = loss_mod.get_loss(
                        samples.state, samples.next_state, samples.action, ~samples.done & ~samples.timeout)
                    loss = loss.mean()
                    if np.isnan(loss) or np.isnan(np.mean(losses)):
                        logger.info('nan! %s %s', np.isnan(loss), np.isnan(np.mean(losses)))
                    logger.info('# Iter %3d: Loss = [train = %.3f, dev = %.3f], after %d steps, grad_norm = %.6f',
                                i, np.mean(losses), loss, n_model_iters, grad_norm_meter.get())

                for n_updates in range(FLAGS.slbo.n_policy_iters):
                    if FLAGS.algorithm != 'MF' and FLAGS.slbo.start == 'buffer':
                        runners['train'].set_state(train_set.sample(FLAGS.plan.n_envs).state)
                    else:
                        runners['train'].reset()

                    data, ep_infos = runners['train'].run(policy, FLAGS.plan.n_trpo_samples)
                    advantages, values = runners['train'].compute_advantage(vfn, data)
                    dist_mean, dist_std, vf_loss = algo.train(max_ent_coef, data, advantages, values)
                    returns = [info['return'] for info in ep_infos]
                    logger.info('[TRPO] # %d: n_episodes = %d, returns: {mean = %.0f, std = %.0f}, '
                                'dist std = %.10f, dist mean = %.10f, vf_loss = %.3f',
                                n_updates, len(returns), np.mean(returns), np.std(returns) / np.sqrt(len(returns)),
                                dist_std, dist_mean, vf_loss)
            
            if (TASK_NUM*FLAGS.slbo.n_stages + T) % FLAGS.ckpt.n_save_stages == 0:
                np.save(f'{FLAGS.log_dir}/stage-{TASK_NUM*FLAGS.slbo.n_stages + T}', saver.state_dict())
                np.save(f'{FLAGS.log_dir}/final', saver.state_dict())
            if FLAGS.ckpt.n_save_stages == 1:
                pickle.dump(recent_train_set, open(f'{FLAGS.log_dir}/stage-{TASK_NUM*FLAGS.slbo.n_stages + T}.inc-buf.pkl', 'wb'))

        evaluate(settings, 'post-slbo')



if __name__ == '__main__':
    with tf.Session(config=get_tf_config()):
        main()
