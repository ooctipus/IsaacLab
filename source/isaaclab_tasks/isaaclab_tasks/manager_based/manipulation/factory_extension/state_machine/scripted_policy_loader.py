def state_machine_loader(sm_path: str, env):
    import importlib

    mod, attr_name = sm_path.split(":")
    cls = getattr(importlib.import_module(mod), attr_name)
    sm = cls()
    sm.initialize(env.cfg.sim.dt * env.cfg.decimation, env.unwrapped.num_envs, env.unwrapped.device)

    class ScriptedPolicy:
        def __init__(self):
            pass

        def __call__(self, *args, **kwds):
            return sm.compute(env)

        def reset(self, env_ids, *args, **kwds):
            sm.reset_idx(env_ids)

        def to(self, device):
            return self

        def eval(self):
            return self

    return ScriptedPolicy()
