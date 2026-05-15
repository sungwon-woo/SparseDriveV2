from mmcv.runner import HOOKS, Hook


@HOOKS.register_module()
class TauAnnealingHook(Hook):
    """Linear annealing of router softmax temperature `tau` over training iterations.

    Walks the model and sets `mod.tau` on any submodule whose class name
    contains "MoE", "Router", or "LatLonPred" and exposes a `tau` attribute.
    Designed for IterBasedRunner; mutates every `update_interval` iters.

    Args:
        tau_start: tau at iter 0
        tau_end: tau at `anneal_iters` (and after)
        anneal_iters: iter count over which tau linearly anneals
        update_interval: how often (in iters) to refresh tau on the model
        target_attr: attribute name to set (default "tau")
    """

    def __init__(
        self,
        tau_start: float = 1.0,
        tau_end: float = 0.3,
        anneal_iters: int = 30000,
        update_interval: int = 200,
        target_attr: str = "tau",
    ):
        self.tau_start = float(tau_start)
        self.tau_end = float(tau_end)
        self.anneal_iters = max(int(anneal_iters), 1)
        self.update_interval = max(int(update_interval), 1)
        self.target_attr = target_attr

    def _current_tau(self, it: int) -> float:
        if it >= self.anneal_iters:
            return self.tau_end
        ratio = it / self.anneal_iters
        return self.tau_start + (self.tau_end - self.tau_start) * ratio

    def _apply(self, model, tau: float) -> int:
        n = 0
        for mod in model.modules():
            cls = type(mod).__name__
            if any(k in cls for k in ("MoE", "Router", "LatLonPred")):
                if hasattr(mod, self.target_attr):
                    setattr(mod, self.target_attr, tau)
                    n += 1
        return n

    def before_run(self, runner):
        model = runner.model.module if hasattr(runner.model, "module") else runner.model
        tau = self._current_tau(runner.iter)
        n = self._apply(model, tau)
        if runner.rank == 0:
            runner.logger.info(
                f"[TauAnnealingHook] init iter={runner.iter} tau={tau:.4f} applied_to={n}"
            )

    def before_train_iter(self, runner):
        if runner.iter % self.update_interval != 0:
            return
        model = runner.model.module if hasattr(runner.model, "module") else runner.model
        tau = self._current_tau(runner.iter)
        n = self._apply(model, tau)
        if runner.rank == 0 and n > 0:
            runner.logger.info(
                f"[TauAnnealingHook] iter={runner.iter} tau={tau:.4f} applied_to={n}"
            )
