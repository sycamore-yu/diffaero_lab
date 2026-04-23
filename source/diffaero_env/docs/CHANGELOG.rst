Changelog
=========

0.1.1 (2026-04-15)
------------------

Added
^^^^^

* Added :class:`Isaac-Drone-Racing-Direct-v0` as the PhysX direct workflow drone racing task.
* Added :class:`Isaac-Drone-Racing-Direct-Warp-v0` as the first Warp/Newton drone racing task registration.
* Added pmd, pmc, and simple backend selection support to the drone racing task configuration and bridge layer.

Changed
^^^^^^^

* Changed the drone racing environment to export policy observations, critic observations, task terms, and sim-state metadata through a stable contract for RL and differential learning.
* Changed the Warp route to use backend-aware bridge selection and experimental backend metadata.

0.1.0 (2026-04-14)
------------------
* Initial scaffold for diffaero_env IsaacLab task extension.
