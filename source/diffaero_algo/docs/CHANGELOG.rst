Changelog
=========

0.1.1 (2026-04-15)
------------------

Added
^^^^^

* Added deterministic and stochastic APG training paths for the direct drone racing environment.
* Added SHAC and SHA2C actor-critic training paths on top of critic observations from the drone racing task.
* Added differential training entry scripts for APG, APG stochastic, SHAC, and SHA2C.

Changed
^^^^^^^

* Changed differential environment adapters to consume shared contract keys and backend-aware sim-state metadata.
* Changed the APG entry path to support the Warp/Newton task route through task-based dispatch.

0.1.0 (2026-04-14)
------------------
* Initial scaffold for diffaero_algo differential learning extension.
