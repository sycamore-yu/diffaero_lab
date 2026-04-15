Changelog
=========

0.1.1 (2026-04-15)
------------------

Added
^^^^^

* Added reusable quadrotor, point-mass discrete, point-mass continuous, and simplified quadrotor dynamics models for DiffAero tasks.
* Added Isaac Lab and Newton adapter entry points for UAV backend integration.

Changed
^^^^^^^

* Changed the Warp route to instantiate a dedicated Newton backend adapter that exposes Warp backend metadata and reads live robot state when available.

0.1.0 (2026-04-14)
------------------
* Initial scaffold for diffaero_uav quadrotor dynamics extension.
