# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Installation script for the 'diffaero_env' python package."""

from setuptools import setup

INSTALL_REQUIRES = [
    "psutil",
    "diffaero_common",
    "diffaero_uav",
]

setup(
    name="diffaero_env",
    packages=["diffaero_env"],
    author="DiffAero Team",
    maintainer="DiffAero Team",
    url="https://github.com/diffaero/diffaero-lab",
    version="0.1.0",
    description="IsaacLab task extension for DiffAero drone racing environments",
    keywords=["diffaero", "isaaclab", "drone", "racing", "task"],
    install_requires=INSTALL_REQUIRES,
    license="Apache-2.0",
    include_package_data=True,
    python_requires=">=3.10",
    classifiers=[
        "Natural Language :: English",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Isaac Sim :: 5.0.0",
        "Isaac Sim :: 5.1.0",
        "Isaac Sim :: 6.0.0",
    ],
    zip_safe=False,
)
