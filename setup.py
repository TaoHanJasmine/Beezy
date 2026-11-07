"""Build the optional statically typed BCJR Cython extension in place."""

from pathlib import Path

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, find_packages, setup


ROOT = Path(__file__).resolve().parent

extensions = [
    Extension(
        "Beezy.ecc._bcjr_cython",
        [str(ROOT / "ecc" / "_bcjr_cython.pyx")],
        include_dirs=[np.get_include()],
        extra_compile_args=["-O3"],
    )
]

setup(
    name="beezy-bcjr-kernels",
    packages=find_packages(
        where=str(ROOT.parent),
        include=["Beezy", "Beezy.*"],
        exclude=["Beezy.tests", "Beezy.tests.*"],
    ),
    package_dir={"Beezy": "."},
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            "language_level": "3",
            "boundscheck": False,
            "wraparound": False,
            "initializedcheck": False,
            "cdivision": True,
        },
    ),
)
