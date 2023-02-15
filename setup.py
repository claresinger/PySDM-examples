import platform

from setuptools import find_packages, setup


def get_long_description():
    with open("README.md", "r", encoding="utf8") as file:
        long_description = file.read()
    return long_description


setup(
    name="PySDM-examples",
    description="PySDM usage examples reproducing results from literature "
    "and depicting how to use PySDM from Python Jupyter notebooks",
    use_scm_version={"local_scheme": lambda _: "", "version_scheme": "post-release"},
    setup_requires=["setuptools_scm"],
    install_requires=[
        "PySDM",
        "PyMPDATA",
        "open-atmos-jupyter-utils",
        "pystrict",
        "matplotlib",
        "joblib",
        "ipywidgets",
        "seaborn",
        "ghapi",
    ]
    + (["pyvinecopulib", "vtk"] if platform.architecture()[0] != "32bit" else []),
    author="https://github.com/open-atmos/PySDM/graphs/contributors",
    author_email="slayoo@staszic.waw.pl",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/open-atmos/PySDM-examples",
    license="GPL-3.0",
    packages=find_packages(include=["PySDM_examples", "PySDM_examples.*"]),
    project_urls={
        "Tracker": "https://github.com/open-atmos/PySDM/issues",
        "Documentation": "https://open-atmos.github.io/PySDM-examples",
        "Source": "https://github.com/open-atmos/PySDM-examples",
    },
)
