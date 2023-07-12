[![License: GPL v3](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0.html)
[![DOI](https://zenodo.org/badge/351755603.svg)](https://zenodo.org/badge/latestdoi/351755603)   

[![Github Actions Build Status](https://github.com/atmos-cloud-sim-uj/PySDM-examples/workflows/main/badge.svg?branch=main)](https://github.com/atmos-cloud-sim-uj/PySDM-examples/actions)
[![PyPI version](https://badge.fury.io/py/PySDM-examples.svg)](https://pypi.org/project/PySDM-examples)
[![API docs](https://img.shields.io/badge/API_docs-pdoc3-blue.svg)](https://open-atmos.github.io/PySDM-examples/)

This repository stores example files for `PySDM` depicting usage of `PySDM` from Python via Jupyter.
For information on the `PySDM` package itself and examples of usage from Julia and Matlab, 
see [PySDM README.md](https://github.com/open-atmos/PySDM/blob/master/README.md) file.

Please use the [PySDM issue-tracking](https://github.com/open-atmos/PySDM/issues) and [discussion](https://github.com/open-atmos/PySDM/discussions) infrastructure for `PySDM-examples` as well.

### 0D box-model coalescence and breakup examples:
- [Shima et al. 2009](http://doi.org/10.1002/qj.441) (Box model, coalescence only, test case employing Golovin analytical solution):
    - Fig. 2:
      [![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.jupyter.org/github/open-atmos/PySDM-examples/blob/main/PySDM_examples/Shima_et_al_2009/fig_2.ipynb)
      [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/open-atmos/PySDM-examples.git/main?urlpath=lab/tree/PySDM_examples/Shima_et_al_2009/fig_2.ipynb)
      [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/open-atmos/PySDM-examples/blob/main/PySDM_examples/Shima_et_al_2009/fig_2.ipynb)    
  
- [Berry 1967](https://doi.org/10.1175/1520-0469(1967)024<0688:CDGBC>2.0.CO;2) (Box model, coalescence only, test cases for realistic kernels):
    - Figs. 5, 8 & 10:
     [![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.jupyter.org/github/open-atmos/PySDM-examples/blob/main/PySDM_examples/Berry_1967/figs_5_8_10.ipynb) 
     [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/open-atmos/PySDM-examples.git/main?urlpath=lab/tree/PySDM_examples/Berry_1967/figs_5_8_10.ipynb)
     [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/open-atmos/PySDM-examples/blob/main/PySDM_examples/Berry_1967/figs_5_8_10.ipynb)

- [Bieli et al. 2022](https://www.essoar.org/doi/abs/10.1002/essoar.10510248.1) (Box model, coalescence and breakup with fixed coalescence efficiency):
    - Fig. 3:
      [![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.jupyter.org/github/open-atmos/PySDM-examples/blob/main/PySDM_examples/Bieli_et_al_2022/make_fig_3.ipynb)
      [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/open-atmos/PySDM-examples.git/main?urlpath=lab/tree/PySDM_examples/Bieli_et_al_2022/make_fig_3.ipynb)
      [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/open-atmos/PySDM-examples/blob/main/PySDM_examples/Bieli_et_al_2022/make_fig_3.ipynb)

- [deJong et al. 2023](https://doi.org/10.5194/egusphere-2022-1243) (Box model, coalescence and breakup):
    - Figs. 3-5 (Berry 1967 coalescence efficiency):
      [![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.jupyter.org/github/open-atmos/PySDM-examples/blob/main/PySDM_examples/deJong_Mackay_2022/box_sensitivities.ipynb)
      [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/open-atmos/PySDM-examples.git/main?urlpath=lab/tree/PySDM_examples/deJong_Mackay_2022/box_sensitivities.ipynb)
      [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/open-atmos/PySDM-examples/blob/main/PySDM_examples/deJong_Mackay_2022/box_sensitivities.ipynb)
    - Appendix figures (validation against Srivastava 1982 analytic solution):
      [![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.jupyter.org/github/open-atmos/PySDM-examples/blob/main/PySDM_examples/deJong_Mackay_2022/appendix.ipynb)
      [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/open-atmos/PySDM-examples.git/main?urlpath=lab/tree/PySDM_examples/deJong_Mackay_2022/appendix.ipynb)
      [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/open-atmos/PySDM-examples/blob/main/PySDM_examples/deJong_Mackay_2022/appendix.ipynb)
  
### 0D box-model immersion freezing-only example:
- [Alpert & Knopf 2016](https://doi.org/10.5194/acp-16-2083-2016) (stochastic immersion freezing with monodisperse vs. lognormal immersed surface areas):
  - Fig. 1:
      [![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.jupyter.org/github/open-atmos/PySDM-examples/blob/main/PySDM_examples/Alpert_and_Knopf_2016/fig_1.ipynb)
      [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/open-atmos/PySDM-examples.git/main?urlpath=lab/tree/PySDM_examples/Alpert_and_Knopf_2016/fig_1.ipynb)
      [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/open-atmos/PySDM-examples/blob/main/PySDM_examples/Alpert_and_Knopf_2016/fig_1.ipynb)    
  - Fig. 2:
      [![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.jupyter.org/github/open-atmos/PySDM-examples/blob/main/PySDM_examples/Alpert_and_Knopf_2016/fig_2.ipynb)
      [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/open-atmos/PySDM-examples.git/main?urlpath=lab/tree/PySDM_examples/Alpert_and_Knopf_2016/fig_2.ipynb)
      [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/open-atmos/PySDM-examples/blob/main/PySDM_examples/Alpert_and_Knopf_2016/fig_2.ipynb)    
  - Fig. 3:
      [![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.jupyter.org/github/open-atmos/PySDM-examples/blob/main/PySDM_examples/Alpert_and_Knopf_2016/fig_3.ipynb)
      [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/open-atmos/PySDM-examples.git/main?urlpath=lab/tree/PySDM_examples/Alpert_and_Knopf_2016/fig_3.ipynb)
      [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/open-atmos/PySDM-examples/blob/main/PySDM_examples/Alpert_and_Knopf_2016/fig_3.ipynb)    
  - Fig. 4: 
      [![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.jupyter.org/github/open-atmos/PySDM-examples/blob/main/PySDM_examples/Alpert_and_Knopf_2016/fig_4.ipynb)
      [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/open-atmos/PySDM-examples.git/main?urlpath=lab/tree/PySDM_examples/Alpert_and_Knopf_2016/fig_4.ipynb)
      [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/open-atmos/PySDM-examples/blob/main/PySDM_examples/Alpert_and_Knopf_2016/fig_4.ipynb)    
  - Fig. 5:
      [![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.jupyter.org/github/open-atmos/PySDM-examples/blob/main/PySDM_examples/Alpert_and_Knopf_2016/fig_5.ipynb)
      [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/open-atmos/PySDM-examples.git/main?urlpath=lab/tree/PySDM_examples/Alpert_and_Knopf_2016/fig_5.ipynb)
      [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/open-atmos/PySDM-examples/blob/main/PySDM_examples/Alpert_and_Knopf_2016/fig_5.ipynb)    
  
### 0D parcel-model condensation only examples:
- [Arabas & Shima 2017](http://dx.doi.org/10.5194/npg-24-535-2017) (monodisperse size spectrum activation/deactivation test case):
  - Fig. 5:
    [![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.jupyter.org/github/open-atmos/PySDM-examples/blob/main/PySDM_examples/Arabas_and_Shima_2017/fig_5.ipynb)
    [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/open-atmos/PySDM-examples.git/main?urlpath=lab/tree/PySDM_examples/Arabas_and_Shima_2017/fig_5.ipynb)
    [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/open-atmos/PySDM-examples/blob/main/PySDM_examples/Arabas_and_Shima_2017/fig_5.ipynb)    
  
- [Yang et al. 2018](https://doi.org/10.5194/acp-18-7313-2018) (polydisperse size spectrum activation/deactivation test case):
  - Fig. 2:
    [![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.jupyter.org/github/open-atmos/PySDM-examples/blob/main/PySDM_examples/Yang_et_al_2018/fig_2.ipynb)
    [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/open-atmos/PySDM-examples.git/main?urlpath=lab/tree/PySDM_examples/Yang_et_al_2018/fig_2.ipynb)
    [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/open-atmos/PySDM-examples/blob/main/PySDM_examples/Yang_et_al_2018/fig_2.ipynb)

- [Abdul-Razzak & Ghan 2000](http://doi.wiley.com/10.1029/1999JD901161) (aerosol activation parameterization for GCMs):
  - Figs. 1 - 5:
    [![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.jupyter.org/github/open-atmos/PySDM-examples/blob/main/PySDM_examples/Abdul_Razzak_Ghan_2000/figs1-5.ipynb)
    [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/open-atmos/PySDM-examples.git/main?urlpath=PySDM_examples/Abdul_Razzak_Ghan_2000/figs1-5.ipynb)
    [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/open-atmos/PySDM-examples/blob/main/PySDM_examples/Abdul_Razzak_Ghan_2000/figs1-5.ipynb)

- [Pyrcel documentation example](https://pyrcel.readthedocs.io/en/latest/examples/basic_run.html) (externally mixed polydisperse size spectrum activation test case):
  - supersaturation, temperature, wet radii evolution and dry spectra plots:
    [![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.jupyter.org/github/open-atmos/PySDM-examples/blob/main/PySDM_examples/Pyrcel/example_basic_run.ipynb)
    [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/open-atmos/PySDM-examples.git/main?urlpath=PySDM_examples/Pyrcel/example_basic_run.ipynb)
    [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/open-atmos/PySDM-examples/blob/main/PySDM_examples/Pyrcel/example_basic_run.ipynb)

- [Lowe et al. 2019](https://doi.org/10.1038/s41467-019-12982-0) (externally mixed polydisperse size spectrum with surface-active organics case):
  - Fig. 1: 
    [![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.jupyter.org/github/open-atmos/PySDM-examples/blob/main/PySDM_examples/Lowe_et_al_2019/fig_1.ipynb)
    [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/open-atmos/PySDM-examples.git/main?urlpath=lab/tree/PySDM_examples/Lowe_et_al_2019/fig_1.ipynb)
    [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/open-atmos/PySDM-examples/blob/main/PySDM_examples/Lowe_et_al_2019/fig_1.ipynb)
  - Fig. 2:
    [![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.jupyter.org/github/open-atmos/PySDM-examples/blob/main/PySDM_examples/Lowe_et_al_2019/fig_2.ipynb)
    [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/open-atmos/PySDM-examples.git/main?urlpath=lab/tree/PySDM_examples/Lowe_et_al_2019/fig_2.ipynb)
    [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/open-atmos/PySDM-examples/blob/main/PySDM_examples/Lowe_et_al_2019/fig_2.ipynb)
  - Fig. 3:
    [![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.jupyter.org/github/open-atmos/PySDM-examples/blob/main/PySDM_examples/Lowe_et_al_2019/fig_3.ipynb)
    [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/open-atmos/PySDM-examples.git/main?urlpath=lab/tree/PySDM_examples/Lowe_et_al_2019/fig_3.ipynb)
    [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/open-atmos/PySDM-examples/blob/main/PySDM_examples/Lowe_et_al_2019/fig_3.ipynb)

### 0D parcel-model condensation/aqueous-chemistry example:
- [Kreidenweis et al. 2003](https://doi.org/10.1029/2002JD002697) (Adiabatic parcel, polydisperse size spectrum, aqueous‐phase SO2 oxidation test case):
  - Fig 1:
    [![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.jupyter.org/github/open-atmos/PySDM-examples/blob/main/PySDM_examples/Kreidenweis_et_al_2003/fig_1.ipynb)
    [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/open-atmos/PySDM-examples.git/main?urlpath=lab/tree/PySDM_examples/Kreidenweis_et_al_2003/fig_1.ipynb)
    [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/open-atmos/PySDM-examples/blob/main/PySDM_examples/Kreidenweis_et_al_2003/fig_1.ipynb)
- [Jaruga and Pawlowska 2018](https://doi.org/10.5194/gmd-11-3623-2018) (same test case as above, different numerical settings):
  - Fig 2:
    [![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.jupyter.org/github/open-atmos/PySDM-examples/blob/main/PySDM_examples/Jaruga_and_Pawlowska_2018/fig_2.ipynb)
    [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/open-atmos/PySDM-examples.git/main?urlpath=lab/tree/PySDM_examples/Jaruga_and_Pawlowska_2018/fig_2.ipynb)
    [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/open-atmos/PySDM-examples/blob/main/PySDM_examples/Jaruga_and_Pawlowska_2018/fig_2.ipynb)    
  - Fig 3:
    [![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.jupyter.org/github/open-atmos/PySDM-examples/blob/main/PySDM_examples/Jaruga_and_Pawlowska_2018/fig_3.ipynb)
    [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/open-atmos/PySDM-examples.git/main?urlpath=lab/tree/PySDM_examples/Jaruga_and_Pawlowska_2018/fig_3.ipynb)
    [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/open-atmos/PySDM-examples/blob/main/PySDM_examples/Jaruga_and_Pawlowska_2018/fig_3.ipynb)    

### 1D kinematic (prescribed-flow, single-column):  
- [Shipway & Hill 2012](https://doi.org/10.1002/qj.1913):
  - Fig 1 (thermodynamics/condensation only, no particle displacement yet):   
    [![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.jupyter.org/github/open-atmos/PySDM-examples/blob/main/PySDM_examples/Shipway_and_Hill_2012/fig_1.ipynb)
    [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/open-atmos/PySDM-examples.git/main?urlpath=lab/tree/PySDM_examples/Shipway_and_Hill_2012/fig_1.ipynb)
    [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/open-atmos/PySDM-examples/blob/main/PySDM_examples/Shipway_and_Hill_2012/fig_1.ipynb)

- [deJong et al. 2023](https://doi.org/10.5194/egusphere-2022-1243) (Kinematic setup as in Shipway and Hill, including breakup with Berry 1967 coalescence efficiency):
    - Figs. 6-8:
      [![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.jupyter.org/github/open-atmos/PySDM-examples/blob/main/PySDM_examples/deJong_Mackay_2022/rainshaft_1D.ipynb)
      [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/open-atmos/PySDM-examples.git/main?urlpath=lab/tree/PySDM_examples/deJong_Mackay_2022/rainshaft_1D.ipynb)
      [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/open-atmos/PySDM-examples/blob/main/PySDM_examples/deJong_Mackay_2022/rainshaft_1D.ipynb)

### 2D kinematic (prescribed-flow) Sc-mimicking aerosol collisional processing (warm-rain) examples:
- [Arabas et al. 2015](https://doi.org/10.5194/gmd-8-1677-2015) 
  - Figs. 8 & 9 (interactive web-GUI with product selection, parameter sliders and netCDF/plot export buttons):    
    [![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.jupyter.org/github/open-atmos/PySDM-examples/blob/main/PySDM_examples/Arabas_et_al_2015/gui.ipynb)
    [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/open-atmos/PySDM-examples.git/main?urlpath=lab/tree/PySDM_examples/Arabas_et_al_2015/gui.ipynb)
    [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/open-atmos/PySDM-examples/blob/main/PySDM_examples/Arabas_et_al_2015/gui.ipynb)       
  
- Bartman et al. 2021 (in preparation):
  - Fig 1 (default-settings based script generating a netCDF file and loading it subsequently to create the animation below):    
    [![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.jupyter.org/github/open-atmos/PySDM-examples/blob/main/PySDM_examples/Bartman_et_al_2021/demo.ipynb)
    [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/open-atmos/PySDM-examples.git/main?urlpath=lab/tree/PySDM_examples/Bartman_et_al_2021/demo.ipynb)
    [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/open-atmos/PySDM-examples/blob/main/PySDM_examples/Bartman_et_al_2021/demo.ipynb)       
  - Fig 2:
    [![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.jupyter.org/github/open-atmos/PySDM-examples/blob/main/PySDM_examples/Bartman_et_al_2021/demo_fig2.ipynb)
    [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/open-atmos/PySDM-examples.git/main?urlpath=lab/tree/PySDM_examples/Bartman_et_al_2021/demo_fig2.ipynb)
    [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/open-atmos/PySDM-examples/blob/main/PySDM_examples/Bartman_et_al_2021/demo_fig2.ipynb)
  - Fig 3:
    [![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.jupyter.org/github/open-atmos/PySDM-examples/blob/main/PySDM_examples/Bartman_et_al_2021/demo_fig3.ipynb)
    [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/open-atmos/PySDM-examples.git/main?urlpath=lab/tree/PySDM_examples/Bartman_et_al_2021/demo_fig3.ipynb)
    [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/open-atmos/PySDM-examples/blob/main/PySDM_examples/Bartman_et_al_2021/demo_fig3.ipynb)       
  
![animation](https://github.com/open-atmos/PySDM/wiki/files/kinematic_2D_example.gif)
