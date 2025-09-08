.. WeDefense documentation master file, created by
   sphinx-quickstart on Sun Aug  3 16:32:13 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to WeDefense's documentation!
=====================================

WeDefense is a toolkit tailored to defend against fake audio.

.. dropdown:: License considerations (Apache 2.0)

   WeDefense is released under the `Apache License, version 2.0 <https://github.com/zlin0/wedefense/blob/main/LICENSE>`_. The Apache license is a popular BSD-like license.
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

.. dropdown:: Warning / Disclaimer

   Although the goal of this toolkit is to promote progress in fake detection, we acknowledge that any open-source software may be misused by malicious actors. We strongly discourage any use of the models or code for harmful purposes, such as:

   * Bypassing biometric authentication
   * Creating or enhancing systems that generate fake or deceptive audio
   Some of the models included here perform well on public datasets, but may not generalize to unseen or adversarial scenarios. Please be aware of these limitations and use this repository responsibly and ethically.

.. dropdown:: Referencing WeDefense (BibTeX)

   If you use WeDefense in your research or business, please cite it using the following BibTeX entry:

   .. code-block:: bibtex
   @inproceedings{wedefense,
      title={WeDefense: A Toolkit to Defend Against Fake Audio},
      author={Lin Zhang and Johan Rohdin and Xin Wang and Junyi Peng and Tianchi Liu and You Zhang and Hieu-Thi Luong and Shuai Wang and Anna Silnova and Chengdong Liang and Nicholas Evans} and
      year={2025},
   }


.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   installation
   quick_start
   contributing
   faq

.. toctree::
   :maxdepth: 2
   :caption: Tutorial

   tutorials/augmentation
   tutorials/tasks/detection
   tutorials/tasks/localization
   tutorials/deployment/export-with-torch-jit-script



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
