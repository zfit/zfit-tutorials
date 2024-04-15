==============
zfit tutorials
==============

A collection of interactive guides and tutorials
for the zfit project.



.. panels::
    :header: text-center
    :img-top-cls: pl-2 pr-2 bw-success

    :img-top: images/P5p_Value.png

    .. link-button:: tutorials/introduction/README
        :type: ref
        :text: Introduction
        :classes: btn-outline-primary btn-block stretched-link


    ---
    :img-top-cls: + pt-4
    :img-top: images/zfit_workflow_v2.png
    .. link-button:: tutorials/components/README
        :type: ref
        :text: Components
        :classes: btn-outline-primary btn-block stretched-link

    ---
    :img-top: images/hepstats-pvalue.png

    .. link-button:: tutorials/guides/README
        :type: ref
        :text: Guides
        :classes: btn-outline-primary btn-block stretched-link

    ---
    :img-top-cls: + pt-4
    :img-top: images/logo_graph_tensorflow.png

    .. link-button:: tutorials/TensorFlow/README
        :type: ref
        :text: TensorFlow
        :classes: btn-outline-primary btn-block stretched-link


.. toctree::
    :maxdepth: 2

    tutorials/introduction/README
    tutorials/components/README
    tutorials/guides/README
    tutorials/TensorFlow/README



| **Components** focuse on a specific component of zfit. They are rather short and function as a lookup
| **guides** are more extensive notebooks that go through several aspects of zfit and combine it with hepstats.


They can all be launched (rocket button upper right), but you can also launch all of them in

.. image:: https://mybinder.org/badge.svg
   :target: https://mybinder.org/v2/gh/zfit/zfit-tutorials/main
