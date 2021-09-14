==============
zfit-tutorials
==============
Tutorials for the zfit project

.. image:: https://mybinder.org/badge.svg
   :target: https://mybinder.org/v2/gh/zfit/zfit-tutorials/master

.. panels::
    :header: text-center
    :img-top-cls: pl-5 pr-5

    :img-top: images/P5p_Value.png

    .. link-button:: tutorials/introduction/README
        :type: ref
        :text: Introduction
        :classes: btn-outline-primary btn-block stretched-link


    ---
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
    :img-top: images/tensorflow-graph.png
    :img-bottom: images/tensorflow-logo-wide.png

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



To start out with zfit, it is recommended to go through `Quickstart with zfit` or the more complete `Introduction`



This repository is structured as follows:
 - **Components**: Tutorials focused on a specific component of zfit. They are rather short and function as a lookup
 - **guides**: More extensive notebooks that go through a full aspect of zfit.

Quickstart with zfit
+++++++++++++++++++++
Guides you through zfit with a minimal example

Introduction
++++++++++++

A more extensive introduction that shows the most crucial aspects.

Components
-----------

This tutorials provide smaller tutorials more specific to certain components


20 Composite Models
+++++++++++++++++++++++

Building models out of other models using sums, products and more is an essential part of model building. This tutorial starts out with the basics of it.

60 Custom PDF
+++++++++++++++++++++++

Being able to build a custom model simply is an essential feature of zfit. This tutorial introduces the two main ways of doing it, a simpler and a more advanced, more flexible way.

62 Multidimensional custom PDF
++++++++++++++++++++++++++++++++++++++++++++++

Building a pdf in multiple dimensions and registering an analytic integral.

80 Toy Study
++++++++++++

A minimal example of how to manually perform toy studies with zfit.

Guides
-------

More extensive guides through a certain topic.

Custom model guide
+++++++++++++++++++

From building a simple custom model to multidimensional models of an angular analysis and functors that depend
on other PDFs and a whole explanation on how models work internally.

Constraints, simultaneous fits, discovery and sPlot
++++++++++++++++++++++++++++++++++++++++++++++++++++

Adding additional knowledge to fits can be done with constraints or through simultaneous fits. Furthermore,
how to make a discovery and use the sPlot technique in conjunction with hepstats is explained.


TensorFlow
-----------

Tutorials about TensorFlow, the zfit backend, itself

Lazy Evaluation, Graphs and TensorFlow
+++++++++++++++++++++++++++++++++++++++++++

An introduction to the declarative programing paradigm and graphs using pure python and TensorFlow.

HPC with TensorFlow
++++++++++++++++++++

Introduction to TensorFlow from a HPC perspective with explanations of the graph and comparison to other frameworks such as Numpy and Numba.
