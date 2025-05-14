# CIRCE

This is the repository for the implementations of the paper [*CIRCE: a Scalable Methodology for Causal Explanations in Cyber-Physical Systems*](https://ieeexplore.ieee.org/document/10771469).

## What is CIRCE

Cyber-physical systems (CPS) are increasingly complex and harder for human users to understand. Integrating explainability methods within their design is a key challenge for their acceptability and management. We consider that causal explanations can provide suitable answers to address this issue. Most approaches to causal explanations, however, rely on global system models, often built offline, which implies heavy computations, delays, and interpretability issues when answering questions at runtime. We propose CIRCE: a scalable method for Contextual, Interpretable and Reactive Causal Explanations in CPS. It is an abduction method that determines the cause of a fact questioned by users at runtime. Its originality lies in finding a cause instead of an entire causal graph to explain CPS behavior and employing a classic local Explanatory AI (XAI) technique to approximate this cause. We validate our method via several simulations of smart home scenarios. Results indicate that CIRCE can provide relevant answers to diverse questions and scales well with the number of variables. Our approach may improve the efficiency and relevance of causality based explanations for CPS and contribute to bridging the gap between CPS explainability and classic XAI techniques.

## How to use

To use this code, you can clone the repository. You can setup an environment for jupyter notbook using:

```
    source setup.sh
```

The code for the method can be found in the [CIRCE](src/CIRCE.py) file. You can find examples of usage or render the result figures in [this notebook](src/CIRCE.ipynb). 

To reproduce the experiment from the paper, run:

```
    python src/quantitative_experiment.py
```

## Citation

If you use this code for an academic work, please use the following citation:

S. Reyd, A. Diaconescu and J. -L. Dessalles, "CIRCE: a Scalable Methodology for Causal Explanations in Cyber-Physical Systems," 2024 IEEE International Conference on Autonomic Computing and Self-Organizing Systems (ACSOS), Aarhus, Denmark, 2024, pp. 81-90, doi: 10.1109/ACSOS61780.2024.00026.

@INPROCEEDINGS{ReydCIRCE,author={Reyd, Samuel and Diaconescu, Ada and Dessalles, Jean-Louis},booktitle={2024 IEEE International Conference on Autonomic Computing and Self-Organizing Systems (ACSOS)}, title={CIRCE: a Scalable Methodology for Causal Explanations in Cyber-Physical Systems}, year={2024},pages={81-90},doi={10.1109/ACSOS61780.2024.00026}}
