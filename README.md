# RBF VARS(Visual Analysis for Rashomon Set)
This is the GitHub Repository for supporting paper *"VARS: Visual Analysis for Rashomon Set of Machine Learning
Models"*. VARS is short for Visual Analysis for Rashomon Set. It is built upon an ML model dataset called the Rashomon dataset. If an ML model is in the Rashomon dataset, it has a similar performance to other machine learning models.

Key Characteristics of a Rashomon Dataset:
===
Multiple Valid Models: The dataset should allow for the development of several models that all perform well but provide different insights or explanations.  
Complexity: The dataset may include intricate relationships and features that can be interpreted in multiple ways.  
Diversity in Model Interpretations: Different models trained on the dataset should yield diverse interpretations or feature importances.  
Sensitivity Analysis: The dataset should facilitate analysis of how small changes in data or model parameters affect the outputs and interpretations.  

Goal:
===
Model Interpretability: To understand how different models interpret the same data and to study the robustness of model explanations.    
Bias and Fairness: To explore how different models might exhibit biases or fairness issues in various ways despite similar performance.    
Ensemble Methods: To analyze the benefits of combining multiple models to achieve better generalization and robustness.  

## Installation

The code requires Python >= 3.8  
NumPy: 1.19.5  
Matplotlib: 3.2.2  
Pandas: 1.1.5.  

if not please use:
```
pip install numpy
pip install matplotlib
pip install pandas
```

## License

VAM code released under the Apache License 2.0. See [LICENSE](LICENSE) for additional details.

## Citing RIT ٩(๑>◡<๑)۶

If you find this repository useful, please consider giving a star :star: and citation the following related papers:

```
@misc{Jin:2024:igaiva,
      title={iGAiVA: Integrated Generative AI and Visual Analytics in a Machine Learning Workflow for Text Classification}, 
      author={Yuanzhe Jin and Adrian Carrasco-Revilla and Min Chen},
      year={2024},
      eprint={2409.15848},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2409.15848}, 
}
```
