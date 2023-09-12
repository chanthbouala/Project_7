<div align="center">
<h1 align="center">
<img src="https://raw.githubusercontent.com/PKief/vscode-material-icon-theme/ec559a9f6bfd399b82bb44393651661b08aaf7ba/icons/folder-markdown-open.svg" width="100" />
<br>Project_7
<br>Implement a scoring model for bank loan
</h1>
<h3>◦ Developed with the software and tools listed below.</h3>

<p align="center">
<img src="https://img.shields.io/badge/tqdm-FFC107.svg?style&logo=tqdm&logoColor=black" alt="tqdm" />
<img src="https://img.shields.io/badge/Streamlit-FF4B4B.svg?style&logo=Streamlit&logoColor=white" alt="Streamlit" />
<img src="https://img.shields.io/badge/Babel-F9DC3E.svg?style&logo=Babel&logoColor=black" alt="Babel" />
<img src="https://img.shields.io/badge/scikitlearn-F7931E.svg?style&logo=scikit-learn&logoColor=white" alt="scikitlearn" />
<img src="https://img.shields.io/badge/Jupyter-F37626.svg?style&logo=Jupyter&logoColor=white" alt="Jupyter" />
<img src="https://img.shields.io/badge/Jinja-B41717.svg?style&logo=Jinja&logoColor=white" alt="Jinja" />
<img src="https://img.shields.io/badge/SciPy-8CAAE6.svg?style&logo=SciPy&logoColor=white" alt="SciPy" />
<img src="https://img.shields.io/badge/Gunicorn-499848.svg?style&logo=Gunicorn&logoColor=white" alt="Gunicorn" />

<img src="https://img.shields.io/badge/Plotly-3F4F75.svg?style&logo=Plotly&logoColor=white" alt="Plotly" />
<img src="https://img.shields.io/badge/Python-3776AB.svg?style&logo=Python&logoColor=white" alt="Python" />
<img src="https://img.shields.io/badge/pandas-150458.svg?style&logo=pandas&logoColor=white" alt="pandas" />
<img src="https://img.shields.io/badge/NumPy-013243.svg?style&logo=NumPy&logoColor=white" alt="NumPy" />
<img src="https://img.shields.io/badge/Numba-00A3E0.svg?style&logo=Numba&logoColor=white" alt="Numba" />
<img src="https://img.shields.io/badge/FastAPI-009688.svg?style&logo=FastAPI&logoColor=white" alt="FastAPI" />
<img src="https://img.shields.io/badge/Markdown-000000.svg?style&logo=Markdown&logoColor=white" alt="Markdown" />
</p>
<img src="https://img.shields.io/github/languages/top/chanthbouala/Project_7?style&color=5D6D7E" alt="GitHub top language" />
<img src="https://img.shields.io/github/languages/code-size/chanthbouala/Project_7?style&color=5D6D7E" alt="GitHub code size in bytes" />
<img src="https://img.shields.io/github/commit-activity/m/chanthbouala/Project_7?style&color=5D6D7E" alt="GitHub commit activity" />
<img src="https://img.shields.io/github/license/chanthbouala/Project_7?style&color=5D6D7E" alt="GitHub license" />
</div>

---

## 📒 Table of Contents
- [📒 Table of Contents](#-table-of-contents)
- [📍 Overview](#-overview)
- [⚙️ Features](#-features)
- [📂 Project Structure](#project-structure)
- [🧩 Modules](#modules)
- [🚀 Getting Started](#-getting-started)
- [🗺 Roadmap](#-roadmap)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)
- [👏 Acknowledgments](#-acknowledgments)

---


## 📍 Overview

### Mission: Implement a scoring model for bank loan
**Context**  
You are a Data Scientist working for a financial company called "Prêt à dépenser", which offers consumer credit to people with little or no loan history. The company wanted to implement a "credit scoring" tool to calculate the probability of a customer repaying their loan, and then classify the application as either granted or refused credit. It therefore wants to develop a classification algorithm based on a variety of data sources (behavioural data, data from other financial institutions, etc.). In addition, customer relationship managers have pointed out that customers are increasingly demanding transparency in credit granting decisions. This demand for transparency from customers is entirely in line with the values that the company wishes to embody. "Prêt à dépenser" has therefore decided to develop an interactive dashboard so that customer relations managers can not only explain credit granting decisions as transparently as possible, but also allow their customers to access their personal information and explore it easily. 

**The data**  
Here is the [data](https://www.kaggle.com/c/home-credit-default-risk/data) you will need to create the dashboard. You will probably need to join the different tables together. 

**Your task**  
1. Build a scoring model that will automatically predict the probability of a customer going bankrupt. 
2. Build an interactive dashboard for customer relations managers to interpret the predictions made by the model, and to improve the customer knowledge of customer relations managers.

Michaël, your manager, has encouraged you to select a Kaggle kernel to make it easier for you to prepare the data needed to build the scoring model. You will analyse this kernel and adapt it to ensure that it meets the needs of your assignment. You will then be able to focus on developing, optimising and understanding the model. 

**Dashboard specifications**  
Michaël has provided you with specifications for the interactive dashboard. It should contain at least the following functions: 
1. Display the score and the interpretation of this score for each customer in a way that is intelligible to someone who is not an expert in data science. 
2. Display descriptive information about a customer (using a filter system). 
3. Enable descriptive information about a customer to be compared with all customers or with a group of similar customers. 

**Deliverables**  
1. The **interactive dashboard** meeting the above specifications and the score prediction API, both deployed in the cloud. 
2. A **file** on a code versioning tool containing: 
- The modelling code (from pre-processing to prediction) 
- The code generating the dashboard 
- The code enabling the model to be deployed in the form of an API 
3. A **methodological note** describing: 
- The model training methodology (2 pages maximum) 
- The business cost function, the optimisation algorithm and the evaluation metric (1 page maximum) 
- The global and local interpretability of the model (1 page maximum) 
- The limitations and possible improvements (1 page maximum) 
4. A **presentation aid** for the presentation, detailing the work carried out.

---

## ⚙️ Features


---


## 📂 Project Structure




---

## 🧩 Modules

<details closed><summary>Root</summary>

| File                                                                                                              | Summary                                 |
| ---                                                                                                               | ---                                     |
| [fastAPI_backend.py](https://github.com/chanthbouala/Project_7/blob/main/P7_API\fastAPI_backend.py)               | HTTPStatus Exception: 429               |
| [functions.py](https://github.com/chanthbouala/Project_7/blob/main/P7_API\functions.py)                           | HTTPStatus Exception: 429               |
| [Procfile](https://github.com/chanthbouala/Project_7/blob/main/P7_API\Procfile)                                   | HTTPStatus Exception: 429               |
| [Dashboard_streamlit.py](https://github.com/chanthbouala/Project_7/blob/main/P7_dashboard\Dashboard_streamlit.py) | HTTPStatus Exception: 429               |
| [functions.py](https://github.com/chanthbouala/Project_7/blob/main/P7_dashboard\functions.py)                     | HTTPStatus Exception: 429               |
| [P7.ipynb](https://github.com/chanthbouala/Project_7/blob/main/P7_notebook\P7.ipynb)                              | Prompt exceeds max token limit: 872857. |

</details>

---

## 🚀 Getting Started

### ✔️ Prerequisites


### 📦 Installation

1. Clone the P7_notebook repository:
```sh
git clone https://github.com/chanthbouala/Project_7
```

2. Change to the project directory:
```sh
cd Project_7
```

3. Install the dependencies:
```sh
pip install -r requirements.txt
```

### 🎮 Using P7_notebook

```sh
jupyter lab P7.ipynb
```

### 🎮 Using fastAPI_backend.py
The API backend is meant to be hosted at Heroku.

### 🎮 Using Dashboard_streamlit.py
The dashboard code is using using streamlit and is meant to be hosted at streamlit services.

---


## 🗺 Roadmap

---

## 🤝 Contributing

Contributions are always welcome! Please follow these steps:
1. Fork the project repository. This creates a copy of the project on your account that you can modify without affecting the original project.
2. Clone the forked repository to your local machine using a Git client like Git or GitHub Desktop.
3. Create a new branch with a descriptive name (e.g., `new-feature-branch` or `bugfix-issue-123`).
```sh
git checkout -b new-feature-branch
```
4. Make changes to the project's codebase.
5. Commit your changes to your local branch with a clear commit message that explains the changes you've made.
```sh
git commit -m 'Implemented new feature.'
```
6. Push your changes to your forked repository on GitHub using the following command
```sh
git push origin new-feature-branch
```
7. Create a new pull request to the original project repository. In the pull request, describe the changes you've made and why they're necessary.
The project maintainers will review your changes and provide feedback or merge them into the main branch.

---

## 📄 License

This project is licensed under the `GNU GPL` License. 

---

## 👏 Acknowledgments


---
