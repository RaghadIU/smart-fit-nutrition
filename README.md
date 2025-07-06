# SmartFit Nutrition Analyzer
## Project Description
SmartFit Nutrition Analyzer is an interactive web app that analyzes food images using a deep learning model. It provides accurate predictions of the food type from a selected set of classes, along with nutritional information like estimated calories and approximate exercise durations needed to burn those calories.

----------

## Requirements
- Python 3.10 or higher
- Libraries: tensorflow, numpy, pillow, streamlit, opencv-python, tensorflow-datasets
------
## Installation

```bash
git clone <https://github.com/RaghadIU/smart-fit-nutrition.git> 
```
```bash
cd smartfit-nutrition-app
```
```bash
python -m venv venv
```
```bash
.\venv\Scripts\activate    # On Windows PowerShell
```
```bash
pip install -r requirements.txt
```
## how to Run 
```bash 
streamlit run app.py
```
------
## features 
- Analyze uploaded food images and predict the food class from a selected list

- Display estimated calories for the detected food

- Show approximate time required to burn calories through various activities (walking, running, cycling, yoga)

- User-friendly and simple interface
-----------

## future improvements 

- Add healthy alternative suggestions for detected foods

- Improve model accuracy by refining selected food classes

- Enhance UI with detailed nutrition statistics and user profile features

