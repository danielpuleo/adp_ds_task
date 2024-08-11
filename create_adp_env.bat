@echo off
REM Name of the virtual environment
set ENV_NAME=adp_env

REM Create a virtual environment
python -m venv %ENV_NAME%

REM Activate the virtual environment
call %ENV_NAME%\Scripts\activate

REM Upgrade pip and setuptools
python -m pip install --upgrade pip setuptools

REM Install required libraries
pip install pandas numpy scikit-learn xgboost matplotlib datasets nltk torch transformers seaborn

REM Download NLTK data
python -m nltk.downloader wordnet stopwords

REM Deactivate the virtual environment
deactivate

echo Virtual environment '%ENV_NAME%' created and packages installed.
pause