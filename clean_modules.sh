# To install the yaware_eurosat conda environment, run:
conda env create -f myenv.yml

# To clean the modules references (create from setup.py), run:
conda activate yaware_eurosat
# delete any cache files
find . -type d -name "__pycache__" -exec rm -rf {} +
find . -type f -name "*.pyc" -delete
pip cache purge

# Delete  the project from  setup (old)
python -m pip uninstall eurosat_project 

python - <<<'import yaware_eurosat; print(yaware_eurosat.__file__)'
rm -rf #path printed above

cd exploratory_notebooks
pip uninstall eurosat_project 

# Reinstall the project in editable mode
python -m pip install -e . 
