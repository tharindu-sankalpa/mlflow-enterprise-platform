mkdir mlflow_iris_example
cd mlflow_iris_example

# List available python versions if unsure: pyenv versions
# Install if needed: pyenv install 3.10.13
pyenv local 3.10.13 # Sets the python version for this directory only
python --version # Verify, should show the version you set

poetry init

poetry add mlflow scikit-learn pandas matplotlib seaborn

poetry shell

python train.py

mlflow ui

mlflow models serve -m "runs:/<YOUR_RUN_ID>/iris-logistic-regression-model" -p 1234 --env-manager local


curl -X POST -H "Content-Type:application/json" --data '{
  "dataframe_split": {
    "columns": [
      "sepal length (cm)",
      "sepal width (cm)",
      "petal length (cm)",
      "petal width (cm)"
    ],
    "data": [
      [5.1, 3.5, 1.4, 0.2],
      [6.7, 3.1, 4.7, 1.5],
      [7.0, 3.2, 4.7, 1.4]
    ]
  }
}' http://127.0.0.1:1234/invocations



mlflow models build-docker -m "runs:/<YOUR_RUN_ID>/iris-logistic-regression-model" -n iris-classifier-service


docker run -p 1235:8080 iris-classifier-service



curl -X POST -H "Content-Type:application/json" --data '{
  "dataframe_split": {
    "columns": [
      "sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"
    ],
    "data": [
      [5.1, 3.5, 1.4, 0.2], [6.7, 3.1, 4.7, 1.5]
    ]
  }
}' http://127.0.0.1:1235/invocations



docker stop <container_id_or_name>


docker rm <container_id_or_name>