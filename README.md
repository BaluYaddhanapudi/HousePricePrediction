# HousePricePrediction

We have developed the House Price Predictor python code over Docker Containerization. 

The prediction of house prices consists of 4 key programs / steps.
 ## Running Docker
 Docker image that will bring up Kafka and Zookeeper. This needs to be executed first to setup and bring up a Kafka topic named 'my_stream
 * Please execute the step: 
 > docker compose up
 * Docker-compose.yml creates the services of Zookeeper (Ports: 32181) and Kafka (9092) on localhost terminal.
 
 ## Streaming Data into Kafka topic
 * Giving topic name as _my-stream_ Execute below and it will stream data inside test.csv to kafka: (in a different terminal)
 > python sendStream.py ../data/test.csv my-stream
 
## Consuming the events from kafka and predicting
* From the topic named _my-stream_ events will be consumed and predict the house price using the program 'model.py' and show the output on the terminal
> python processStream.py my-stream

 
 ## Modeling HousePricePrediction
 * Please execute (inside bin):
 > python model.py
 * This program takes the _data/train.csv_ as the input into a DataFrame. Performs pre-processing of numerical and categorical columns.
 * Using _One-Hot Encoder_ for categorical columns and then models the data using _Ridge Regression_ model.
 * We have tried different regression models and observed that _Ridge Regression_ model was giving effective results, with Training data R2 Score as 91%
 * Given Test data _test.csv_ for prediction:
 Prediction below (4 rows):
 
|       Id   |   SalePrice|
|----------|----------|
|     1461  |106830.452898|
|     1462  |164410.404970|
|     1463  |185631.037439|
|     1464  |191504.118363|

