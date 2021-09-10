# HousePricePrediction

We have developed the House Price Predictor python code over Docker Containerization. 

The prediction of hosue prices consists of 4 key programs / steps.
  1. Docker image that will bring up Kafka and Zookeeper. This needs to be executed first to setup and bring up a Kafka topic named 'my_stream'
  2. Sendstream.py - that will stream the data from any input 'csv' file as a stream of events to the Kafka topic 'my_stream'.
  3. processstream.py - that will consume the events from kafka , predict the house price using the program 'model.py' and show the output on the terminal.
  4. model.py - reads the input file as csv, preprocesses the data and predicts using ridge_regression model.
