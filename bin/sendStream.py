#!/usr/bin/env python

"""Generates a stream to Kafka from a time series csv file.
"""

import argparse
import csv
import json
import sys
import time
from dateutil.parser import parse
from confluent_kafka import Producer
import socket


def acked(err, msg):
    if err is not None:
        print("Failed to deliver message: %s: %s" % (str(msg.value()), str(err)))
    else:
        print("Message produced: %s" % (str(msg.value())))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('filename', type=str,
                        help='Time series csv file.')
    parser.add_argument('topic', type=str,
                        help='Name of the Kafka topic to stream.')
    parser.add_argument('--speed', type=float, default=1, required=False,
                        help='Speed up time series by a given multiplicative factor.')
    args = parser.parse_args()

    topic = args.topic
    p_key = args.filename

    conf = {'bootstrap.servers': "localhost:9092",
            'client.id': socket.gethostname()}
    producer = Producer(conf)

    rdr = csv.reader(open(args.filename))

    data_list = list(rdr)

    #next(rdr)  # Skip header
    firstline = True
    print("This is in")
    print(len(data_list[0]))
    #row = next(rdr, None)
    #print(row[0])

    print(data_list[0])
    key_list = data_list[0]

    print("This is keylist:")
    print(key_list)

    result = {}

    while True:

        try:

            if firstline is True:
                line1 = next(rdr, None)

                #timestamp, value = line1[0], float(line1[1])

                # for rows in rdr:
                #     key = rows['Id']
                #     result[key] = rows
                # print(rdr)

                print("This inside loop: ")

                # for i in range(len(key_list)):
                #     print(key_list[i])

                # print("This is value:")
                # for i in range(len(data_list[1])):
                #     print(data_list[1][i])
                

                len_data = len(data_list)

                for j in range(len(data_list)):
                    for i in range(len(key_list)):
                        print("This is j", j, "This is i", i)
                        result[key_list[i]] = data_list[j][i]

                    print("After converting:")
                    print(result)
                    # Convert csv columns to key value pair
                    #result = {}
                    #result[timestamp] = value
                    # Convert dict to json as message format
                    jresult = json.dumps(result)
                    producer.produce(topic, key=p_key, value=jresult, callback=acked)
                firstline = False

            # else:
            #     line = next(rdr, None)
            #     d1 = parse(timestamp)
            #     d2 = parse(line[0])
            #     diff = ((d2 - d1).total_seconds())/args.speed
            #     time.sleep(diff)
            #     timestamp, value = line[0], float(line[1])
            #     result = {}
            #     result[timestamp] = value
            #     jresult = json.dumps(result)

            #     producer.produce(topic, key=p_key, value=jresult, callback=acked)

            producer.flush()

        except TypeError:
            sys.exit()


if __name__ == "__main__":
    main()
