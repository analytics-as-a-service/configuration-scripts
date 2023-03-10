from os import environ
from pyspark.sql import SparkSession
from subprocess import getoutput
from pyspark import SparkConf
from pyspark import SparkContext

for i in environ.keys():
    if len(i)-i.find("MASTER_SVC_PORT")==15 and environ[i].find("7077")>=0:
        master=environ[i].replace("tcp","spark")
        break

conf = SparkConf()
conf.setMaster(master)
conf.set('spark.driver.host',getoutput('hostname -I').strip())
conf.set('spark.jars.ivy',f"/tmp/modules/.ivy2")

try:
    if sc:
        sc.stop()
    if spark:
        spark.stop()
except:
    pass

sc=SparkContext(conf=conf)
spark = SparkSession(sc)

from sys import exit
temp=environ.keys()
if "MYSQL_USERNAME" not in temp or "MYSQL_PASSWORD" not in temp or "MYSQL_DATABASE" not in temp or "MYSQL_TABLE" not in temp or "KAFKA_TOPIC" not in temp or "MODEL_LOCATION" not in temp or "PIPELINE_LOCATION" not in temp:
    exit("Error: Required environment variables MYSQL_USERNAME, MYSQL_PASSWORD, MYSQL_DATABASE, MYSQL_TABLE, KAFKA_TOPIC, MODEL_LOCATION, and PIPELINE_LOCATION . Missing any one of these caused this error")

import pandas as pd
from pyspark.sql.functions import from_json,col
import json
kafkaservers = ""
for i in environ.keys():
    temp = i.find("INTERNAL_PORT")
    if temp>=0 and len(i)-temp==13 and environ[i].find("9092")>=0:
        if kafkaservers:
            kafkaservers+=","+environ[i].replace("tcp://","")
        else:
            kafkaservers+=environ[i].replace("tcp://","")

livedata = spark \
  .readStream \
  .format("kafka") \
  .option("kafka.bootstrap.servers", kafkaservers) \
  .option("subscribe", environ["KAFKA_TOPIC"]) \
  .load()

jsonschema = {'type': 'struct',
'fields': [{'name': 'TransactionID',
  'type': 'integer',
  'nullable': True,
  'metadata': {}},
  {'name': 'TransactionDT',
  'type': 'integer',
  'nullable': True,
  'metadata': {}},
  {'name': 'TransactionAmt',
  'type': 'double',
  'nullable': True,
  'metadata': {}},
  {'name': 'ProductCD', 'type': 'string', 'nullable': True, 'metadata': {}},
  {'name': 'card1', 'type': 'integer', 'nullable': True, 'metadata': {}},
  {'name': 'card2', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'card3', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'card4', 'type': 'string', 'nullable': True, 'metadata': {}},
  {'name': 'card5', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'card6', 'type': 'string', 'nullable': True, 'metadata': {}},
  {'name': 'addr1', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'addr2', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'dist1', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'dist2', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'P_emaildomain',
  'type': 'string',
  'nullable': True,
  'metadata': {}},
  {'name': 'R_emaildomain',
  'type': 'string',
  'nullable': True,
  'metadata': {}},
  {'name': 'C1', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'C2', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'C3', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'C4', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'C5', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'C6', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'C7', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'C8', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'C9', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'C10', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'C11', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'C12', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'C13', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'C14', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'D1', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'D2', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'D3', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'D4', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'D5', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'D6', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'D7', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'D8', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'D9', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'D10', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'D11', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'D12', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'D13', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'D14', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'D15', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'M1', 'type': 'string', 'nullable': True, 'metadata': {}},
  {'name': 'M2', 'type': 'string', 'nullable': True, 'metadata': {}},
  {'name': 'M3', 'type': 'string', 'nullable': True, 'metadata': {}},
  {'name': 'M4', 'type': 'string', 'nullable': True, 'metadata': {}},
  {'name': 'M5', 'type': 'string', 'nullable': True, 'metadata': {}},
  {'name': 'M6', 'type': 'string', 'nullable': True, 'metadata': {}},
  {'name': 'M7', 'type': 'string', 'nullable': True, 'metadata': {}},
  {'name': 'M8', 'type': 'string', 'nullable': True, 'metadata': {}},
  {'name': 'M9', 'type': 'string', 'nullable': True, 'metadata': {}},
  {'name': 'V1', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V2', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V3', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V4', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V5', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V6', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V7', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V8', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V9', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V10', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V11', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V12', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V13', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V14', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V15', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V16', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V17', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V18', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V19', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V20', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V21', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V22', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V23', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V24', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V25', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V26', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V27', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V28', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V29', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V30', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V31', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V32', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V33', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V34', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V35', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V36', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V37', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V38', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V39', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V40', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V41', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V42', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V43', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V44', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V45', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V46', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V47', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V48', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V49', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V50', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V51', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V52', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V53', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V54', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V55', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V56', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V57', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V58', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V59', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V60', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V61', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V62', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V63', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V64', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V65', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V66', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V67', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V68', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V69', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V70', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V71', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V72', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V73', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V74', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V75', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V76', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V77', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V78', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V79', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V80', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V81', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V82', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V83', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V84', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V85', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V86', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V87', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V88', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V89', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V90', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V91', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V92', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V93', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V94', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V95', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V96', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V97', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V98', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V99', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V100', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V101', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V102', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V103', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V104', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V105', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V106', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V107', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V108', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V109', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V110', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V111', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V112', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V113', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V114', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V115', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V116', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V117', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V118', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V119', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V120', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V121', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V122', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V123', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V124', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V125', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V126', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V127', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V128', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V129', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V130', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V131', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V132', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V133', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V134', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V135', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V136', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V137', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V138', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V139', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V140', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V141', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V142', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V143', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V144', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V145', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V146', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V147', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V148', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V149', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V150', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V151', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V152', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V153', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V154', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V155', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V156', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V157', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V158', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V159', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V160', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V161', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V162', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V163', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V164', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V165', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V166', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V167', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V168', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V169', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V170', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V171', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V172', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V173', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V174', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V175', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V176', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V177', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V178', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V179', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V180', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V181', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V182', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V183', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V184', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V185', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V186', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V187', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V188', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V189', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V190', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V191', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V192', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V193', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V194', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V195', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V196', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V197', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V198', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V199', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V200', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V201', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V202', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V203', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V204', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V205', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V206', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V207', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V208', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V209', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V210', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V211', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V212', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V213', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V214', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V215', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V216', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V217', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V218', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V219', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V220', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V221', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V222', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V223', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V224', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V225', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V226', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V227', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V228', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V229', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V230', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V231', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V232', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V233', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V234', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V235', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V236', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V237', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V238', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V239', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V240', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V241', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V242', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V243', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V244', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V245', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V246', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V247', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V248', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V249', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V250', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V251', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V252', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V253', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V254', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V255', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V256', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V257', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V258', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V259', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V260', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V261', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V262', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V263', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V264', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V265', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V266', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V267', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V268', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V269', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V270', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V271', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V272', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V273', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V274', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V275', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V276', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V277', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V278', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V279', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V280', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V281', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V282', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V283', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V284', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V285', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V286', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V287', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V288', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V289', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V290', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V291', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V292', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V293', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V294', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V295', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V296', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V297', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V298', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V299', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V300', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V301', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V302', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V303', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V304', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V305', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V306', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V307', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V308', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V309', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V310', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V311', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V312', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V313', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V314', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V315', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V316', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V317', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V318', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V319', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V320', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V321', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V322', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V323', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V324', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V325', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V326', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V327', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V328', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V329', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V330', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V331', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V332', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V333', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V334', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V335', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V336', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V337', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V338', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'V339', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'id-01', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'id-02', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'id-03', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'id-04', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'id-05', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'id-06', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'id-07', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'id-08', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'id-09', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'id-10', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'id-11', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'id-12', 'type': 'string', 'nullable': True, 'metadata': {}},
  {'name': 'id-13', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'id-14', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'id-15', 'type': 'string', 'nullable': True, 'metadata': {}},
  {'name': 'id-16', 'type': 'string', 'nullable': True, 'metadata': {}},
  {'name': 'id-17', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'id-18', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'id-19', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'id-20', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'id-21', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'id-22', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'id-23', 'type': 'string', 'nullable': True, 'metadata': {}},
  {'name': 'id-24', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'id-25', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'id-26', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'id-27', 'type': 'string', 'nullable': True, 'metadata': {}},
  {'name': 'id-28', 'type': 'string', 'nullable': True, 'metadata': {}},
  {'name': 'id-29', 'type': 'string', 'nullable': True, 'metadata': {}},
  {'name': 'id-30', 'type': 'string', 'nullable': True, 'metadata': {}},
  {'name': 'id-31', 'type': 'string', 'nullable': True, 'metadata': {}},
  {'name': 'id-32', 'type': 'double', 'nullable': True, 'metadata': {}},
  {'name': 'id-33', 'type': 'string', 'nullable': True, 'metadata': {}},
  {'name': 'id-34', 'type': 'string', 'nullable': True, 'metadata': {}},
  {'name': 'id-35', 'type': 'string', 'nullable': True, 'metadata': {}},
  {'name': 'id-36', 'type': 'string', 'nullable': True, 'metadata': {}},
  {'name': 'id-37', 'type': 'string', 'nullable': True, 'metadata': {}},
  {'name': 'id-38', 'type': 'string', 'nullable': True, 'metadata': {}},
  {'name': 'DeviceType', 'type': 'string', 'nullable': True, 'metadata': {}},
  {'name': 'DeviceInfo', 'type': 'string', 'nullable': True, 'metadata': {}},
  {'name': 'TransactionTime',
  'type': 'string',
  'nullable': True,
  'metadata': {}},
  {'name': 'Seconds', 'type': 'string', 'nullable': True, 'metadata': {}}]}
jsonschema1 = json.dumps(jsonschema).replace("integer","string").replace("double","string").replace("-","_")

df_newdata = livedata.select(from_json(col("value").cast("string"),jsonschema1).alias("value1")).select("value1.*")

newdata = df_newdata.select(*[col(i["name"].replace("-","_")).cast(i["type"]).alias(i["name"].replace("-","_")) for i in jsonschema["fields"]])

for i in environ.keys():
    temp = i.find("HDFS_NN_PORT")
    if temp>=0 and len(i)-temp==12 and environ[i].find("9000")>=0:
        hadoop=environ[i].replace("tcp","hdfs")
        break
for i in environ.keys():
    temp = i.find("MYSQL_PORT")
    if temp>=0 and len(i)-temp==10 and environ[i].find("3306")>=0:
        mysql=environ[i].replace("tcp","mysql")
        break

from pyspark.ml import PipelineModel
pipeline_model = PipelineModel.load(f"{hadoop}{environ['PIPELINE_LOCATION']}")
df_newdata = pipeline_model.transform(newdata)
_newdata = df_newdata.select(*newdata.columns,col("X").alias("features"))

from pyspark.ml.classification import LogisticRegressionModel
lrt = LogisticRegressionModel.load(f"{hadoop}{environ['MODEL_LOCATION']}")

def foreach_batch_function(df, epoch_id):
    df.write.mode("append").format("jdbc") \
.option("driver","com.mysql.cj.jdbc.Driver") \
.option("url", f"jdbc:{mysql}/{environ['MYSQL_DATABASE']}") \
.option("dbtable", f"{environ['MYSQL_TABLE']}") \
.option("user", f"{environ['MYSQL_USERNAME']}") \
.option("password", f"{environ['MYSQL_PASSWORD']}").save()

writestream = lrt.transform(_newdata).select(*newdata.columns,"prediction").writeStream \
.foreachBatch(foreach_batch_function).start()

writestream.awaitTermination()
