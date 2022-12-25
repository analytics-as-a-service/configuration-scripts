# Spark Session, Pipeline, Functions, and Metrics
from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext
from pyspark.ml.feature import OneHotEncoder, StringIndexer, StandardScaler, VectorAssembler, Imputer
from pyspark.ml import Pipeline
from pyspark.sql.functions import rand
from pyspark.mllib.evaluation import MulticlassMetrics

# Keras / Deep Learning
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras import optimizers, regularizers
from tensorflow.keras.optimizers import Adam

# Elephas for Deep Learning on Spark
from elephas.ml_model import ElephasEstimator

from pyspark.sql import SparkSession

spark = SparkSession(sc)

from pyspark.sql.types import StringType,IntegerType,StructField

for i in environ.keys():
    temp = i.find("HDFS_NN_PORT")
    if temp>=0 and len(i)-temp==12 and environ[i].find("9000")>=0:
        hadoop=environ[i].replace("tcp","hdfs")
        break

train_transaction = spark.read.format('com.databricks.spark.csv').options(header="true",inferschema='true').load(f'{hadoop}/dataset/train_transaction.csv')
train_identity =  spark.read.format('com.databricks.spark.csv').options(header="true",inferschema='true').load(f'{hadoop}/dataset/train_identity.csv')

train_d = train_transaction.join(train_identity,"TransactionID")

import pandas as pd

cols = pd.read_csv('./imp_features.csv',index_col='Unnamed: 0')
cols = cols.values[:,0]

train = train_d.select(*cols)

cat_var = [ i["name"] for i in train.schema.jsonValue()["fields"] if i["type"]=="string" ]
num_var = [ i["name"] for i in train.schema.jsonValue()["fields"] if i["type"]=="integer" or i["type"]=="double" ]

def select_features_to_scale(df=train, feature_list=[], lower_skew=-2, upper_skew=2):
    selected_features = []
    temp = df.select(*feature_list).toPandas()
    for feature in feature_list:
        if temp[feature].kurtosis() < -2 or temp[feature].kurtosis() > 2:
            selected_features.append(feature)
    return selected_features

if "isFraud" in cat_var:
    cat_var.remove("isFraud")
if "isFraud" in num_var:
    num_var.remove("isFraud")

stages = []

string_indexer = StringIndexer(inputCols=cat_var, outputCols=[i + "_index" for i in cat_var]).setHandleInvalid("keep")
imputer = Imputer(inputCols=string_indexer.getOutputCols(),outputCols=[i + "_imputed" for i in cat_var]).setStrategy("mode")
encoder = OneHotEncoder(inputCols=imputer.getOutputCols(),outputCols=[i + "_class_vec" for i in cat_var])
stages += [string_indexer,imputer, encoder]

imputer = Imputer(inputCols=num_var,outputCols=[i+"_imputed" for i in num_var]).setStrategy("mean")
stages += [imputer]

unscaled_features = select_features_to_scale(df=train, feature_list=num_var)
unscaled_assembler = VectorAssembler(inputCols=[i+"_imputed" for i in unscaled_features], outputCol="unscaled_features")
scaler = StandardScaler(inputCol="unscaled_features", outputCol="scaled_features")
stages += [unscaled_assembler, scaler]

label_str_indexer =  StringIndexer(inputCol="isFraud", outputCol="Y")
num_unscaled_diff_list = list(set(num_var) - set(unscaled_features))
assembler_inputs = [i + "_class_vec" for i in cat_var] + [i+"_imputed" for i in num_unscaled_diff_list] + ["scaled_features"]
assembler = VectorAssembler(inputCols=assembler_inputs, outputCol="X")
stages += [label_str_indexer, assembler]

pipeline = Pipeline(stages=stages)
pipeline_model = pipeline.fit(train)

df_train = pipeline_model.transform(train)

_train = df_train.select("X","Y").toDF("features","label")

!pip3 install sklearn

import random
import numpy as np
from pyspark.sql import Row
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import col, explode, array, lit

vectorized=_train
k = 2
minorityClass = 1
majorityClass = 0
percentageOver = 200
percentageUnder = 100
if(percentageUnder > 100|percentageUnder < 10):
    raise ValueError("Percentage Under must be in range 10 - 100");
if(percentageOver < 100):
    raise ValueError("Percentage Over must be in at least 100");min_Array[i][0]
dataInput_min = vectorized[vectorized['label'] == minorityClass]
dataInput_maj = vectorized[vectorized['label'] == majorityClass]

ratio = dataInput_maj.count()/dataInput_min.count()

a = range(round(ratio))
oversampled_minority_df = dataInput_min.withColumn("dummy", explode(array([lit(x) for x in a]))).drop('dummy')
oversampled_df = dataInput_maj.unionAll(oversampled_minority_df)

# sampled_majority_df = dataInput_maj.sample(False, 1/ratio)
# undersampled_df = dataInput_min.unionAll(sampled_majority_df)

from pyspark.ml.classification import LogisticRegression, RandomForestClassifier

lr = LogisticRegression(maxIter=10)
lrt = lr.fit(oversampled_df)

pipeline_model.save(f"{hadoop}/pipeline")

lrt.save(f"{hadoop}/model")

rf = RandomForestClassifier()
rft = rf.fit(oversampled_df)

# from sklearn.metrics import classification_report

# predvsact = lrt.transform(_train).select("label","prediction").toPandas()
# print(classification_report(predvsact["label"],predvsact["prediction"]))

# predvsact = rft.transform(_train).select("label","prediction").toPandas()
# print(classification_report(predvsact["label"],predvsact["prediction"]))

# nb_classes = _train.select("label").distinct().count()
# input_dim = len(_train.select("features").first()[0])

# model = Sequential()
# model.add(Dense(10, input_shape=(input_dim,), activity_regularizer=regularizers.l2(0.01)))
# model.add(Activation('relu'))
# model.add(Dropout(rate=0.3))
# model.add(Dense(nb_classes))
# model.add(Activation('sigmoid'))
# model.compile(loss='binary_crossentropy', optimizer='adam')

# optimizer_conf = optimizers.Adam(learning_rate=0.01)
# opt_conf = optimizers.serialize(optimizer_conf)

# estimator = ElephasEstimator()
# estimator.setFeaturesCol("features")
# estimator.setLabelCol("label")
# estimator.set_keras_model_config(model.to_json())
# estimator.set_categorical_labels(True)
# estimator.set_nb_classes(nb_classes)
# estimator.set_num_workers(1)
# estimator.set_epochs(2) 
# estimator.set_batch_size(64)
# estimator.set_verbosity(1)
# estimator.set_validation_split(0.10)
# estimator.set_optimizer_config(opt_conf)
# estimator.set_mode("synchronous")
# estimator.set_loss("binary_crossentropy")
# estimator.set_metrics(['acc'])

# dl_pipeline = Pipeline(stages=[estimator])

# dlt = dl_pipeline.fit(_train)



# import random
# import numpy as np
# from functools import reduce
# import pyspark.sql.functions as F
# from pyspark.sql import Row
# from pyspark.sql.functions import rand,col,when,concat,substring,lit,udf,lower,sum as ps_sum,count as ps_count,row_number
# from pyspark.sql.window import *
# from pyspark.sql import DataFrame
# from pyspark.ml.feature import VectorAssembler,BucketedRandomProjectionLSH,VectorSlicer
# from pyspark.sql.window import Window
# from pyspark.ml.linalg import Vectors,VectorUDT,SparseVector
# from pyspark.sql.functions import array, create_map, struct

# def subudf(arr):
#     # Must decorate func as udf to ensure that its callback form is the arg to df iterator construct
#     a = arr[0]
#     b = arr[1]
#     if isinstance(a, SparseVector):
#         a = a.toArray()
#     if isinstance(b, SparseVector):
#         b = b.toArray()
#     array_ = a - b
#     return random.uniform(0, 1) * Vectors.dense(array_)
# subtract_vector_udf = udf(subudf)

# def addudf(arr):
#     # Must decorate func as udf to ensure that its callback form is the arg to df iterator construct
#     a = arr[0]
#     b = arr[1]
#     if isinstance(a, SparseVector):
#         a = a.toArray()
#     if isinstance(b, SparseVector):
#         b = b.toArray()
#     array_ = a + b
#     return Vectors.dense(array_)
# add_vector_udf = udf(subudf)

# def smote(vectorized_sdf, seed, k, bucketLength, multiplier):
#     dataInput_min = vectorized_sdf[vectorized_sdf['label'] == 1]
#     dataInput_maj = vectorized_sdf[vectorized_sdf['label'] == 0]
    
#     # LSH, bucketed random projection
#     brp = BucketedRandomProjectionLSH(inputCol="features", outputCol="hashes",seed=seed, bucketLength=bucketLength)
#     # smote only applies on existing minority instances    
#     model = brp.fit(dataInput_min)
#     model.transform(dataInput_min)

#     # here distance is calculated from brp's param inputCol
#     self_join_w_distance = model.approxSimilarityJoin(dataInput_min, dataInput_min, float("inf"), distCol="EuclideanDistance")

#     # remove self-comparison (distance 0)
#     self_join_w_distance = self_join_w_distance.filter(self_join_w_distance.EuclideanDistance > 0)

#     over_original_rows = Window.partitionBy("datasetA").orderBy("EuclideanDistance")

#     self_similarity_df = self_join_w_distance.withColumn("r_num", row_number().over(over_original_rows))

#     self_similarity_df_selected = self_similarity_df.filter(self_similarity_df.r_num <= k)

#     over_original_rows_no_order = Window.partitionBy('datasetA')

#     # list to store batches of synthetic data
#     res = []
    
#     # two udf for vector add and subtract, subtraction include a random factor [0,1]
#     subtract_vector_udf = udf(lambda arr: random.uniform(0, 1)*(subtract_vector_udf(arr)), VectorUDT())
#     add_vector_udf = udf(lambda arr: add_vector_udf(arr), VectorUDT())
    
#     # retain original columns
#     original_cols = dataInput_min.columns
    
#     for i in range(multiplier):
#         print("generating batch %s of synthetic instances"%i)
#         # logic to randomly select neighbour: pick the largest random number generated row as the neighbour
#         df_random_sel = self_similarity_df_selected.withColumn("rand", F.rand()).withColumn('max_rand', F.max('rand').over(over_original_rows_no_order))\
#                             .where(F.col('rand') == F.col('max_rand')).drop(*['max_rand','rand','r_num'])
#         # create synthetic feature numerical part
#         df_vec_diff = df_random_sel.select('*', subtract_vector_udf(array('datasetA.features', 'datasetB.features')).alias('vec_diff'))
#         df_vec_modified = df_vec_diff.select('*', add_vector_udf(array('datasetA.features', 'vec_diff')).alias('features'))
        
#         # for categorical cols, either pick original or the neighbour's cat values
#         for c in original_cols:
#             # randomly select neighbour or original data
#             col_sub = random.choice(['datasetA','datasetB'])
#             val = "{0}.{1}".format(col_sub,c)
#             if c != 'features':
#                 # do not unpack original numerical features
#                 df_vec_modified = df_vec_modified.withColumn(c,col(val))
        
#         # this df_vec_modified is the synthetic minority instances,
#         df_vec_modified = df_vec_modified.drop(*['datasetA','datasetB','vec_diff','EuclideanDistance'])
        
#         res.append(df_vec_modified)
    
#     dfunion = reduce(DataFrame.unionAll, res)
#     # union synthetic instances with original full (both minority and majority) df
#     oversampled_df = dfunion.union(vectorized_sdf.select(dfunion.columns))
    
#     return oversampled_df

# oversampled_df = smote(_train, 32, 2, 10, 12)

# oversampled_df.count()
