#Written by Clare Xinyu Xu
#z5175081

#task1.1
from pyspark.sql import DataFrame
from pyspark.sql.types import DoubleType
from pyspark.sql.functions import udf
from pyspark.ml.feature import Tokenizer, CountVectorizer, StringIndexer,OneHotEncoderEstimator
from pyspark.ml import Pipeline, Transformer

def base_features_gen_pipeline(input_descript_col="descript", input_category_col="category", output_feature_col="features", output_label_col="label"):
    wordTokenizer = Tokenizer(inputCol = input_descript_col,outputCol = "words")
    countVectors = CountVectorizer(inputCol = "words",outputCol = output_feature_col)
    label_maker = StringIndexer(inputCol = input_category_col,outputCol = output_label_col)
    pipeline = Pipeline(stages = [wordTokenizer,countVectors,label_maker])
    return pipeline


#task 1.2
def func(nb,svm):
    nb = int(nb)
    svm = int(svm)
    int_type = str(nb) + str(svm)
    int_type = int (int_type,2)
    int_type = int_type / 1.0 
    return int_type

def gen_meta_features(training_df, nb_0, nb_1, nb_2, svm_0, svm_1, svm_2):

    #evaluator = MulticlassClassificationEvaluator(predictionCol="prediction",metricName='f1')
    for i in range(5):
        condition = training_df['group'] == i
        c_train = training_df.filter(~condition).cache()
        c_test = training_df.filter(condition).cache()
        #nb prediction
        nb_model = nb_0.fit(c_train)
        nb_pred = nb_model.transform(c_test).drop('category','descript','words','nb_raw_0','nb_prob_0')
        
        nb_model_1 = nb_1.fit(c_train)
        nb_pred_1 = nb_model_1.transform(c_test).select('id','nb_pred_1')
        
        nb_model_2 = nb_2.fit(c_train)
        nb_pred_2 = nb_model_2.transform(c_test).select('id','nb_pred_2')
        
        #svm prediction
        svm_model = svm_0.fit(c_train)
        svm_pred = svm_model.transform(c_test).select('id','svm_pred_0')
       
        svm_model_1 = svm_1.fit(c_train)
        svm_pred_1 = svm_model_1.transform(c_test).select('id','svm_pred_1')
        
        svm_model_2 = svm_2.fit(c_train)
        svm_pred_2 = svm_model_2.transform(c_test).select('id','svm_pred_2')
        
        nb_pred = nb_pred.join(nb_pred_1, on=['id']).join(nb_pred_2, on=['id']).join(svm_pred, on=['id']).join(svm_pred_1, on=['id']).join(svm_pred_2, on=['id'])
        
        func_udf = udf(func,DoubleType())
        nb_pred = nb_pred.withColumn('joint_pred_0',func_udf(nb_pred['nb_pred_0'],nb_pred['svm_pred_0']))
        nb_pred = nb_pred.withColumn('joint_pred_1',func_udf(nb_pred['nb_pred_1'],nb_pred['svm_pred_1']))
        nb_pred = nb_pred.withColumn('joint_pred_2',func_udf(nb_pred['nb_pred_2'],nb_pred['svm_pred_2']))
        if i == 0:
            nb_pred_sum = nb_pred
        else:
            nb_pred_sum = nb_pred_sum.union(nb_pred)
            
    return nb_pred_sum
       

       
        
    

def test_prediction(test_df, base_features_pipeline_model, gen_base_pred_pipeline_model, gen_meta_feature_pipeline_model, meta_classifier):
    testing_set = base_features_pipeline_model.transform(test_df)
    test = gen_base_pred_pipeline_model.transform(testing_set)
    columns_to_drop = ['nb_raw_0', 'nb_prob_0','nb_raw_1', 'nb_prob_1','nb_raw_2', 'nb_prob_2','svm_raw_0', 'svm_prob_0','svm_raw_1','svm_prob_1','svm_raw_2', 'svm_prob_2']
    test = test.drop(*columns_to_drop)      
    func_udf = udf(func,DoubleType())
    test = test.withColumn('joint_pred_0',func_udf(test['nb_pred_0'],test['svm_pred_0']))
    test = test.withColumn('joint_pred_1',func_udf(test['nb_pred_1'],test['svm_pred_1']))
    test = test.withColumn('joint_pred_2',func_udf(test['nb_pred_2'],test['svm_pred_2']))
        
    #test = test.orderBy(test['id'].asc())
    test_meta_features = gen_meta_feature_pipeline_model.transform(test).select('id','label','meta_features')
    meta_pred = meta_classifier.transform(test_meta_features).select('id','label','final_prediction')
    return meta_pred