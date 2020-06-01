import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.SparkSession
import org.apache.spark.rdd.RDD.rddToPairRDDFunctions
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer, HashingTF, Tokenizer, CountVectorizer, CountVectorizerModel}
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier, LogisticRegression}
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.NGram
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator

case class PersonMessage(clas: Int, Message: String)


object email_spammer{
  def main(args: Array[String]) ={

    val spark = new SparkSession.Builder().appName("Apple_Analytics").config("spark.some.config.option", "some-value").getOrCreate();
    import spark.implicits._

    val sc = spark.sparkContext

    val text = sc.textFile("hdfs://localhost:9000/Module8/CaseStudy1/SMSSpamCollection")
    val df = text.map(x= > {val splits = x.split("\t");
      if (splits(0) == "ham"){PersonMessage(0, splits(1))}
      else {PersonMessage(1, splits(1))}}).toDF()

    val tokenizer = new Tokenizer().setInputCol("Message").setOutputCol("words")
    val wordsData = tokenizer.transform(df)

    val hashingTF = new HashingTF().setInputCol("words").setOutputCol("features").setNumFeatures(50)
    val featurizedData = hashingTF.transform(wordsData)

    val labelIndexer = new StringIndexer().setInputCol("clas").setOutputCol("label")
    val df2 = labelIndexer.fit(featurizedData).transform(featurizedData)

    val cvModel: CountVectorizerModel = new CountVectorizer().setInputCol("words").setOutputCol("count_vector").setVocabSize(50).fit(df2)
    val df3 = cvModel.transform(df2)

    df3.show

    val Array(trainingData, testData) = df3.randomSplit(Array(0.7, 0.3), 53)

    val lr = new LogisticRegression().setMaxIter(10).setRegParam(0.3).setElasticNetParam(0.8)
    val model = lr.fit(trainingData)

    val predictions = model.transform(testData)
    predictions.show

    val evaluator = new BinaryClassificationEvaluator().setLabelCol("label").setRawPredictionCol("rawPrediction").setMetricName("areaUnderROC")

    val accuracy = evaluator.evaluate(predictions)
    println("Test Accuracy using Logistic Regression = " + accuracy)
    // ----------------------------------------------------------------------------

    val LabelIndexer = new StringIndexer().setInputCol("clas").setOutputCol("indexedLabel").fit(featurizedData)
    val featureIndexer = new
        VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(2).fit(featurizedData)
    val Array(trainingData1, testData1) = featurizedData.randomSplit(Array(0.7, 0.3))

    val rf = new RandomForestClassifier().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures").setNumTrees(10)

    val labelConverter = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(LabelIndexer.labels)

    val pipeline = new Pipeline().setStages(Array(LabelIndexer, featureIndexer, rf, labelConverter))

    val modelR = pipeline.fit(trainingData1)
    val predictionsR = modelR.transform(testData1)

    val evaluatorR = new MulticlassClassificationEvaluator().setLabelCol("indexedLabel").setPredictionCol("prediction").setMetricName("accuracy")

    val accuracyR = evaluatorR.evaluate(predictionsR)
    println("Test Accuracy using Random Forest = " + accuracyR)
    // ----------------------------------------------------------------------------

    val bi_gram = new NGram().setN(2).setInputCol("words").setOutputCol("bigrams")
    val bi_gramDataFrame = bi_gram.transform(wordsData)

    val hashingTF2gram = new HashingTF().setInputCol("bigrams").setOutputCol("features").setNumFeatures(50)
    val featurizedData2gram = hashingTF2gram.transform(bi_gramDataFrame)
    val LabelIndexer2gram = new StringIndexer().setInputCol("clas").setOutputCol("indexedLabel").fit(featurizedData2gram)
    val featureIndexer2gram = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(2).fit(featurizedData2gram)
    val Array(trainingData2gram, testData2gram) = featurizedData2gram.randomSplit(Array(0.7, 0.3))

    val rf2gram = new RandomForestClassifier().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures").setNumTrees(10)

    val labelConverter2gram = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(LabelIndexer2gram.labels)

    val pipeline2gram = new Pipeline().setStages(Array(LabelIndexer, featureIndexer, rf, labelConverter))

    val modelR2gram = pipeline2gram.fit(trainingData2gram)
    val predictionsR2gram = modelR2gram.transform(testData2gram)

    val evaluatorR2gram = new MulticlassClassificationEvaluator().setLabelCol("indexedLabel").setPredictionCol("prediction").setMetricName("accuracy")

    val accuracyR2gram = evaluatorR2gram.evaluate(predictionsR2gram)
    println("Test Accuracy using Random Forest for Bi gram arrangement = " + accuracyR2gram)

    // ----------------------------------------------------------------------------


    val tri_gram = new NGram().setN(3).setInputCol("words").setOutputCol("trigrams")
    val tri_gramDataFrame = tri_gram.transform(wordsData)

    val hashingTF3gram = new HashingTF().setInputCol("trigrams").setOutputCol("features").setNumFeatures(50)
    val featurizedData3gram = hashingTF3gram.transform(tri_gramDataFrame)

    val LabelIndexer3gram = new StringIndexer().setInputCol("clas").setOutputCol("indexedLabel").fit(featurizedData3gram)
    val featureIndexer3gram = new VectorIndexer().setInputCol("features").setOutputCol("indexedFeatures").setMaxCategories(3).fit(featurizedData3gram)
    val Array(trainingData3gram, testData3gram) = featurizedData3gram.randomSplit(Array(0.7, 0.3))

    val rf3gram = new RandomForestClassifier().setLabelCol("indexedLabel").setFeaturesCol("indexedFeatures").setNumTrees(10)

    val labelConverter3gram = new IndexToString().setInputCol("prediction").setOutputCol("predictedLabel").setLabels(LabelIndexer3gram.labels)

    val pipeline3gram = new Pipeline().setStages(Array(LabelIndexer, featureIndexer, rf, labelConverter))

    val modelR3gram = pipeline3gram.fit(trainingData3gram)
    val predictionsR3gram = modelR3gram.transform(testData3gram)

    val evaluatorR3gram = new MulticlassClassificationEvaluator().setLabelCol("indexedLabel").setPredictionCol("prediction").setMetricName("accuracy")

    val accuracyR3gram = evaluatorR3gram.evaluate(predictionsR3gram)
    println("Test Accuracy using Random Forest for Tri gram arrangement = " + accuracyR3gram)

  }

}
