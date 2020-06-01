import org.apache.hadoop.fs.{FileSystem, Path}
import sys.process._
import org.apache.spark.ml.feature.Tokenizer
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.StopWordsRemover
import org.apache.spark.ml.feature.{CountVectorizer, CountVectorizerModel}
import org.apache.spark.ml.clustering.LDA
import org.apache.spark.ml.clustering.KMeans

case class PersonMessage(Person: String, hdfs_loc: String, Sent_Email: Int)
case class MessageSent(Message_ID: String, Date: String, From: String, To: String, Subject: String)




object mail_analytics{
  def main(args: Array[String]) = {
    val spark = new SparkSession.Builder().appName("Email_Analytics").config("spark.some.config.option", "some-value").getOrCreate();
    import spark.implicits._

    val sc = spark.sparkContext

    val pattern = """^\s*(\d*)\s*(\d*)\s*(\d*)\s(hdfs://localhost:9000/Module8/CaseStudy2/maildir/)(\S*)\/""".r
    val pattern1 = """^Message-ID:\s(\S*)\,\sDate:\s(\C*)\,\sFrom:\s(\S*)\,\sTo:\s(\S*)\C*Subject:\s(\S*)\,""".r

    val lst = FileSystem.get(sc.hadoopConfiguration).listStatus(new Path("hdfs://localhost:9000/Module8/CaseStudy2/maildir/"))

    var EmailSentDf = Seq.empty[PersonMessage].toDF
    var MessageSentDf = Seq.empty[MessageSent].toDF

    lst.foreach(x= > {val check = FileSystem.get(sc.hadoopConfiguration).exists(new Path(x.getPath + "/_sent_mail")); if (check == true){
      val strg =("hdfs dfs -count ")+x.getPath+"/_sent_mail/"!!; val patrn=pattern.findAllIn(strg); val tempDf= Seq(PersonMessage(patrn.group(5), patrn.group(4)+patrn.group(5), patrn.group(2).toInt)).toDF; EmailSentDf=EmailSentDf.union(tempDf); val emaillst=FileSystem.get( sc.hadoopConfiguration ).listStatus( new Path(patrn.group(4)+patrn.group(5)));
      emaillst.foreach(y= > {val text=sc.textFile(y.toString());val txt=text.filter(line = > line.contains("Message-ID") | | line.contains("Date") | | line.contains("From") | | line.contains("To") | | line.contains("Subject"))
        val emailextract=pattern1.findAllIn(txt.toString); val msgsenttemp= Seq(MessageSent(emailextract.group(1), emailextract.group(2), emailextract.group(3), emailextract.group(4), emailextract.group(5))).toDF; MessageSentDf=MessageSentDf.union(msgsenttemp)})} else None})

    val df=MessageSentDf

    val df1 = df.withColumn("week", weekofyear(unix_timestamp($"Date", "EEE, dd MMM yyyy HH”).cast("timestamp")))
      val maxweek = df1.agg(max("week")).first()(0).asInstanceOf[Int]
      df1.groupBy("from").count().withColumn("avgcount", $"count" / maxweek).sort($"avgcount".desc).show



    val tokenizer = new Tokenizer().setInputCol("subject").setOutputCol("words")
    val transformed = tokenizer.transform(df1)

    val topusers = df1.groupBy("from").count().sort($"count".desc).take(1).map(_.getString(0))
    transformed.filter($"subject" != = "").filter($"from".isin(topusers:_ *)).withColumn("keyword",explode($"words")).groupBy("keyword").count().sort($"count".desc).show
    transformed.filter($"subject" != = "").filter(!$"from".isin(topusers: _ *)).withColumn("keyword",explode($"words")).groupBy("keyword").count().sort($"count".desc).show

    val remover = new StopWordsRemover().setInputCol("words").setOutputCol("filtered")
    val cleaned = remover.transform(transformed)
    cleaned.filter($"subject" != = "").withColumn("keyword", explode($"words")).groupBy("keyword").count().sort($"count".desc).show

    val stopwords = new StopWordsRemover().getStopWords + + Array("-", "re:", "fw:")
    val remover = new StopWordsRemover().setStopWords(stopwords).setInputCol("words").setOutputCol("filtered")
    val cleaned = remover.transform(transformed)
    val keywords = cleaned.filter($"subject" != = "").withColumn("keyword", explode($"filtered"))keywords.groupBy("keyword").count().sort($"count".desc).show

    val df2 = cleaned.withColumn("msgtype", when($"subject".startsWith("Re:"), 1).otherwise(when($"subject".startsWith("Fw:"), 2).otherwise(0)))

    df2.groupBy("week").pivot("msgtype").count().show()

    val df4 = df2.filter($"subject" != = "")
    val cvmodel = new CountVectorizer().setInputCol("filtered").setOutputCol("features").fit(df4)
    val featured = cvmodel.transform(df4)

    val kmeans = new KMeans().setK(4).setSeed(1L)
    val model = kmeans.fit(featured)
    val predictions = model.transform(featured)

    val lda = new LDA().setK(4).setMaxIter(10)
    val model = lda.fit(featured)
    val topics = model.describeTopics(4)

  }
}
