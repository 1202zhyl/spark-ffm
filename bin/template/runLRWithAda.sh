$SPARK_HOME/bin/spark-submit \
    --class LRWithAdaExample \
    --master local[*] \
    target/scala-2.11/imllib_2.11-0.0.1.jar \
    hdfs://localhost:9000/data/lr/a9a \
    hdfs://localhost:9000/data/lr/a9a.t \
    4
