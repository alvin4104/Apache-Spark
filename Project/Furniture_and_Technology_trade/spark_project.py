{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "e77ab927-c8de-4d2e-ab56-95e674880623",
   "metadata": {},
   "outputs": [],
   "source": [
    "from pyspark.sql import SparkSession\n",
    "from pyspark.sql.types import  StructType, StructField, StringType, IntegerType"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "266aa810-9eae-4b06-b5b0-c514c660395d",
   "metadata": {},
   "source": [
    "# Tạo SparkSession\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "bd177b1f-c989-457f-be38-9c024d54c4e7",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "24/12/21 16:25:00 WARN SparkSession: Using an existing Spark session; only runtime SQL configurations will take effect.\n"
     ]
    }
   ],
   "source": [
    "spark = SparkSession.builder \\\n",
    "        .appName(\"alvinnguyen41\") \\\n",
    "        .getOrCreate()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "b9fc40cf-f07d-45d5-80a3-b28afedee730",
   "metadata": {},
   "outputs": [],
   "source": [
    "schema = StructType([\n",
    "          StructField(\"Name\", StringType(), True),\n",
    "          StructField(\"Age\", IntegerType(), True)\n",
    "])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "ea10fa11-90dd-4190-8cb4-e9793c1e4729",
   "metadata": {},
   "outputs": [],
   "source": [
    "data = [(\"Alice\", 28),(\"Box\", 27)]\n",
    "df = spark.createDataFrame(data, schema)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "2d51d418-9bce-468f-ab31-3ad3f873f16b",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "+-----+---+\n",
      "| Name|Age|\n",
      "+-----+---+\n",
      "|Alice| 28|\n",
      "|  Box| 27|\n",
      "+-----+---+\n",
      "\n"
     ]
    }
   ],
   "source": [
    "df.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "8759ed06-e961-4345-865d-2705ef02e12c",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
