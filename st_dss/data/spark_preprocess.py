"""Objective 01 — Spark / PySpark preprocessing pipeline.

Steps (all in Spark, so the same job scales to a cluster):
  1. cleaning   : speed == 0 or > 90 mph  -> missing
  2. alignment  : per-sensor ordering on a regular 5-min grid; gaps filled by
                  forward fill, then backward fill (window functions)
  3. windowing  : lag_0..lag_{L-1} inputs, calendar features (time of day,
                  day of week) and raw targets y_h at t+h for each horizon
                  (targets stay null where the true value was missing, so
                  evaluation is masked — the METR-LA convention)
  4. split      : chronological train / val / test by timestamp
"""
import os
import time

from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F


def get_spark(app="ST-DSS"):
    return (SparkSession.builder.master(os.environ.get("SPARK_MASTER", "local[*]"))
            .appName(app)
            .config("spark.sql.shuffle.partitions", "16")
            .config("spark.driver.memory", os.environ.get("SPARK_DRIVER_MEMORY", "4g"))
            .config("spark.sql.session.timeZone", "UTC")
            .config("spark.ui.showConsoleProgress", "false")
            .getOrCreate())


def preprocess(long_parquet, out_dir, n_lags, horizons, split, log=print):
    t0 = time.time()
    spark = get_spark()
    spark.sparkContext.setLogLevel("ERROR")
    df = spark.read.parquet(long_parquet)
    n_raw = df.count()

    # 1. cleaning
    df = df.withColumn("speed_raw",
                       F.when((F.col("speed") > 0) & (F.col("speed") <= 90), F.col("speed")))
    missing = df.filter(F.col("speed_raw").isNull()).count()

    # 2. alignment + gap filling
    w = Window.partitionBy("sensor").orderBy("ts")
    w_ff = w.rowsBetween(Window.unboundedPreceding, 0)
    w_bf = w.rowsBetween(0, Window.unboundedFollowing)
    df = df.withColumn("speed_f", F.last("speed_raw", ignorenulls=True).over(w_ff))
    df = df.withColumn("speed_f", F.coalesce("speed_f", F.first("speed_raw", ignorenulls=True).over(w_bf)))

    # calendar features
    df = (df.withColumn("tod", (F.hour("ts") * 60 + F.minute("ts")) / F.lit(1440.0))
            .withColumn("dow", (F.dayofweek("ts") + 5) % 7)          # Monday = 0
            .withColumn("weekend", (F.col("dow") >= 5).cast("int")))

    # 3. windowing
    for k in range(n_lags):
        df = df.withColumn(f"lag_{k}", F.lag("speed_f", k).over(w).cast("float"))
    for h in horizons:
        df = df.withColumn(f"y_{h}", F.lead("speed_raw", h).over(w).cast("float"))
    df = df.dropna(subset=[f"lag_{n_lags - 1}"])

    # 4. chronological split
    ts = df.select("ts").distinct().orderBy("ts")
    n_ts = ts.count()
    bounds = ts.collect()
    t_train = bounds[int(n_ts * split[0])][0]
    t_val = bounds[int(n_ts * (split[0] + split[1]))][0]
    df = df.withColumn("split", F.when(F.col("ts") < t_train, "train")
                       .when(F.col("ts") < t_val, "val").otherwise("test"))

    cols = (["ts", "sensor", "tod", "dow", "weekend"] + [f"lag_{k}" for k in range(n_lags)]
            + [f"y_{h}" for h in horizons] + ["split"])
    out = df.select(*cols)
    try:
        out.write.mode("overwrite").partitionBy("split").parquet(out_dir)
        counts = {r["split"]: r["count"] for r in
                  spark.read.parquet(out_dir).groupBy("split").count().collect()}
    except Exception as e:  # typically Windows without Hadoop winutils
        log(f"[spark] Spark parquet write failed ({type(e).__name__}); "
            "falling back to Arrow collection + pyarrow write (Windows/winutils issue)")
        counts = _write_via_arrow(spark, out, out_dir, log)
    spark.stop()
    report = dict(raw_rows=n_raw, missing_rows=missing, missing_pct=100 * missing / n_raw,
                  train_end=str(t_train), val_end=str(t_val), seconds=time.time() - t0, **counts)
    log(f"[spark] {report}")
    return report


def _write_via_arrow(spark, out, out_dir, log=print):
    """Fallback writer: all transformations still run in Spark; only the final
    file write is done with pyarrow (same split=<name>/ folder layout)."""
    import shutil
    spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
    shutil.rmtree(out_dir, ignore_errors=True)
    counts = {}
    for name in ["train", "val", "test"]:
        part_dir = os.path.join(out_dir, f"split={name}")
        os.makedirs(part_dir, exist_ok=True)
        pdf = out.filter(F.col("split") == name).drop("split").toPandas()
        pdf.to_parquet(os.path.join(part_dir, "part-0.parquet"), index=False)
        counts[name] = len(pdf)
        log(f"[spark] wrote {name}: {len(pdf):,} rows")
        del pdf
    return counts
