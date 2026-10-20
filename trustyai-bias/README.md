# TrustyAI Bias Monitoring for the Server Failure Model

**Scenario**: Server Predictive Maintenance

This folder trains the server failure prediction model from `workbench/working`, deploys it with OpenVINO Model Server, and monitors it with the TrustyAI bias metrics.
The folder is self-contained: run the notebooks from this folder.

- **SPD** (statistical parity difference): difference in the rate of the favorable outcome between two groups. Fair range: -0.1 to 0.1.
- **DIR** (disparate impact ratio): ratio of the same rates. Fair range: 0.8 to 1.25.

The protected attributes are the `workload_type_*` columns (`1` = that workload type, `0` = all other types).
The favorable outcome is `0`, no failure predicted.

## Files

| File | Purpose |
|---|---|
| `server_metrics.csv` | Server metrics data set, 15000 records. |
| `data_utils.py` | Preprocessing. Scales the 10 numeric columns, keeps the `workload_type_*` one-hot columns as `0`/`1`, and uses a fixed 14-column order (`FEATURES`). |
| `train_model.ipynb` | Trains the model, builds the serving model, exports it, and uploads it to S3. |
| `trustyai_bias_test.ipynb` | Sends inference requests, maps the field names, computes SPD and DIR, and schedules the metrics. |

## Model Requirements for Bias Metrics

| Requirement | Reason |
|---|---|
| One-hot columns are not scaled | TrustyAI selects each group by exact value (`0.0` or `1.0`). |
| Output is a class (`0`/`1`, INT32) | The favorable outcome is compared by value, so a probability never matches. |
| Input is FP64 | TrustyAI does not match group values on FP32 input columns, and returns `NaN`. |
| `fingerprint.pb` is not uploaded | OpenVINO Model Server can try to load that file as the model. |

The serving model built in `train_model.ipynb` wraps the trained model: FP64 input, cast to FP32, and a `0.5` threshold.

## Prerequisites

- OpenShift AI with the TrustyAI component enabled, and the KServe logger CA bundle configured in the `inferenceservice-config` config map (cluster administrator).
- A project that contains only OpenVINO Model Server models. TrustyAI adds its logger to every inference service in the project.
- A workbench with an S3 connection, which provides the `AWS_*` environment variables.
- In the project, the logger CA config map and the TrustyAI service:

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: kserve-logger-ca-bundle
  annotations:
    service.beta.openshift.io/inject-cabundle: "true"
data: {}
---
apiVersion: trustyai.opendatahub.io/v1
kind: TrustyAIService
metadata:
  name: trustyai-service
spec:
  storage:
    format: "PVC"
    folder: "/inputs"
    size: "1Gi"
  data:
    filename: "data.csv"
    format: "CSV"
  metrics:
    schedule: "5s"
```

## Procedure

1. Run all cells of `train_model.ipynb`.
   The notebook creates `test_data.csv` and uploads the model to `models/failure-classifier/1/` in the S3 bucket.
2. Deploy the model:
   - Serving runtime: OpenVINO Model Server
   - Model framework: `tensorflow - 2`
   - Path: `models/failure-classifier`
   - Model name: `failure-classifier`
3. In `trustyai_bias_test.ipynb`, set `MODEL_NAME`, `INFER_URL`, `TRUSTY_ROUTE`, and `TRUSTY_TOKEN`, then run all cells.
4. In the OpenShift AI dashboard, open the model and select the **Model bias** tab.

## Results of a Test Run

3000 test records, `batchSize` 3000:

| Workload type | SPD | DIR |
|---|---|---|
| compute | 0.0031 | 1.0037 |
| database | -0.0185 | 0.9782 |
| storage | 0.0126 | 1.0150 |
| web | 0.0028 | 1.0034 |

All values are in the fair range.

## Notes

- `batchSize` is the number of most recent inference records that a metric uses.
- With a small `batchSize`, for example 300, the values vary more between batches, because each workload group has fewer records.
- TrustyAI stores the data schema per model name. If the model name already has data with a different input type, TrustyAI rejects the new inference records. Use a new model name in that case.
- Each run of step 5 of `trustyai_bias_test.ipynb` schedules new metrics. Delete the previous ones with `DELETE /metrics/group/fairness/<metric>/request`.
- Use the same form for all scheduled metrics of a metric type: either all with a `requestName`, or all without. TrustyAI does not publish the metrics of the other form.
