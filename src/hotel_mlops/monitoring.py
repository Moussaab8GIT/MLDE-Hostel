"""Read metrics from persistent data at scrape time, including after API restarts."""
from prometheus_client.core import GaugeMetricFamily

from hotel_mlops.settings import CONFIG, DB_PATH, REPORTS_DIR
from hotel_mlops.storage import feedback_metrics, flow_reports


class ProjectCollector:
    def collect(self):
        config = CONFIG["monitoring"]
        reports = flow_reports(REPORTS_DIR)
        for name, description, value in (
            ("training_runs_recorded", "Persisted training flow executions", len(reports)),
            ("training_failures_recorded", "Persisted failed training executions",
             sum(report["status"] == "failed" for report in reports)),
        ):
            metric = GaugeMetricFamily(name, description)
            metric.add_metric([], value)
            yield metric
        if reports:
            latest = reports[-1]
            for name, value in (("training_last_success", int(latest["status"] == "completed")),
                                ("training_last_duration_seconds", latest["duration_seconds"]),
                                ("training_last_finished_timestamp", latest["finished_at"])):
                metric = GaugeMetricFamily(name, name)
                metric.add_metric([], value)
                yield metric
        results = feedback_metrics(DB_PATH, config["minimum_feedback"], config["feedback_window_days"])
        for field in ("feedback_count", "precision", "recall", "f1"):
            metric_name = "model_feedback_count" if field == "feedback_count" else "model_feedback_" + field
            metric = GaugeMetricFamily(metric_name,
                                       "Feedback " + field + " in configured rolling window",
                                       labels=["model_version"])
            for result in results:
                if result[field] is not None:
                    metric.add_metric([result["model_version"]], result[field])
            yield metric
