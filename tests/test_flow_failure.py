from types import SimpleNamespace
import pytest

import hotel_mlops.orchestration.training_flow as workflow
from hotel_mlops.storage import flow_reports


def test_validation_failure_blocks_training_and_writes_local_alert(tmp_path, monkeypatch):
    trained = []
    def reject(*args):
        raise ValueError("controlled invalid batch")
    monkeypatch.setattr(workflow, "validate_training_data", reject)
    monkeypatch.setattr(workflow, "train_log_and_promote", lambda *args: trained.append(True))
    monkeypatch.setattr(workflow, "REPORTS_DIR", tmp_path)
    monkeypatch.setattr(workflow, "get_run_logger", lambda: SimpleNamespace(error=lambda *args: None))
    with pytest.raises(ValueError, match="controlled invalid batch"):
        workflow.hotel_cancellation_training.fn()
    assert trained == []
    reports = flow_reports(tmp_path)
    assert len(reports) == 1
    assert reports[0]["status"] == "failed"
    assert reports[0]["error_type"] == "ValueError"
