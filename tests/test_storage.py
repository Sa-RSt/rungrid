import datetime
import io
from uuid import uuid4

import pytest

from rungrid.experiment import StaleTrial, TrialResult, VarNamespace
from rungrid.plumbing.storage import BucketFileStorage, CSVSink


def test_csv_sink():
    """Verify that CSVSink writes correct headers and trial rows to a stream."""
    stream = io.StringIO()
    sink = CSVSink(stream)

    v = VarNamespace(set())
    # Create an unfinished trial
    trial = StaleTrial(42, uuid4(), v, tags={"tag1", "tag2"})
    sink.put_trial(trial)

    content = stream.getvalue()
    lines = content.strip().splitlines()
    assert len(lines) == 2  # header + 1 row

    # Check headers
    assert "uuid,optuna_trial_number,tags,step_names" in lines[0]

    # Check trial data in row
    row = lines[1]
    assert str(trial.uuid) in row
    assert "42" in row
    # tags should be semicolon separated inside CSV
    assert "tag1;tag2" in row or "tag2;tag1" in row


def test_bucket_file_storage(tmp_path):
    """Verify BucketFileStorage bucketing, filtering, and trial movement on completion."""
    finished_dir = tmp_path / "finished"
    unfinished_dir = tmp_path / "unfinished"

    storage = BucketFileStorage(
        finished_dir=finished_dir,
        unfinished_dir=unfinished_dir,
        serializing_backend="pickle",
    )

    v = VarNamespace(set())
    trial1 = StaleTrial(1, uuid4(), v, tags={"active", "task1"})
    trial2 = StaleTrial(2, uuid4(), v, tags={"active", "task2"})

    # Save both trials (unfinished by default)
    storage.put_trial(trial1)
    storage.put_trial(trial2)

    # Debug prints
    print("FINISHED DIR:", list(finished_dir.rglob("*")))
    print("UNFINISHED DIR:", list(unfinished_dir.rglob("*")))

    # Verify both can be retrieved as unfinished
    trials = list(storage.get_trials(finished=False))
    print("RETRIEVED TRIALS:", trials)
    assert len(trials) == 2
    uuids = [t.uuid for t in trials]
    assert trial1.uuid in uuids
    assert trial2.uuid in uuids

    # Retrieve with search_tag
    task1_trials = list(storage.get_trials(search_tag="task1"))
    assert len(task1_trials) == 1
    assert task1_trials[0].uuid == trial1.uuid

    # Update trial1 to finished state (attach a dummy result)
    res = TrialResult.make_ok("dummy-result", datetime.datetime.now())
    trial1_finished = trial1.with_result(res)

    # Save finished trial
    storage.put_trial(trial1_finished)

    # Verify trial1 is now finished and trial2 is unfinished
    finished_trials = list(storage.get_trials(finished=True))
    assert len(finished_trials) == 1
    assert finished_trials[0].uuid == trial1.uuid

    unfinished_trials = list(storage.get_trials(finished=False))
    assert len(unfinished_trials) == 1
    assert unfinished_trials[0].uuid == trial2.uuid


def test_bucket_file_storage_consume_and_tag(tmp_path):
    """Verify that consume_and_tag atomically tags and returns the matched trial."""
    finished_dir = tmp_path / "finished"
    unfinished_dir = tmp_path / "unfinished"

    storage = BucketFileStorage(
        finished_dir=finished_dir,
        unfinished_dir=unfinished_dir,
        serializing_backend="pickle",
    )

    v = VarNamespace(set())
    trial = StaleTrial(1, uuid4(), v, tags={"untagged"})
    storage.put_trial(trial)

    # Consume and tag the trial with "processed"
    consumed = storage.consume_and_tag("processed", search_tag="untagged")
    assert consumed is not None
    assert consumed.uuid == trial.uuid
    assert "processed" in consumed.tags

    # Verify the updated trial is in storage with the new tag
    all_trials = list(storage.get_trials())
    assert len(all_trials) == 1
    assert "processed" in all_trials[0].tags


def test_bucket_file_storage_invalid_backend_type_raises_type_error(tmp_path):
    """Verify BucketFileStorage raises TypeError if serializing_backend has an invalid type."""
    with pytest.raises(
        TypeError, match="serializing_backend: expected type str or LoaderDumper"
    ):
        BucketFileStorage(
            finished_dir=tmp_path / "finished",
            unfinished_dir=tmp_path / "unfinished",
            serializing_backend=123,
        )


def test_bucket_storage_thorough_integration(tmp_path):
    """Thorough integration testing of BucketFileStorage using multiple tags, putting, and filtering."""
    finished_dir = tmp_path / "fin"
    unfinished_dir = tmp_path / "unfin"
    storage = BucketFileStorage(
        finished_dir=finished_dir,
        unfinished_dir=unfinished_dir,
        serializing_backend="pickle",
    )

    v = VarNamespace(set())
    trial_a = StaleTrial(101, uuid4(), v, tags={"alpha", "shared"})
    trial_b = StaleTrial(102, uuid4(), v, tags={"beta", "shared"})
    trial_c = StaleTrial(103, uuid4(), v, tags={"gamma"})

    # Put initial trials (all unfinished)
    storage.put_trials([trial_a, trial_b, trial_c])

    # Get trials - unfiltered
    all_trials = list(storage.get_trials())
    assert len(all_trials) == 3

    # Get trials - by single tag
    shared_trials = list(storage.get_trials(search_tag="shared"))
    assert len(shared_trials) == 2
    uuids = {t.uuid for t in shared_trials}
    assert trial_a.uuid in uuids
    assert trial_b.uuid in uuids

    # Get trials - by another tag
    gamma_trials = list(storage.get_trials(search_tag="gamma"))
    assert len(gamma_trials) == 1
    assert gamma_trials[0].uuid == trial_c.uuid

    # Consume and tag trial_a
    consumed = storage.consume_and_tag(applied_tag="processed", search_tag="alpha")
    assert consumed is not None
    assert consumed.uuid == trial_a.uuid
    assert "processed" in consumed.tags
    assert "alpha" in consumed.tags

    # Verify updated trial_a is retrieved with new tag "processed"
    processed_trials = list(storage.get_trials(search_tag="processed"))
    assert len(processed_trials) == 1
    assert processed_trials[0].uuid == trial_a.uuid
    assert "shared" in processed_trials[0].tags

    # Finish trial_b and put back
    res = TrialResult.make_ok(9.9, datetime.datetime.now())
    trial_b_finished = trial_b.with_result(res)
    storage.put_trial(trial_b_finished)

    # Get trials - finished only
    finished = list(storage.get_trials(finished=True))
    assert len(finished) == 1
    assert finished[0].uuid == trial_b.uuid

    # Get trials - unfinished only
    unfinished = list(storage.get_trials(finished=False))
    assert len(unfinished) == 2
    unf_uuids = {t.uuid for t in unfinished}
    assert trial_a.uuid in unf_uuids
    assert trial_c.uuid in unf_uuids
