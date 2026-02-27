from runner import RetryPolicy, backoff_seconds, classify_exception


class FakeHttpError(Exception):
    def __init__(self, status_code):
        self.status_code = status_code


def test_classify_exception_marks_retryable_status_code():
    code, retryable = classify_exception(FakeHttpError(429))
    assert code == "FakeHttpError:429"
    assert retryable is True


def test_backoff_seconds_increases_with_retry_attempts():
    policy = RetryPolicy(backoff_seconds=1.0, max_backoff_seconds=5.0)
    assert backoff_seconds(policy, 1) == 1.0
    assert backoff_seconds(policy, 2) == 2.0
    assert backoff_seconds(policy, 3) == 4.0
    assert backoff_seconds(policy, 10) == 5.0
