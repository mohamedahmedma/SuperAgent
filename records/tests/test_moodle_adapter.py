"""MoodleAdapter — transport behaviour and reshaping, with no network.

Requests are intercepted, so these assert what the adapter does with a response rather
than whether a Moodle happens to be running. The behaviours worth pinning are the ones
that look like something else when they go wrong:

  * Moodle reports FAILURE WITH HTTP 200. A client trusting the status code returns an
    empty result, which the assistant reads out as "no grades recorded".
  * Moodle REDIRECTS a wrong host and answers with HTML, which reads as a broken
    endpoint rather than a configuration error.
  * `found: false` is NOT an error, and turning it into one leaks the difference
    between "no such student" and "not your child".
"""
import json
from unittest.mock import patch

import pytest

from records.lms import LmsUnavailable, MoodleAdapter


class FakeResponse:
    def __init__(self, payload=None, status_code=200, text="", headers=None):
        self._payload = payload
        self.status_code = status_code
        self.text = text or json.dumps(payload)
        self.headers = headers or {}

    def json(self):
        if self._payload is None:
            raise ValueError("not JSON")
        return self._payload


def adapter(**kwargs) -> MoodleAdapter:
    # Caching off by default so each test exercises the call it makes.
    kwargs.setdefault("cache_ttl_seconds", 0)
    return MoodleAdapter("http://moodle.test", "token-abc", **kwargs)


GRADES_OK = {
    "found": True,
    "studentidnumber": "S-1001",
    "subjects": [
        {
            "courseid": "9001",
            "idnumber": "2026-T1-G7A-MATH",
            "fullname": "Mathematics",
            "percentage": 65.0,
            "academic": {"percentage": 80.0, "unavailable": ""},
            "gradedcount": 3,
            "excludedcount": 1,
            "pendingcount": 1,
            "iscomplete": False,
            "categories": [{"name": "Assessments", "percentage": 80.0}],
        }
    ],
}


class TestTransportFailures:
    def test_an_exception_in_a_200_body_is_a_failure(self):
        """The one that catches people out.

        Moodle answers HTTP 200 with an `exception` key. Treating that as success
        returns no subjects, and the assistant tells a parent there are no grades.
        """
        body = {"exception": "webservice_access_exception", "errorcode": "accessexception",
                "message": "Access control exception"}
        with patch("requests.post", return_value=FakeResponse(body)):
            with pytest.raises(LmsUnavailable):
                adapter().get_subject_grades(student_ref="S-1001", term="2026-T1-")

    def test_a_redirect_names_the_cause(self):
        """Moodle bounces any host that is not its wwwroot and returns HTML.

        Following it would turn a configuration error into a parse failure. The message
        has to say what is actually wrong, because the symptom looks nothing like it.
        """
        response = FakeResponse(None, status_code=303,
                                headers={"location": "http://localhost:8081/"})
        with patch("requests.post", return_value=response):
            with pytest.raises(LmsUnavailable) as caught:
                adapter().get_subject_grades(student_ref="S-1001", term="2026-T1-")

        assert "wwwroot" in str(caught.value)

    def test_a_non_json_body_is_a_failure(self):
        with patch("requests.post", return_value=FakeResponse(None, text="<html>")):
            with pytest.raises(LmsUnavailable):
                adapter().get_subject_grades(student_ref="S-1001", term="2026-T1-")

    def test_a_transport_error_is_a_failure(self):
        with patch("requests.post", side_effect=OSError("connection refused")):
            with pytest.raises(LmsUnavailable):
                adapter().get_subject_grades(student_ref="S-1001", term="2026-T1-")

    def test_a_non_200_status_is_a_failure(self):
        with patch("requests.post", return_value=FakeResponse({}, status_code=500)):
            with pytest.raises(LmsUnavailable):
                adapter().get_subject_grades(student_ref="S-1001", term="2026-T1-")

    def test_redirects_are_not_followed(self):
        """Explicitly asserted, because the default is to follow them."""
        with patch("requests.post", return_value=FakeResponse(GRADES_OK)) as post:
            adapter().get_subject_grades(student_ref="S-1001", term="2026-T1-")

        assert post.call_args.kwargs["allow_redirects"] is False

    def test_a_timeout_is_always_set(self):
        """A hung call must not hang a chat turn a parent is waiting on."""
        with patch("requests.post", return_value=FakeResponse(GRADES_OK)) as post:
            adapter(timeout_seconds=7).get_subject_grades(student_ref="S-1001", term="2026-T1-")

        assert post.call_args.kwargs["timeout"] == 7


class TestUnknownStudent:
    def test_found_false_is_empty_not_an_error(self):
        """Raising here would leak what the facade deliberately hides.

        "No such student", "not your child" and "records restricted" are made
        indistinguishable upstream so a caller cannot enumerate the school roll. An
        exception on one of them puts the difference straight back.
        """
        with patch("requests.post", return_value=FakeResponse({"found": False, "subjects": []})):
            assert adapter().get_subject_grades(student_ref="NOPE", term="2026-T1-") == []

    def test_the_same_holds_for_attendance(self):
        with patch("requests.post", return_value=FakeResponse({"found": False, "subjects": []})):
            assert adapter().get_subject_attendance(student_ref="NOPE", term="2026-T1-") == []


class TestReshaping:
    def test_both_percentages_survive(self):
        with patch("requests.post", return_value=FakeResponse(GRADES_OK)):
            subjects = adapter().get_subject_grades(student_ref="S-1001", term="2026-T1-")

        assert subjects[0].percentage == 65.0
        assert subjects[0].academic_percentage == 80.0

    def test_null_and_zero_stay_distinct(self):
        """`float(value or 0)` would collapse them.

        Null means nothing has been graded; zero means the child scored nothing.
        Reporting the first as the second accuses a child of failing a term nobody has
        marked.
        """
        payload = {
            "found": True,
            "subjects": [
                {"idnumber": "A", "percentage": None, "academic": {"percentage": 0.0}},
            ],
        }
        with patch("requests.post", return_value=FakeResponse(payload)):
            subjects = adapter().get_subject_grades(student_ref="S-1001", term="2026-T1-")

        assert subjects[0].percentage is None
        assert subjects[0].academic_percentage == 0.0

    def test_the_unavailable_reason_is_carried(self):
        payload = {
            "found": True,
            "subjects": [{"idnumber": "A", "percentage": 70.0,
                          "academic": {"percentage": None,
                                       "unavailable": "aggregation_not_summable"}}],
        }
        with patch("requests.post", return_value=FakeResponse(payload)):
            subjects = adapter().get_subject_grades(student_ref="S-1001", term="2026-T1-")

        assert subjects[0].academic_percentage is None
        assert subjects[0].academic_unavailable == "aggregation_not_summable"

    def test_attendance_points_are_carried_for_term_aggregation(self):
        """Without these a term figure can only be an average of percentages, which
        weights a subject with two registers the same as one with forty."""
        payload = {
            "found": True,
            "subjects": [{"idnumber": "A", "percentage": 87.5, "takensessions": 4,
                          "points": 7.0, "maxpoints": 8.0, "bystatus": []}],
        }
        with patch("requests.post", return_value=FakeResponse(payload)):
            subjects = adapter().get_subject_attendance(student_ref="S-1001", term="2026-T1-")

        assert subjects[0].points == 7.0
        assert subjects[0].max_points == 8.0

    def test_the_school_student_number_is_what_gets_sent(self):
        """Not a Moodle user id — the facade never learns one."""
        with patch("requests.post", return_value=FakeResponse(GRADES_OK)) as post:
            adapter().get_subject_grades(student_ref="S-1001", term="2026-T1-")

        assert post.call_args.kwargs["data"]["studentidnumber"] == "S-1001"
        assert post.call_args.kwargs["data"]["term"] == "2026-T1-"


class TestCaching:
    def test_a_repeated_call_is_served_from_cache(self):
        """One chat turn asks for children, then grades, then attendance."""
        with patch("requests.post", return_value=FakeResponse(GRADES_OK)) as post:
            cached = adapter(cache_ttl_seconds=60)
            cached.get_subject_grades(student_ref="S-1001", term="2026-T1-")
            cached.get_subject_grades(student_ref="S-1001", term="2026-T1-")

        assert post.call_count == 1

    def test_a_different_student_is_not_served_the_first_one(self):
        """The cache key must include the student. Anything less is a data leak."""
        with patch("requests.post", return_value=FakeResponse(GRADES_OK)) as post:
            cached = adapter(cache_ttl_seconds=60)
            cached.get_subject_grades(student_ref="S-1001", term="2026-T1-")
            cached.get_subject_grades(student_ref="S-2002", term="2026-T1-")

        assert post.call_count == 2

    def test_a_different_term_is_not_served_the_first_one(self):
        with patch("requests.post", return_value=FakeResponse(GRADES_OK)) as post:
            cached = adapter(cache_ttl_seconds=60)
            cached.get_subject_grades(student_ref="S-1001", term="2026-T1-")
            cached.get_subject_grades(student_ref="S-1001", term="2026-T2-")

        assert post.call_count == 2

    def test_grades_and_attendance_do_not_share_an_entry(self):
        """Same student, same term, different function — the key includes it."""
        with patch("requests.post", return_value=FakeResponse(GRADES_OK)) as post:
            cached = adapter(cache_ttl_seconds=60)
            cached.get_subject_grades(student_ref="S-1001", term="2026-T1-")
            cached.get_subject_attendance(student_ref="S-1001", term="2026-T1-")

        assert post.call_count == 2

    def test_an_expired_entry_is_refetched(self):
        with patch("requests.post", return_value=FakeResponse(GRADES_OK)) as post:
            expiring = adapter(cache_ttl_seconds=0)
            expiring.get_subject_grades(student_ref="S-1001", term="2026-T1-")
            expiring.get_subject_grades(student_ref="S-1001", term="2026-T1-")

        assert post.call_count == 2

    def test_a_failure_is_not_cached(self):
        """A cached outage would keep answering "unavailable" after Moodle recovered."""
        failing = FakeResponse({"exception": "x", "errorcode": "e", "message": "m"})
        cached = adapter(cache_ttl_seconds=60)

        with patch("requests.post", return_value=failing):
            with pytest.raises(LmsUnavailable):
                cached.get_subject_grades(student_ref="S-1001", term="2026-T1-")

        with patch("requests.post", return_value=FakeResponse(GRADES_OK)) as post:
            subjects = cached.get_subject_grades(student_ref="S-1001", term="2026-T1-")

        assert post.call_count == 1
        assert subjects[0].percentage == 65.0
