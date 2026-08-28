"""Provisioning a school: the naming, the refusals, and the order the side effects run in.

This is the coverage `scripts/schools.py` never had. `provision` creates a database and
`split` carves one, and until the rules moved out of the argparse handlers there was no
way to assert either without a real server -- so nothing did, on the two operations in the
estate that can lose a school's rows.

Everything below the first section drives fakes. `plan_provision` is pure, so most of the
rules are testable with three strings and no I/O at all, which is the point of extracting
it.
"""
from pathlib import Path

import pytest

from sis.application.services.estate import (
    TEMPLATE_VAR,
    EstateService,
    SchoolAlreadyProvisioned,
    plan_provision,
)
from sis.domain.errors import InvalidCode, ValidationError
from sis.infrastructure.estate import DotEnvConfigStore
from sis.tenancy import TenancyMisconfigured, env_suffix

POSTGRES = "postgresql+psycopg2://sis:pw@db:5432/sis_{slug}"


# ---------------------------------------------------------------------------
# Naming. Pure: no server, no file.
# ---------------------------------------------------------------------------


def test_the_slug_is_the_code_lowercased_with_punctuation_folded() -> None:
    plan = plan_provision("NC-1", template=POSTGRES, existing_codes=())
    assert plan.database_url == "postgresql+psycopg2://sis:pw@db:5432/sis_nc_1"


def test_the_database_and_the_variable_fold_the_code_the_same_way() -> None:
    """The one thing that must never drift.

    If the database name and the variable naming it disagreed about how a code folds,
    a school would be created in one place and looked for in another -- and the symptom
    is `TenancyMisconfigured` at the next restart, long after the provisioning that
    caused it.
    """
    plan = plan_provision("NC.2", template=POSTGRES, existing_codes=())
    assert plan.env_var == f"SIS_DATABASE_URL_{env_suffix('NC.2')}"
    assert plan.database_url.endswith(env_suffix("NC.2").lower())


def test_a_lowercase_code_resolves_to_the_same_school_as_its_normalised_form() -> None:
    assert plan_provision("nc", template=POSTGRES, existing_codes=()).code == "NC"


def test_the_new_school_is_appended_to_the_existing_list_in_order() -> None:
    plan = plan_provision("NCS", template=POSTGRES, existing_codes=("MAIN", "ALEX"))
    assert plan.schools_value == "MAIN,ALEX,NCS"


def test_the_plan_says_exactly_which_two_variables_change() -> None:
    plan = plan_provision("NCS", template=POSTGRES, existing_codes=("MAIN",))
    assert plan.config_changes == {
        "SIS_DATABASE_URL_NCS": "postgresql+psycopg2://sis:pw@db:5432/sis_ncs",
        "SIS_SCHOOLS": "MAIN,NCS",
    }


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


def test_a_school_that_already_exists_is_refused() -> None:
    with pytest.raises(SchoolAlreadyProvisioned):
        plan_provision("MAIN", template=POSTGRES, existing_codes=("MAIN",))


def test_an_invalid_code_never_reaches_a_url() -> None:
    with pytest.raises(InvalidCode):
        plan_provision("no spaces", template=POSTGRES, existing_codes=())


def test_a_template_with_no_placeholder_is_refused() -> None:
    """Every school would render the same URL and quietly share one database."""
    with pytest.raises(TenancyMisconfigured) as error:
        plan_provision("NCS", template="postgresql://db/sis", existing_codes=())
    assert "placeholder" in str(error.value)


def test_an_unset_template_is_refused_by_name() -> None:
    with pytest.raises(TenancyMisconfigured) as error:
        plan_provision("NCS", template="   ", existing_codes=())
    assert TEMPLATE_VAR in str(error.value)


def test_a_database_name_postgresql_would_truncate_is_refused() -> None:
    """63 bytes is silent truncation, and two long codes sharing a prefix collide."""
    with pytest.raises(ValidationError):
        plan_provision(
            "A" * 60, template="postgresql://db:5432/sis_{slug}", existing_codes=()
        )


# ---------------------------------------------------------------------------
# The order the side effects run in
# ---------------------------------------------------------------------------


class RecordingProvisioner:
    def __init__(self, *, exists: bool = False, fail_on: str = "") -> None:
        self._exists = exists
        self._fail_on = fail_on
        self.calls: list[str] = []

    def exists(self, database_url: str) -> bool:
        self.calls.append("exists")
        return self._exists

    def create(self, database_url: str) -> None:
        self.calls.append("create")
        if self._fail_on == "create":
            raise RuntimeError("server refused")

    def migrate(self, database_url: str) -> None:
        self.calls.append("migrate")
        if self._fail_on == "migrate":
            raise RuntimeError("migration failed")


class RecordingConfig:
    def __init__(self) -> None:
        self.written: dict[str, str] = {}

    def read(self) -> dict[str, str]:
        return dict(self.written)

    def update(self, values: dict[str, str]) -> None:
        self.written.update(values)


def test_the_database_is_created_and_migrated_before_anything_is_recorded() -> None:
    provisioner, config = RecordingProvisioner(), RecordingConfig()
    EstateService(provisioner, config).provision(
        "NCS", template=POSTGRES, existing_codes=("MAIN",)
    )
    assert provisioner.calls == ["exists", "create", "migrate"]
    assert config.written["SIS_SCHOOLS"] == "MAIN,NCS"


def test_a_failed_migration_records_nothing() -> None:
    """The ordering rule, asserted.

    A school written into SIS_SCHOOLS with no reachable database makes
    `get_registry` raise at startup, so the service stops booting entirely. An orphaned
    database records nothing anywhere and the next attempt refuses to overwrite it. Only
    one of those is recoverable without editing a file by hand.
    """
    provisioner, config = RecordingProvisioner(fail_on="migrate"), RecordingConfig()
    with pytest.raises(RuntimeError):
        EstateService(provisioner, config).provision(
            "NCS", template=POSTGRES, existing_codes=("MAIN",)
        )
    assert config.written == {}


def test_an_existing_database_stops_provisioning_before_it_creates() -> None:
    provisioner, config = RecordingProvisioner(exists=True), RecordingConfig()
    with pytest.raises(SchoolAlreadyProvisioned):
        EstateService(provisioner, config).provision(
            "NCS", template=POSTGRES, existing_codes=()
        )
    assert provisioner.calls == ["exists"]
    assert config.written == {}


# ---------------------------------------------------------------------------
# The .env file
# ---------------------------------------------------------------------------


@pytest.fixture
def env_file(tmp_path: Path) -> Path:
    path = tmp_path / ".env"
    path.write_text(
        "# The estate's configuration.\n"
        "SIS_SCHOOLS=MAIN\n"
        "\n"
        "# Which database the main branch answers from.\n"
        "SIS_DATABASE_URL_MAIN=postgresql://db/sis_main\n",
        encoding="utf-8",
    )
    return path


def test_updating_a_key_leaves_every_comment_where_it_was(env_file: Path) -> None:
    """A generated file would delete the explanation this .env carries."""
    DotEnvConfigStore(env_file).update({"SIS_SCHOOLS": "MAIN,NCS"})
    text = env_file.read_text(encoding="utf-8")
    assert "# The estate's configuration." in text
    assert "# Which database the main branch answers from." in text
    assert "SIS_SCHOOLS=MAIN,NCS" in text


def test_a_new_school_is_appended_and_readable_back(env_file: Path) -> None:
    store = DotEnvConfigStore(env_file)
    store.update(
        {
            "SIS_SCHOOLS": "MAIN,NCS",
            "SIS_DATABASE_URL_NCS": "postgresql://db/sis_ncs",
        }
    )
    values = store.read()
    assert values["SIS_SCHOOLS"] == "MAIN,NCS"
    assert values["SIS_DATABASE_URL_NCS"] == "postgresql://db/sis_ncs"
    assert values["SIS_DATABASE_URL_MAIN"] == "postgresql://db/sis_main"


def test_an_existing_key_is_edited_in_place_and_not_duplicated(env_file: Path) -> None:
    DotEnvConfigStore(env_file).update({"SIS_DATABASE_URL_MAIN": "postgresql://db/x"})
    body = env_file.read_text(encoding="utf-8")
    assert body.count("SIS_DATABASE_URL_MAIN=") == 1


def test_a_password_containing_a_hash_survives_the_round_trip(env_file: Path) -> None:
    """Unquoted, everything after the `#` is read as a comment and the password is wrong."""
    secret = "postgresql://sis:pa#ss@db/sis_ncs"
    store = DotEnvConfigStore(env_file)
    store.update({"SIS_DATABASE_URL_NCS": secret})
    assert store.read()["SIS_DATABASE_URL_NCS"] == secret


def test_reading_a_file_that_is_not_there_is_empty_not_an_error(tmp_path: Path) -> None:
    assert DotEnvConfigStore(tmp_path / "absent.env").read() == {}


def test_the_lock_is_released_so_a_second_write_succeeds(env_file: Path) -> None:
    store = DotEnvConfigStore(env_file)
    store.update({"SIS_SCHOOLS": "MAIN,A"})
    store.update({"SIS_SCHOOLS": "MAIN,A,B"})
    assert store.read()["SIS_SCHOOLS"] == "MAIN,A,B"
    assert not env_file.with_name(".env.lock").exists()


# ---------------------------------------------------------------------------
# The paths the adapters resolve
# ---------------------------------------------------------------------------


def test_the_provisioner_finds_the_real_alembic_config() -> None:
    """Counting directory levels from __file__ is how this broke the first time.

    `PROJECT_ROOT` was `parents[2]`, which is `sis/`, so every migration was attempted
    with `sis/sis/alembic.ini` and every provision failed with alembic reporting a
    missing `script_location` -- a message that says nothing about the real cause.
    """
    from sis.infrastructure.estate.provisioners import PROJECT_ROOT, _ALEMBIC_INI

    assert (PROJECT_ROOT / "sis" / "app.py").exists()
    assert _ALEMBIC_INI.exists()


def test_a_url_scheme_picks_its_own_provisioner() -> None:
    from sis.infrastructure.estate import (
        PostgresProvisioner,
        SqliteProvisioner,
        provisioner_for,
    )

    assert isinstance(provisioner_for("sqlite:///./x.db"), SqliteProvisioner)
    assert isinstance(
        provisioner_for("postgresql://db/x", admin_url="postgresql://db/postgres"),
        PostgresProvisioner,
    )


def test_postgres_provisioning_without_an_admin_credential_is_refused() -> None:
    """Fail closed, and name the variable. The application's own role must not do this."""
    from sis.infrastructure.estate import ProvisioningFailed, provisioner_for

    with pytest.raises(ProvisioningFailed) as error:
        provisioner_for("postgresql://db/sis_ncs", admin_url="")
    assert "SIS_ADMIN_DATABASE_URL" in str(error.value)


def test_a_password_never_reaches_an_error_message() -> None:
    from sis.infrastructure.estate.provisioners import _redacted

    assert "hunter2" not in _redacted("postgresql://sis:hunter2@db:5432/sis_ncs")
