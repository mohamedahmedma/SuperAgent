"""Creating a school, database and all, from the registrar console.

Distinct from `POST /v1/schools`, and the distinction is the whole reason this file
exists. That route writes a `schools` row *into the database the request is already bound
to* -- it names a branch inside an estate that already holds it. This one brings a new
database into existence, migrates it, and records its connection so the next process start
finds it. One is a row; the other is a tenant.

## This route is opt-in, and off unless a deployment arms it

Creating a database needs a role that can `CREATE DATABASE`, and a role that can create
one can drop one. The process answering parent requests has no business holding that, so
the credential is read from `SIS_ADMIN_DATABASE_URL` and **this route refuses with 503
when it is unset**. A deployment that would rather provision from a terminal simply never
sets it, and `python -m sis.schools provision` remains the way in.

## What a caller gets back, and what it does not

Never the connection string. It carries the estate's database password, and a registrar
console that displayed it once would have it in a browser cache, a screenshot and a
support ticket by the end of the week. The response names the school, the database and
the variable that points at it, which is everything an operator needs to find it.

## The multi-worker caveat, stated because it will surprise someone

`SIS_SCHOOLS` and the per-school URLs are read once per process. This route updates the
registry in **the worker that served the request**, so that worker can answer for the new
school immediately -- and the others cannot, until they restart and re-read `.env`. With
more than one worker, treat provisioning as complete only after a rolling restart. The
alternative would be a shared cache invalidation across workers, which is a much larger
mechanism than the once-a-term operation it would serve.
"""
import os

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from sis.api.deps import RegistrarCaller
from sis.api.routers import domain_errors, error_responses
from sis.application.services.estate import TEMPLATE_VAR, EstateService, plan_provision
from sis.infrastructure.estate import DotEnvConfigStore
from sis.infrastructure.estate.provisioners import PROJECT_ROOT, provisioner_for
from sis.infrastructure.estate.seeding import seed_school_row
from sis.tenancy import SCHOOLS_VAR, get_registry, reset_registry_cache

router = APIRouter(prefix="/v1/admin", tags=["admin"])

#: The credential that lets this process create a database. Absent by design.
ADMIN_URL_VAR = "SIS_ADMIN_DATABASE_URL"


class ProvisionSchoolIn(BaseModel):
    code: str = Field(
        ...,
        description="The school's code, e.g. `NCS`. Becomes both the database name and "
        "the environment variable that points at it, folded the same way for each.",
        examples=["NCS"],
    )
    name_en: str = Field("", description="English name; defaults to the code.")
    name_ar: str = Field("", description="Arabic name.")


class ProvisionSchoolOut(BaseModel):
    code: str
    database: str = Field(description="The database created, without its connection.")
    env_var: str = Field(description="The variable now carrying its connection in .env.")
    schools: str = Field(description="SIS_SCHOOLS as it now reads.")
    restart_required: bool = Field(
        description="True whenever the service runs more than one worker: the others "
        "still hold the registry they read at startup."
    )


@router.post(
    "/schools",
    response_model=ProvisionSchoolOut,
    status_code=status.HTTP_201_CREATED,
    summary="Create a school's database and record its connection",
    description=(
        "Creates one database for the school, migrates it to head, writes its connection "
        "into the deployment's `.env`, and seeds the school's own row. Distinct from "
        "`POST /v1/schools`, which names a branch inside a database that already exists.\n\n"
        "**503** when `SIS_ADMIN_DATABASE_URL` or `SIS_DATABASE_URL_TEMPLATE` is unset — "
        "provisioning over HTTP is opt-in, because it needs a role that can create and "
        "therefore drop a database.\n\n"
        "**409** when the code is already served, or its database already exists. "
        "Provisioning never runs over a database that may hold another school's rows."
    ),
    responses=error_responses(401, 403, 409, 422, 503),
)
def provision_school(
    body: ProvisionSchoolIn, caller: RegistrarCaller
) -> ProvisionSchoolOut:
    template = os.getenv(TEMPLATE_VAR, "").strip()
    admin_url = os.getenv(ADMIN_URL_VAR, "").strip()
    if not template or not admin_url:
        missing = " and ".join(
            name
            for name, value in ((TEMPLATE_VAR, template), (ADMIN_URL_VAR, admin_url))
            if not value
        )
        raise _unavailable(
            f"provisioning from the API is not enabled on this deployment: {missing} "
            f"is not set. Use `python -m sis.schools provision {body.code}` from the "
            "server, or set it to arm this route."
        )

    existing = get_registry().codes

    with domain_errors():
        plan = plan_provision(body.code, template=template, existing_codes=existing)
        service = EstateService(
            provisioner_for(plan.database_url, admin_url=admin_url),
            DotEnvConfigStore(PROJECT_ROOT / ".env"),
        )
        service.provision(body.code, template=template, existing_codes=existing)

    # Make the school routable in this worker, so the seeding below resolves it. The
    # registry was read before it existed; without this the next line raises UnknownSchool
    # naming the school that was just created.
    os.environ[plan.env_var] = plan.database_url
    os.environ[SCHOOLS_VAR] = plan.schools_value
    reset_registry_cache()

    seed_school_row(plan.code, name_en=body.name_en, name_ar=body.name_ar)

    return ProvisionSchoolOut(
        code=plan.code,
        database=plan.database_url.rsplit("/", 1)[-1].split("?", 1)[0],
        env_var=plan.env_var,
        schools=plan.schools_value,
        restart_required=True,
    )


def _unavailable(message: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail={"code": "provisioning_disabled", "message": message, "field": None},
    )
