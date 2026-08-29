"""Demo and test data for the SIS: one fictional school, both sections, every role.

    python -m sis.demo load     write it
    python -m sis.demo reset    remove it and write it again
    python -m sis.demo status   say what is there
    python -m sis.demo accounts print the credentials table
    python -m sis.demo classes  print every class with its generated title

Three files: `blueprint.py` declares what exists, `names.py` holds the invented people,
`seeder.py` does the writing. Nothing in here is imported by the service at runtime — the
package exists to be run, and `sis/app.py` never touches it.
"""
