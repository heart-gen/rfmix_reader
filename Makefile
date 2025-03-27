    version:
        poetry version $(v)
        git add pyproject.toml
        git commit -m "v$$(poetry version -s)"
        git tag v$$(poetry version -s)
        git push origin main --tags
        poetry build
        poetry publish
