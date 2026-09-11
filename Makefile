.PHONY: test test-slow lint release

test:
	poetry run pytest -q

test-slow:
	poetry run pytest -q --run-slow

lint:
	poetry run ruff check --select F rfmix_reader tests

# Usage: make release v=0.7.0
# Bumps the version, commits and tags; the Release workflow builds and
# publishes when the tag is pushed (after the test workflow passes).
release: lint test
	poetry version $(v)
	git add pyproject.toml
	git commit -m "Release version v$$(poetry version -s)"
	git tag v$$(poetry version -s)
	@echo "Now: git push origin main --tags"
