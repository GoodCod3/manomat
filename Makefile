PYTHON=python
FILENAME_MAIN=main.py

run:
	${PYTHON} src/${FILENAME}


lint:
	poetry run flake8

isort:
	poetry run  isort . --check-only

isort-fix:
	poetry run  isort .

pre-commit:
	pre-commit run --all-files

freeze-dependencies:
	poetry export -f requirements.txt --output requirements.txt --without-hashes


release-major:
	@poetry version major && \
	echo "Publicando versión $$(poetry version --no-interaction | cut -d ' ' -f 2)" && \
	git ci -am "New release v$$(poetry version --no-interaction | cut -d ' ' -f 2)" && \
	git push origin master && \
	git tag v$$(poetry version --no-interaction | cut -d ' ' -f 2) && \
	git push origin --tags

release-minor:
	@poetry version minor && \
	echo "Publicando versión $$(poetry version --no-interaction | cut -d ' ' -f 2)" && \
	git ci -am "New release v$$(poetry version --no-interaction | cut -d ' ' -f 2)" && \
	git push origin master && \
	git tag v$$(poetry version --no-interaction | cut -d ' ' -f 2) && \
	git push origin --tags


release-patch:
	@poetry version patch && \
	echo "Publicando versión $$(poetry version --no-interaction | cut -d ' ' -f 2)" && \
	git ci -am "New release v$$(poetry version --no-interaction | cut -d ' ' -f 2)" && \
	git push origin master && \
	git tag v$$(poetry version --no-interaction | cut -d ' ' -f 2) && \
	git push origin --tags
