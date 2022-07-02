#!/bin/bash
find . -name 'coverage.txt' -delete
poetry run pytest --cov-report term --cov comment_classification tests/ >>.logs/coverage.txt
