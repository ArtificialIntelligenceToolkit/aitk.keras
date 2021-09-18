test::
	pytest -vvv tests

upload::
	rm -rf dist/*
	python setup.py bdist_wheel
	twine upload dist/*
