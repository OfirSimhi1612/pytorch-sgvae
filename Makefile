train:
	python -m src.scripts.train

test:
	python -m src.scripts.test $(ARGS)

create-dataset:
	python -m src.scripts.create_dataset