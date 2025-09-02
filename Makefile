PY=python
.PHONY: install run clean finetune eval
install:
	$(PY) -m pip install -r requirements.txt
run:
	$(PY) tta_mobilenetv3_pncache.py --config config.yaml
finetune:
	$(PY) train_finetune.py --config config_train.yaml
eval:
	$(PY) eval_tta.py --config config_eval.yaml
clean:
	rm -f predictions.csv
