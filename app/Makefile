PYTHON ?= .venv/bin/python
NODE_BIN ?= ../.tools/node/bin
REPCHAIN := web/backend/app/repchain

.PHONY: check backend frontend-dev frontend-build

check:
	$(PYTHON) -m compileall -q web/backend
	$(PYTHON) -c "import sys; sys.path.insert(0, '$(REPCHAIN)'); import analysis_confidence as c; c.load_calibrator(c.load_config())"
	$(PYTHON) $(REPCHAIN)/analysis_v1.py --help >/dev/null
	$(PYTHON) $(REPCHAIN)/repchain_pipeline.py --help >/dev/null
	$(PYTHON) $(REPCHAIN)/build_issue_cards.py --help >/dev/null
	$(PYTHON) $(REPCHAIN)/feedback_builder.py --help >/dev/null
	$(PYTHON) $(REPCHAIN)/render_clip_report.py --help >/dev/null

backend:
	$(PYTHON) -m uvicorn app.main:app --app-dir web/backend --host 0.0.0.0 --port 8011

frontend-dev:
	cd web/frontend && PATH="$(abspath $(NODE_BIN)):$$PATH" npm run dev

frontend-build:
	cd web/frontend && PATH="$(abspath $(NODE_BIN)):$$PATH" npm run build
