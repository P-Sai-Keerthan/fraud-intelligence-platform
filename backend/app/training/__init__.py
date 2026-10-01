"""
Candidate model training (Step 4C-2e).

    python -m app.training.candidates --dataset v2     (from backend/)

trains candidate models with the corrected evaluation recipe
(app/evaluation/stacking.py) on the SAVED evaluation split and writes them
to models/candidates/<dataset>/<name>/. It never writes to models/saved/
(the production models used by /predict) and checks that those files are
byte-identical before and after.
"""
