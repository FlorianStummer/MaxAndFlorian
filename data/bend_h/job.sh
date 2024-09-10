#!/bin/sh
source /eos/experiment/shadows/user/flstumme/venv/mlpy/bin/activate
N=$1

echo "N = "${N}
python /eos/experiment/shadows/user/flstumme/ai/MaxAndFlorian/data/bend_h/prepare_dataset.py ${N}
echo "done"
date
