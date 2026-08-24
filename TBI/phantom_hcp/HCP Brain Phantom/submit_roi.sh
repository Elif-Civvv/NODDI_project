#!/bin/bash
for f in roi_astrogliosis_hcp.pbs roi_astrogliosis_novel.pbs roi_edema_hcp.pbs roi_edema_novel.pbs; do
    qsub "$f" && echo "submitted $f"
done
