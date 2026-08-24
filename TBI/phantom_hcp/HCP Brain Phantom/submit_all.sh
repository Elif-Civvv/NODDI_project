#!/bin/bash
# Submit all six per-(protocol, pathology) MCMC jobs.
for f in run_mcmc_hcp_astrogliosis.pbs run_mcmc_hcp_edema.pbs run_mcmc_hcp_chronic_tbi.pbs \
         run_mcmc_novel_astrogliosis.pbs run_mcmc_novel_edema.pbs run_mcmc_novel_chronic_tbi.pbs; do
    qsub "$f" && echo "submitted $f"
done
