#!/bin/bash

sbatch swds/train_job_e_m.sh
sbatch swds/train_job_e_n.sh
sbatch swds/train_job_e_r.sh
sbatch swds/train_job_m_e.sh
sbatch swds/train_job_m_n.sh
sbatch swds/train_job_m_r.sh
sbatch swds/train_job_n_e.sh
sbatch swds/train_job_n_m.sh
sbatch swds/train_job_n_r.sh
sbatch swds/train_job_r_e.sh
sbatch swds/train_job_r_m.sh
sbatch swds/train_job_r_n.sh
