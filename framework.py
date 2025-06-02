import numpy as np
import pandas as pd
import xarray as xr
import warnings
import json
import os
import sys
from scipy.stats import qmc
import multiprocessing as mp
warnings.filterwarnings("ignore")
import time
from scipy.optimize import minimize
from neldermead import minimize_neldermead
from metrics import metrics, SCMmetrics
import pickle
import sys
sys.path.append("/global/homes/z/zhangtao/E3SM/UQ_climate/alg/BayesianOptimization")
sys.path.append('TuRBO/')
from turbo import Turbo1
from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from bayes_opt.util import load_logs

from loguru import logger
logger.remove()
fmt = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <cyan>{level}</cyan> | {message}"
logger.add(sys.stdout, format=fmt)
log_path = f'tuning.log'
if os.path.exists(log_path):
    os.remove(log_path)
ii = logger.add(log_path)

class uq:
    def __init__(self,name):
        self.config_paras(name)

    def config_paras(self, name):
        with open(name, 'r') as file:
            data = json.load(file)

        self.para_names = list(data.keys())
        self.lbs = [data[k][0] for k in data]
        self.ubs = [data[k][1] for k in data]
        self.dim = len(self.para_names)

        print(self.para_names)
        print(self.lbs)
        print(self.ubs)

    def lhs_sample(self,num, continue_run=False,continue_id=0):
        if continue_run:
            tmp = np.load('data/lhs_sample.npy')
            self.sample_scaled = tmp[continue_id:continue_id+num,:]
            self.continue_id = continue_id
        else:
            sampler = qmc.LatinHypercube(d=self.dim)
            sample = sampler.random(n=num)
            self.sample_scaled = qmc.scale(sample, self.lbs, self.ubs)
            np.save('lhs_sample',self.sample_scaled)

    def run_case(self,id,data,dry_run=False,batch_run=False):
        print(f'{id=}')
      
        #if id % 2 == 0:
        #    model_path = '/lcrc/group/e3sm/ac.tzhang/E3SMv3/20241006.v3.LR.F2010_tuning/case_scripts'
        #else:
        model_path = '/pscratch/sd/z/zhangtao/E3SMv3/HR/v3.HR.F2010.STE.tune/case_scripts'
        os.chdir(model_path)

        #config parameter set to replace the values
        para_set = {}
        mesg = ''
        for i,n in enumerate(self.para_names):
            para_set[n] = data[i]
            mesg = f'{mesg} {n}={data[i]:.5e}'

        print(para_set)

        # replace the values
        for k in para_set:
            rep_command = "sed -i '/\<"+k+"\>/c\ "+k+"="+str(para_set[k])+"' user_nl_eam"
            os.system(rep_command)

        # run the model
        if dry_run == False:
            if batch_run == False:
                os.system("sbatch submit_2mons.sh > case_id")
                jid = os.popen("tail -n 1 case_id |awk '{print $4}'").read().strip()

                logger.debug(f'CaseID={id},Submit E3SM with job id {jid}')

                while os.popen("squeue -u zhangtao").read().find(jid) != -1:
                    time.sleep(60)

                logger.debug(f'Finish E3SM with job id {jid}')
            
            else:
                os.system('./submit_2mons.sh')
                
              

        self.archive_model(id)

    def evaluate_scm(self,data):
        model_path = "/pscratch/sd/z/zhangtao/E3SMv3/SCM_runs/e3sm_scm_ARM97/T000"
        os.chdir(model_path)
        os.system("export LD_LIBRARY_PATH=/opt/intel/oneapi/mkl/2023.2.0/lib/intel64:$LD_LIBRARY_PATH")

        #config parameter set to replace the values
        para_set = {}
        mesg = ''
        for i,n in enumerate(self.para_names):
            para_set[n] = data[i]
            mesg = f'{mesg} {n}={data[i]:.5e}'

        # replace the values
        for k in para_set:
            rep_command = "sed -i '/\<"+k+"\>/c\ "+k+"="+str(para_set[k])+"' atm_in"
            os.system(rep_command)
        
        #os.system(f"./e3sm.exe > /dev/null 2>&1")
        os.system("./e3sm.exe")

        mm = SCMmetrics()
        value = mm.get_ARM97()
        return value

        

    def run_scm(self,id,data,turn_run=False):
        id = self.continue_id + id
        logger.info(f'SCM {id=}')
      
        if turn_run:
            model_path = '/pscratch/sd/z/zhangtao/E3SMv3/SCM_runs/e3sm_scm_ARM97'
        else:
            model_path = '/pscratch/sd/z/zhangtao/E3SMv3/SCM_runs/e3sm_scm_ARM95'
        os.chdir(model_path)
        if turn_run:
            prefix = 'T'
            os.chdir(f'T000')
        else:
            prefix = 'S' # sampling run
            os.system(f"rm -rf {prefix}{id:03d}")
            os.system(f"cp -r run {prefix}{id:03d}")
            os.chdir(f'{prefix}{id:03d}')
    
        #config parameter set to replace the values
        para_set = {}
        mesg = ''
        for i,n in enumerate(self.para_names):
            para_set[n] = data[i]
            mesg = f'{mesg} {n}={data[i]:.5e}'

        #logger.debug(para_set)

        # replace the values
        for k in para_set:
            rep_command = "sed -i '/\<"+k+"\>/c\ "+k+"="+str(para_set[k])+"' atm_in"
            os.system(rep_command)
        
        os.system(f"./e3sm.exe > /dev/null 2>&1")
    
        if not turn_run:
            os.system("rm e3sm.exe")
        logger.info(f'Finish SCM run {id}')              

    def tune_scm(self,x):
        x = np.ravel(x)
        
        for i in range(len(x)):
            if x[i] < self.lbs[i] or x[i] > self.ubs[i]:
                return 1e6
            
        self.run_scm(self.tune_id,x,turn_run=True)
        mm = SCMmetrics()
        mvalue = mm.get_ARM97()

        self.tune_id += 1
        print(x,mvalue)
        return mvalue


    def archive_model(self,id):
        #os.system('./case.st_archive >& /dev/null')
        #os.system('zppy -c post.v3.LR.F2010.cfg >& /dev/null')

        base_path = '/global/homes/z/zhangtao/cfs_e3sm/v3.HR.STE.tune'
        target_path = f'{base_path}/T{id:03d}'

        os.system(f'mkdir -p {target_path}')
        #os.system(f'mv ../diag/post {target_path} ')
        os.system(f'cp user_nl_eam ../run/atm_in {target_path}')
        os.system(f'cp ../run/v3.HR.F2010.STE.tune.eam.h0.2013-0* {target_path}')
        os.system(f'cp ../run/v3.HR.F2010.STE.tune.eam.h3.2013-0* {target_path}')


    def analyse(self, method):
        if method == 'sample':
            #pool = mp.Pool(1)
            #pool.starmap(self.run_case, [(i+1,d) for i,d in enumerate(self.sample_scaled)])
            for i,data in enumerate(self.sample_scaled):
                self.run_case(i,data)

        if method == 'scm_sample':
            pool = mp.Pool(30)
            pool.starmap(self.run_scm, [(i,d) for i,d in enumerate(self.sample_scaled)])
            #for i,data in enumerate(self.sample_scaled):
            #    self.run_scm(i,data)
            #    if i >= 3:
            #        sys.exit()
        
        if method == 'nelder-mead':
            # init gauss
            m0 = metrics.metrics('/global/homes/z/zhangtao/cfs_e3sm/v3.HR.STE.tune/T058')
            init_gauss = np.array(list(m0.get_parameters(self.para_names).values()))
            self.tune_id = 68
            
            def tune_run(para):
                #boundary check
                penalty = 0
                for i,p in enumerate(para):
                    if p < self.lbs[i]:
                        penalty = 100
                    elif p > self.ubs[i]:
                        penalty = 100
                
                if penalty == 0:
                    self.run_case(self.tune_id,para,batch_run=True)
                    mm = metrics.metrics(f'/global/homes/z/zhangtao/cfs_e3sm/v3.HR.STE.tune/T{self.tune_id:03d}')
                    m_value = mm.calc_metrics().values
                    print(f'm value = {m_value:.3f}')
                    self.tune_id += 1
                else:
                    m_value = penalty
                    print('The next parameter is out of the bounds!!!')
         
                return m_value

            init = pd.read_csv('init_downhill.txt')
            custom_simplex = init.iloc[:,:-1].values
            init_y = init.iloc[:,-1].values
            result = minimize_neldermead(tune_run,x0=init_gauss,initial_simplex=custom_simplex,initial_y=init_y,options={'maxiter': 12})
            #result = minimize(tune_run, x0=init_gauss,method='Nelder-Mead', options={'maxiter': 100})
            #print(result)
            # archieve result 
            with open('NM.pkl','wb') as pfile:
                pickle.dump(result.final_simplex,pfile)
                
        if method == 'BO':
            self.tune_id = 11
            
            def tune_run(clubb_c8, zmconv_tau, nucleate_ice_subgrid, zmconv_ke, p3_qc_accret_expon, p3_autocon_coeff, p3_wbf_coeff):
                para = [clubb_c8, zmconv_tau, nucleate_ice_subgrid, zmconv_ke, p3_qc_accret_expon, p3_autocon_coeff, p3_wbf_coeff]
                self.run_case(self.tune_id,para)
                mm = metrics.metrics(f'/global/homes/z/zhangtao/cfs_e3sm/v3.HR.STE.tune/T{self.tune_id:03d}')
                m_value = mm.calc_metrics().values
                self.tune_id += 1
                
                return m_value
            
            bo_pbounds = {}
            for i,p in enumerate(self.para_names):
                bo_pbounds[p] = (self.lbs[i],self.ubs[i])
                
            new_optimizer = BayesianOptimization(
                f=tune_run,
                pbounds=bo_pbounds,
                verbose=2,
                random_state=7,
            )
            
            load_logs(new_optimizer, logs=["./init.txt"])
            new_optimizer.maximize(init_points=0, n_iter=200)

        if method == 'TurBO':
            self.continue_id = 0
            self.tune_id = 0

            f = turboProb(self)

            turbo1 = Turbo1(
                f=f,  # Handle to objective function
                lb=f.lb,  # Numpy array specifying lower bounds
                ub=f.ub,  # Numpy array specifying upper bounds
                n_init=50,  # Number of initial bounds from an Latin hypercube design
                max_evals = 200,  # Maximum number of evaluations
                batch_size=10,  # How large batch size TuRBO uses
                verbose=True,  # Print information from each batch
                use_ard=True,  # Set to true if you want to use ARD for the GP kernel
                max_cholesky_size=2000,  # When we switch from Cholesky to Lanczos
                n_training_steps=60,  # Number of steps of ADAM to learn the hypers
                min_cuda=1024,  # Run on the CPU for small datasets
                device="cpu",  # "cpu" or "cuda"
                dtype="float64",  # float64 or float32
            )

            turbo1.optimize()
            X = turbo1.X
            fX = turbo1.fX
            np.savez('turbo_result.npz', X=X, fX=fX)

                    
class turboProb:
    def __init__(self, fr):
        self.dim = fr.dim
        self.lb = np.array(fr.lbs)
        self.ub = np.array(fr.ubs)
        self.fr = fr

    def __call__(self, x):
        x = x.reshape(1,-1)
        val = self.fr.tune_scm(x)
        return val
            
            
            


