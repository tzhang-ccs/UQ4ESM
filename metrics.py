import xarray as xr
import numpy as np
import os
import sys

class SCMmetrics:
    def __init__(self):
        self.base_path = "/pscratch/sd/z/zhangtao/E3SMv3/SCM_runs"
        self.PQIs_mod = ['PRECT']
        self.PQIs_obs = ['Prec']

    def get_ARM95(self):
        fid_mod = xr.open_dataset(f"{self.base_path}/e3sm_scm_ARM95/T000/e3sm_scm_ARM95.eam.h0.1995-07-18-19800.nc")
        fid_cnl = xr.open_dataset(f"{self.base_path}/e3sm_scm_ARM95/run/e3sm_scm_ARM95.eam.h0.1995-07-18-19800.nc")
        fid_obs = xr.open_dataset("/global/cfs/cdirs/e3sm/inputdata/atm/cam/scam/iop/ARM95_iopfile_4scam.nc")
        
        kai = 0
        for i,p in enumerate(self.PQIs_mod):
            data_mod = fid_mod[p][::2][:400].values
            data_cnl = fid_cnl[p][::2][:400].values
            data_obs = fid_obs[self.PQIs_obs[i]][::3][:400,0].values

            rmse_mod = np.sqrt(np.mean((data_mod - data_obs) ** 2))
            rmse_cnl = np.sqrt(np.mean((data_cnl - data_obs) ** 2))
            kai += rmse_mod / rmse_cnl
            kai = kai / len(self.PQIs_mod)

        
        return kai
    
    def get_ARM97(self):
        fid_mod = xr.open_dataset(f"{self.base_path}/e3sm_scm_ARM97/T000/e3sm_scm_ARM97.eam.h0.1997-06-19-84585.nc")
        fid_cnl = xr.open_dataset(f"{self.base_path}/e3sm_scm_ARM97/run/e3sm_scm_ARM97.eam.h0.1997-06-19-84585.nc")
        fid_obs = xr.open_dataset("/global/cfs/cdirs/e3sm/inputdata/atm/cam/scam/iop/ARM97_iopfile_4scam.nc")
        
        kai = 0
        for i,p in enumerate(self.PQIs_mod):
            data_mod = fid_mod[p][::2][:600].values
            data_cnl = fid_cnl[p][::2][:600].values
            data_obs = fid_obs[self.PQIs_obs[i]][72:,0][::3][:600,0].values / 1000 # mm/s to m/s

            rmse_mod = np.sqrt(np.mean((data_mod - data_obs) ** 2))
            rmse_cnl = np.sqrt(np.mean((data_cnl - data_obs) ** 2))
            kai += rmse_mod / rmse_cnl

        return kai / len(self.PQIs_mod)





class metrics:
    def __init__(self,base_path):
        self.base_path = base_path
        self.dataset = {
            "LWCF":self.get_LWCF(),
            "SWCF":self.get_SWCF(),
            "FLUTC":self.get_FLUTC(),
            "T850": self.get_T850(),
            "Q850": self.get_Q850(),
            "PRECT": self.get_PRECT(),
            "RESTOM": self.get_RESTOM()
        }
        
    def get_parameters(self,names):
        parameters = {}
        for p in names:
            v = os.popen(f'grep -w {p} {self.base_path}/atm_in').read().split('=')[1]
            v = float(v)
            parameters[p] = v
            
        return parameters
        
    def calc_metrics(self,diag=False):
        kai = 0
        cts = np.zeros(2)
        
        if diag == True:
            diag_dict = {}
        
        for ii,mon in enumerate(['07']):
            #### LWCF, SWCF, FLUTC
            for var in ['LWCF','SWCF','FLUTC','T850','Q850','PRECT']:
                mod = self.dataset[var][f'mod_{mon}'].mean(dim='lon')[0,:]
                cntl = self.dataset[var][f'cntl_{mon}'].mean(dim='lon')[0,:]
                obs = self.dataset[var][f'obs_{mon}'].mean(dim='lon')
                
                if var == 'FLUTC' or var == 'T850':
                    mod = mod.sel(lat=slice(-60,90))
                    cntl = cntl.sel(lat=slice(-60,90))
                    obs = obs.sel(lat=slice(-60,90))
                    
                if var == 'PRECT' and mon == '01':
                    continue
                
                mse_mod = 0
                mse_cntl = 0
                for i in range(len(obs)):
                    mse_mod += self.weight[i]*(mod[i] - obs[i]) ** 2
                    mse_cntl += self.weight[i]*(cntl[i] - obs[i]) ** 2
                    
                kai += mse_mod/mse_cntl
                
                if diag == True:
                    diag_dict[f'{var}_{mon}'] = mse_mod/mse_cntl
        
            #RESTOM constrait
            cts[ii] = self.dataset['RESTOM'][f'mod_{mon}'].weighted(self.weight).mean(('lat','lon'))
        
        if diag == True:
            return kai/6, diag_dict
    
        return kai/6

    def get_LWCF(self):
        case_name = ['OBS','mod','cntl']
        dataset = {}

        ########### data
        for c in case_name:
            #for mon in ['01','07']:
            for mon in ['07']:
                if c == 'OBS':
                    #base_path = '/global/cfs/cdirs/e3sm/diagnostics/observations/Atm/climatology/ceres_ebaf_toa_v4.1/'
                    base_path = '/global/cfs/cdirs/e3sm/wlin/data/obs_for_e3sm_diags/climatology/ceres_ebaf_toa_v4.1/'
                    #data_path = f'{base_path}/ceres_ebaf_toa_v4.1_{mon}_2001{mon}_2018{mon}_climo.nc'
                    data_path = f'{base_path}/ceres_ebaf_toa_v4.1_2013{mon}.nc'
                    fid = xr.open_dataset(data_path)
                    data =fid['rlutcs'][0,...] - fid['rlut'][0,...]
                    dataset[f'obs_{mon}'] = data

                elif c == 'mod':
                    base_path = '/pscratch/sd/z/zhangtao/E3SMv3/HR/v3.HR.F2010.STE.tune/run'
                    data_path = f'{self.base_path}/v3.HR.F2010.STE.tune.eam.h0.2013-{mon}.latxlon.nc'
                    fid = xr.open_dataset(data_path)
                    lat = fid['lat']
                    lon = fid['lon']
                    data = fid['LWCF']
                    dataset[f'mod_{mon}'] = data

                else:
                    base_path = '/pscratch/sd/z/zhangtao/E3SMv3/HR/v3.HR.F2010.STE.tune/run/CNTL'
                    data_path = f'{base_path}/v3.HR.F2010.STE.tune.eam.h0.2013-{mon}.latxlon.nc'
                    fid = xr.open_dataset(data_path)
                    lat = fid['lat']
                    lon = fid['lon']
                    self.weight = np.cos(np.deg2rad(lat))
                    data = fid['LWCF']
                    dataset[f'cntl_{mon}'] = data
                    
        return dataset

    def get_SWCF(self):
        case_name = ['OBS','mod','cntl']
        dataset = {}

        ########### data
        for c in case_name:
            #for mon in ['01','07']:
            for mon in ['07']:
                if c == 'OBS':
                    #base_path = '/global/cfs/cdirs/e3sm/diagnostics/observations/Atm/climatology/ceres_ebaf_toa_v4.1/'
                    base_path = '/global/cfs/cdirs/e3sm/wlin/data/obs_for_e3sm_diags/climatology/ceres_ebaf_toa_v4.1/'
                    #data_path = f'{base_path}/ceres_ebaf_toa_v4.1_{mon}_2001{mon}_2018{mon}_climo.nc'
                    data_path = f'{base_path}/ceres_ebaf_toa_v4.1_2013{mon}.nc'
                    fid = xr.open_dataset(data_path)
                    data =fid['rsutcs'][0,...] - fid['rsut'][0,...]
                    dataset[f'obs_{mon}'] = data

                elif c == 'mod':
                    base_path = '/pscratch/sd/z/zhangtao/E3SMv3/HR/v3.HR.F2010.STE.tune/run'
                    data_path = f'{self.base_path}/v3.HR.F2010.STE.tune.eam.h0.2013-{mon}.latxlon.nc'
                    fid = xr.open_dataset(data_path)
                    lat = fid['lat']
                    lon = fid['lon']
                    data = fid['SWCF']
                    dataset[f'mod_{mon}'] = data

                else:
                    base_path = '/pscratch/sd/z/zhangtao/E3SMv3/HR/v3.HR.F2010.STE.tune/run/CNTL'
                    data_path = f'{base_path}/v3.HR.F2010.STE.tune.eam.h0.2013-{mon}.latxlon.nc'
                    fid = xr.open_dataset(data_path)
                    lat = fid['lat']
                    lon = fid['lon']
                    data = fid['SWCF']
                    dataset[f'cntl_{mon}'] = data
                
        return dataset


    def get_PRECT(self):
        case_name = ['mod','OBS','cntl']
        dataset = {}
        
        ########### data 
        for c in case_name:
            for mon in ['07']:
            #for mon in ['01','07']:
                if c == 'OBS':
                    #base_path = '/global/cfs/cdirs/e3sm/diagnostics/observations/Atm/climatology/GPCP_v3.2/'
                    base_path = '/global/cfs/cdirs/e3sm/wlin/data/obs_for_e3sm_diags/climatology/GPCP_v3.2/'
                    #data_path = f'{base_path}/GPCP_v3.2_{mon}_1983{mon}_2021{mon}_climo.nc'
                    data_path = f'{base_path}/sat_gauge_precip_2013{mon}.nc'
                    fid = xr.open_dataset(data_path)
                    data = fid['sat_gauge_precip'][0,...].interp(lat=lat,lon=lon,method="nearest")
                    dataset[f'obs_{mon}'] = data 
                elif c == 'mod':
                    base_path = '/pscratch/sd/z/zhangtao/E3SMv3/HR/v3.HR.F2010.STE.tune/run'
                    data_path = f'{self.base_path}/v3.HR.F2010.STE.tune.eam.h0.2013-{mon}.latxlon.nc'
                    fid = xr.open_dataset(data_path)
                    lat = fid['lat']
                    lon = fid['lon']
                    data = fid['PRECT'] * 86400 * 1000
                    dataset[f'mod_{mon}'] = data
                
                else:
                    base_path = '/pscratch/sd/z/zhangtao/E3SMv3/HR/v3.HR.F2010.STE.tune/run/CNTL'
                    data_path = f'{base_path}/v3.HR.F2010.STE.tune.eam.h0.2013-{mon}.latxlon.nc'
                    fid = xr.open_dataset(data_path)
                    lat = fid['lat']
                    lon = fid['lon']
                    data = fid['PRECT']* 86400 * 1000
                    dataset[f'cntl_{mon}'] = data
                    
        return dataset

    def get_T850(self):
        case_name = ['mod','cntl','OBS']
        dataset = {}
        
        ########### data 
        for c in case_name:
            for mon in ['07']:
            #for mon in ['01','07']:
                if c == 'OBS':
                    #base_path = '/global/cfs/cdirs/e3sm/diagnostics/observations/Atm/climatology/ERA5/'
                    base_path = '/global/cfs/cdirs/e3sm/wlin/data/obs_for_e3sm_diags/climatology/ERA5'
                    #data_path = f'{base_path}/ERA5_{mon}_1979{mon}_2019{mon}_climo.nc'
                    data_path = f'{base_path}/ERA5_2013{mon}.nc'
                    fid = xr.open_dataset(data_path)
                    data = fid['ta'][0,6,...].interp(lat=lat,lon=lon,method="nearest")
                    dataset[f'obs_{mon}'] = data 

                elif c == 'mod':
                    base_path = '/pscratch/sd/z/zhangtao/E3SMv3/HR/v3.HR.F2010.STE.tune/run'
                    data_path = f'{self.base_path}/v3.HR.F2010.STE.tune.eam.h0.2013-{mon}.latxlon.nc'
                    fid = xr.open_dataset(data_path)
                    lat = fid['lat']
                    lon = fid['lon']
                    data = fid['T850']
                    dataset[f'mod_{mon}'] = data

                else:
                    base_path = '/pscratch/sd/z/zhangtao/E3SMv3/HR/v3.HR.F2010.STE.tune/run/CNTL'
                    data_path = f'{base_path}/v3.HR.F2010.STE.tune.eam.h0.2013-{mon}.latxlon.nc'
                    fid = xr.open_dataset(data_path)
                    lat = fid['lat']
                    lon = fid['lon']
                    data = fid['T850']
                    dataset[f'cntl_{mon}'] = data
                
        return dataset
    

    def get_Q850(self):
        case_name = ['mod','cntl','OBS']
        dataset = {}
        
        ########### data 
        for c in case_name:
            for mon in ['07']:
            #for mon in ['01','07']:
                if c == 'OBS':
                    #base_path = '/global/cfs/cdirs/e3sm/diagnostics/observations/Atm/climatology/ERA5/'
                    base_path = '/global/cfs/cdirs/e3sm/wlin/data/obs_for_e3sm_diags/climatology/ERA5/'
                    #data_path = f'{base_path}/ERA5_{mon}_1979{mon}_2019{mon}_climo.nc'
                    data_path = f'{base_path}/ERA5_2013{mon}.nc'
                    fid = xr.open_dataset(data_path)
                    data = fid['hus'][0,6,...].interp(lat=lat,lon=lon,method="nearest")
                    dataset[f'obs_{mon}'] = data 
                
                elif c == 'mod':
                    base_path = '/pscratch/sd/z/zhangtao/E3SMv3/HR/v3.HR.F2010.STE.tune/run'
                    data_path = f'{self.base_path}/v3.HR.F2010.STE.tune.eam.h0.2013-{mon}.latxlon.nc'
                    fid = xr.open_dataset(data_path)
                    lat = fid['lat']
                    lon = fid['lon']
                    data = fid['Q850']
                    dataset[f'mod_{mon}'] = data
                
                else:
                    base_path = '/pscratch/sd/z/zhangtao/E3SMv3/HR/v3.HR.F2010.STE.tune/run/CNTL'
                    data_path = f'{base_path}/v3.HR.F2010.STE.tune.eam.h0.2013-{mon}.latxlon.nc'
                    fid = xr.open_dataset(data_path)
                    lat = fid['lat']
                    lon = fid['lon']
                    data = fid['Q850']
                    dataset[f'cntl_{mon}'] = data
                
        return dataset

    def get_FLUTC(self):
        case_name = ['OBS','mod','cntl']
        dataset = {}

        ########### data 
        for c in case_name:
            for mon in ['07']:
            #for mon in ['01','07']:
                if c == 'OBS':
                    base_path = '/global/cfs/cdirs/e3sm/wlin/data/obs_for_e3sm_diags/climatology/ceres_ebaf_toa_v4.1/'
                    #base_path = '/global/cfs/cdirs/e3sm/diagnostics/observations/Atm/climatology/ceres_ebaf_toa_v4.1/'
                    #data_path = f'{base_path}/ceres_ebaf_toa_v4.1_{mon}_2001{mon}_2018{mon}_climo.nc'
                    data_path = f'{base_path}/ceres_ebaf_toa_v4.1_2013{mon}.nc'
                    fid = xr.open_dataset(data_path)
                    data = fid['rlutcs'][0,...]
                    dataset[f'obs_{mon}'] = data 
                    
                elif c == 'mod':
                    base_path = '/pscratch/sd/z/zhangtao/E3SMv3/HR/v3.HR.F2010.STE.tune/run'
                    data_path = f'{self.base_path}/v3.HR.F2010.STE.tune.eam.h0.2013-{mon}.latxlon.nc'
                    fid = xr.open_dataset(data_path)
                    lat = fid['lat']
                    lon = fid['lon']
                    data = fid['FLUTC']
                    dataset[f'mod_{mon}'] = data

                else:
                    base_path = '/pscratch/sd/z/zhangtao/E3SMv3/HR/v3.HR.F2010.STE.tune/run/CNTL'
                    data_path = f'{base_path}/v3.HR.F2010.STE.tune.eam.h0.2013-{mon}.latxlon.nc'
                    fid = xr.open_dataset(data_path)
                    lat = fid['lat']
                    lon = fid['lon']
                    data = fid['FLUTC']
                    dataset[f'cntl_{mon}'] = data

        return dataset
    
    def get_RESTOM(self):
        case_name = ['mod','cntl']
        dataset = {}

        ########### data 
        for c in case_name:
            for mon in ['07']:
            #for mon in ['01','07']:
                if c == 'mod':
                    base_path = '/pscratch/sd/z/zhangtao/E3SMv3/HR/v3.HR.F2010.STE.tune/run'
                    data_path = f'{self.base_path}/v3.HR.F2010.STE.tune.eam.h0.2013-{mon}.latxlon.nc'
                    fid = xr.open_dataset(data_path)
                    lat = fid['lat']
                    lon = fid['lon']
                    data = fid['FSNT'] - fid['FLNT']
                    dataset[f'mod_{mon}'] = data

                else:
                    base_path = '/pscratch/sd/z/zhangtao/E3SMv3/HR/v3.HR.F2010.STE.tune/run/CNTL'
                    data_path = f'{base_path}/v3.HR.F2010.STE.tune.eam.h0.2013-{mon}.latxlon.nc'
                    fid = xr.open_dataset(data_path)
                    lat = fid['lat']
                    lon = fid['lon']
                    data = fid['FSNT'] - fid['FLNT']
                    dataset[f'cntl_{mon}'] = data

        return dataset