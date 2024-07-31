/****************************************************************************
 *
 *  lc_st_rt.c
 *
 *  Run time initiliasation for the liquid crystal shape tensor free energy
 *
 *  Edinburgh Soft Matter and Statistical Physics Group
 *  and Edinburgh Parallel Computing Centre
 *
 *  (c) 2012-2022 The University of Edinburgh
 *
 ****************************************************************************/

#include <assert.h>

#include "lc_st_rt.h"

/*****************************************************************************
 *
 *  fe_lc_st_run_time
 *
 *  Pick up the liquid crystal shape tensor specific  parameters from the input.
 *
 *  In addition to this also routines shape_tensor_runtime and 
 *  blue_phase_runtime are called.
 *
 *****************************************************************************/

int fe_lc_st_run_time(pe_t * pe, rt_t * rt, fe_lc_st_t * fe) {

//  int n;
  fe_lc_st_param_t param = {0};

  assert(fe);

  pe_info(pe, "\n");
  pe_info(pe, "Liquid crystal shape tensor coupling parameters\n");

//  n = rt_double_parameter(rt, "lc_st_gamma", &param.gamma0);
//  if (n == 0) pe_fatal(pe, "Please specify lc_st_gamma in input\n");

//  n = rt_double_parameter(rt, "lc_droplet_delta", &param.delta);
//  if (n == 0) pe_fatal(pe, "Please specify lc_droplet_delta in input\n");

//  n = rt_double_parameter(rt, "lc_droplet_W", &param.w);
//  if (n == 0) pe_fatal(pe, "Please specify lc_droplet_W in input\n");

//  pe_info(pe, "Isotropic/LC control gamma0 = %12.5e\n", param.gamma0);
//  pe_info(pe, "Isotropic/LC control delta  = %12.5e\n", param.delta);
//  pe_info(pe, "Anchoring parameter  W      = %12.5e\n", param.w);

  {
    /* Optional activity parameters */
//    int nz0 = 0;
//    int nz1 = 0;

//    nz0 = rt_double_parameter(rt, "lc_droplet_active_zeta0", &param.zeta0);
//    nz1 = rt_double_parameter(rt, "lc_droplet_active_zeta1", &param.zeta1);

//    if (nz0 || nz1) {
      /* Report */
//      pe_info(pe, "Emulsion activity: zeta0    = %12.5e\n", param.zeta0);
//      pe_info(pe, "Emulsion activity: zeta1    = %12.5e\n", param.zeta1);
//    }
  }
  
  fe_lc_st_param_set(fe, param);

  return 0;
}
  
  
