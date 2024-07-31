/****************************************************************************
 *
 *  lc_st_rt.h
 *
 *  Run time initiliasation for the liquid crystal-shape tensor free energy
 *
 *  Edinburgh Soft Matter and Statistical Physics Group
 *  and Edinburgh Parallel Computing Centre
 *
 *  (c) 2012-2016 The University of Edinburgh
 *
 ****************************************************************************/

#ifndef LC_ST_RT_H
#define LC_ST_RT_H

#include "pe.h"
#include "runtime.h"
#include "lc_st.h"

int fe_lc_st_run_time(pe_t * pe, rt_t * rt, fe_lc_st_t * fe);

#endif
