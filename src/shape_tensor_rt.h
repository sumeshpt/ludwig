/*****************************************************************************
 *
 *  shape_tensor_rt.h
 *
 *  $Id$
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2011-2017 The University of Edinbrugh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifndef LUDWIG_SHAPE_TENSOR_RT_H
#define LUDWIG_SHAPE_TENSOR_RT_H

#include "pe.h"
#include "coords.h"
#include "runtime.h"
#include "shape_tensor.h"
#include "shape_tensor_beris_edwards.h"

__host__ int shape_tensor_init_rt(pe_t * pe, rt_t * rt,
				 fe_st_t * fe,
				 beris_edwst_t * be);
__host__ int shape_tensor_rt_initial_conditions(pe_t * pe, rt_t * rt, cs_t * cs,
					      fe_st_t * fe, field_t * r);

#endif
