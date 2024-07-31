/*****************************************************************************
 *
 *  shape_tensor_beris_edwards.h
 *
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *  (c) 2009-2020 The University of Edinburgh
 *
 *****************************************************************************/

#ifndef LUDWIG_SHAPE_TENSOR_BERIS_EDWARDS_H
#define LUDWIG_SHAPE_TENSOR_BERIS_EDWARDS_H

#include "coords.h"
#include "leesedwards.h"
#include "free_energy.h"
#include "hydro.h"
#include "field.h"
#include "map.h"
#include "colloids.h"
#include "noise.h"

typedef struct beris_edwst_s beris_edwst_t;
typedef struct beris_edwst_param_s beris_edwst_param_t;

struct beris_edwst_param_s {
  double xi;     /* Effective aspect ratio (from relevant free energy) */
  double Gamma;  /* Rotational diffusion constant */
  double var;    /* Noise variance */
//
//  double tmatrix[3][3][NQAB];  /* Constant noise tensor */
};

__host__ int beris_edwst_create(pe_t * pe, cs_t * cs, lees_edw_t * le,
			      beris_edwst_t ** pobj);
__host__ int beris_edwst_free(beris_edwst_t * be);
__host__ int beris_edwst_memcpy(beris_edwst_t * be, int flag);
__host__ int beris_edwst_param_set(beris_edwst_t * be, beris_edwst_param_t * values);
__host__ int beris_edwst_param_commit(beris_edwst_t * be);

__host__ int beris_edwst_update(beris_edwst_t * stbe, fe_t * fe, field_t * fr,
			      field_grad_t * fr_grad, hydro_t * hydro,
			      colloids_info_t * cinfo,
			      map_t * map, noise_t * noise);

//__host__ __device__ int beris_edw_tmatrix(double t[3][3][NQAB]);
//
#endif
