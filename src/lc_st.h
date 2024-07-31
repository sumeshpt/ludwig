/*****************************************************************************
 *
 *  lc_st.h
 *
 *  Routines related to liquid crystal with shape tensor free energy
 *  and molecular field.
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2012-2021 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  Sumesh Thampi (sumesh@iitm.ac.in)
 *
 *****************************************************************************/

#ifndef LUDWIG_LC_ST_H
#define LUDWIG_LC_ST_H

#include "coords.h"
#include "shape_tensor.h"
#include "blue_phase.h"
#include "field.h"
#include "field_grad.h"
#include "hydro.h"
#include "map.h"
#include "wall.h"

typedef struct fe_lc_st_s fe_lc_st_t;
typedef struct fe_lc_st_param_s fe_lc_st_param_t;

struct fe_lc_st_s {
  fe_t super;
  pe_t * pe;                      /* Parallel environment */
  cs_t * cs;                      /* Coordinate system */
  fe_lc_st_param_t * param;  /* Coupling parameters */
  fe_lc_t * lc;                   /* LC free energy  etc */
  fe_st_t * st;               /* Shape tensor free energy etc */
  fe_lc_st_t * target;       /* Target pointer */
};

struct fe_lc_st_param_s {
  double gamma0; /* \gamma(phi) = gamma0 + delta x (1 + phi) */
//  double delta;  /* As above */
//  double w;      /* Surface anchoring constant */
//  double zeta0 ; /* emulsion activity parameter */
//  double zeta1;  /* emulsion activity parameter */
};
__host__ int fe_lc_st_create(pe_t * pe, cs_t * cs, fe_lc_t * lc,
				  fe_st_t * st,
				  fe_lc_st_t ** p);
__host__ int fe_lc_st_free(fe_lc_st_t * fe);
__host__ int fe_lc_st_param(fe_lc_st_t * fe,
				 fe_lc_st_param_t * param);
__host__ int fe_lc_st_param_set(fe_lc_st_t * fe,
				     fe_lc_st_param_t param);
__host__ int fe_lc_st_target(fe_lc_st_t * fe, fe_t ** target);
/*
__host__ int fe_lc_droplet_bodyforce(fe_lc_droplet_t * fe, hydro_t * hydro);

__host__ __device__
int fe_lc_droplet_gamma(fe_lc_droplet_t * fe, int index,  double * gamma);
*/
__host__ __device__ int fe_lc_st_fed(fe_lc_st_t * fe, int index,
					  double * fed);
__host__ __device__ int fe_lc_st_stress(fe_lc_st_t * fe, int index,
					     double s[3][3]);
__host__ __device__ void fe_lc_st_stress_v(fe_lc_st_t * fe,
						int index,
						double s[3][3][NSIMDVL]);
__host__ __device__ int fe_lc_st_mol_field(fe_lc_st_t * fe,
						int index,
						double h[3][3]);
__host__ __device__ void fe_lc_st_mol_field_v(fe_lc_st_t * fe,
						   int index,
						   double h[3][3][NSIMDVL]);
/*
__host__ __device__ int fe_lc_droplet_mu(fe_lc_droplet_t * fe, int index,
					 double * mu);

*/
__host__ __device__
int fe_lc_st_str_symm(fe_lc_st_t * fe, int index, double s[3][3]);
__host__ __device__
int fe_lc_st_str_anti(fe_lc_st_t * fe, int index, double s[3][3]);
__host__ __device__
void fe_lc_st_str_symm_v(fe_lc_st_t * fe, int index, double s[3][3][NSIMDVL]);
__host__ __device__
void fe_lc_st_str_anti_v(fe_lc_st_t * fe, int index, double s[3][3][NSIMDVL]);
/*
__host__ int  fe_lc_droplet_bodyforce_wall(fe_lc_droplet_t * fe, lees_edw_t * le,
			      hydro_t * hydro, map_t * map, wall_t * wall);
*/
#endif
