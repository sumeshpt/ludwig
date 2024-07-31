/*****************************************************************************
 *
 *  fe_st.h
 *
 *  Shape tensor free energy.
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2010-2023 The University of Edinburgh
 *
 *****************************************************************************/

#ifndef LUDWIG_FE_ST_H
#define LUDWIG_FE_ST_H

#include "pe.h"
#include "coords.h"
#include "leesedwards.h"
#include "free_energy.h"
#include "field.h"
#include "field_grad.h"
#include "io_harness.h"

//#include "lc_anchoring.h"

typedef struct fe_st_s fe_st_t;
typedef struct fe_st_param_s fe_st_param_t;

/* Free energy structure */

struct fe_st_s {
  fe_t super;
  pe_t * pe;                  /* Parallel environment */
  cs_t * cs;                  /* Coordinate system */
  fe_st_param_t * param;      /* Parameters */
  field_t * r;                /* R_ab (compresse rank 1 field) */
  field_grad_t * dr;          /* Gradients thereof */
  field_t * p;                /* Active term P_a = Q_ak d_m Q_mk */
  field_grad_t * dp;          /* Active term gradient d_a P_b */
  fe_st_t * target;           /* Device structure */
};

/* Shape tensor free energy parameters */

struct fe_st_param_s {
  double a0;
  double q0;
  double gamma;
  double kappa0;
  double kappa1;

  double xi;                              /* Flow aligning parameter */
  double zeta0;                           /* active stress term delta_ab */
  double zeta1;                           /* active stress term Q_ab */
  double zeta2;                           /* active stress d_a P_b + d_b P_a */
  double redshift;                        /* Redshift */
  double rredshift;                       /* Reciprocal redshift */
  double epsilon;                         /* Dielectric anistropy */
  double amplitude0;                      /* Initial amplitude from input */
  double e0[3];                           /* Electric field (external) */
  double coswt;                           /* Electric field (phase) */

  int is_redshift_updated;                /* Switch */
  int is_active;                          /* Switch for active fluid */

  //st_anchoring_param_t coll;              /* Anchoring parameters (colloids) */
  //st_anchoring_param_t wall;              /* Anchoring parameters (wall) */

  double k;
  double lambda0;
  double lambda[3];
  int nside;
  int twod;
};
__host__ int fe_st_create(pe_t * pe, cs_t * cs, lees_edw_t * le,
			  field_t * r, field_grad_t * dr, fe_st_t ** fe);
__host__ int fe_st_free(fe_st_t * fe);
__host__ int fe_st_param_set(fe_st_t * fe, const fe_st_param_t * values);
__host__ int fe_st_param_commit(fe_st_t * fe);
__host__ int fe_st_redshift_set(fe_st_t * fe,  double redshift);
__host__ int fe_st_redshift_compute(cs_t * cs, fe_st_t * fe);
__host__ int fe_st_target(fe_st_t * fe, fe_t ** target);
__host__ int fe_st_active_stress(fe_st_t * fe);
/* Host / device functions */
__host__ __device__
int fe_st_param(fe_st_t * fe, fe_st_param_t * vals);

__host__ __device__
int fe_st_fed(fe_st_t * fe, int index, double * fed);

__host__ __device__
int fe_st_mol_field(fe_st_t * fe, int index, double h[3][3]);

__host__ __device__
int fe_st_stress(fe_st_t * fe, int index, double s[3][3]);

__host__ __device__
int fe_st_bulk_stress(fe_st_t * fe, int index, double s[3][3]);

__host__ __device__
int fe_st_grad_stress(fe_st_t * fe, int index, double s[3][3]);

__host__ __device__
int fe_st_str_symm(fe_st_t * fe, int index, double s[3][3]);

__host__ __device__
int fe_st_str_anti(fe_st_t * fe, int index, double s[3][3]);

__host__ __device__
int fe_st_compute_fed(fe_st_t * fe, double r[3][3],
		      double dr[3][3][3], double * fed);

__host__ __device__
int fe_st_compute_bulk_fed(fe_st_t * fe, double r[3][3], double * fed);

__host__ __device__
int fe_st_compute_gradient_fed(fe_st_t * fe, double r[3][3],
		      double dr[3][3][3], double * fed);

__host__ __device__
int fe_st_compute_h(fe_st_t * fe, double r[3][3],
		    double dr[3][3][3],	double dsr[3][3], double h[3][3]);

__host__ __device__
int fe_st_compute_stress(fe_st_t * fe, double r[3][3], double dr[3][3][3],
			 double h[3][3], double sth[3][3]);
__host__ __device__
int fe_st_compute_stress_active(fe_st_t * fe, double r[3][3], double dp[3][3],
				double sa[3][3]);

__host__ __device__
int fe_st_chirality(fe_st_t * fe, double * chirality);

__host__ __device__
int fe_st_reduced_temperature(fe_st_t * fe,  double * tau);

__host__ __device__
void fe_st_mol_field_v(fe_st_t * fe, int index, double h[3][3][NSIMDVL]);

__host__ __device__
void fe_st_stress_v(fe_st_t * fe, int index, double s[3][3][NSIMDVL]);

__host__ __device__
void fe_st_str_symm_v(fe_st_t * fe, int index, double s[3][3][NSIMDVL]);

__host__ __device__
void fe_st_str_anti_v(fe_st_t * fe, int index, double s[3][3][NSIMDVL]);

__host__ __device__
void fe_st_compute_h_v(fe_st_t * fe,
		       double r[3][3][NSIMDVL], 
		       double dr[3][3][3][NSIMDVL],
		       double dsr[3][3][NSIMDVL], 
		       double h[3][3][NSIMDVL]);
__host__ __device__
void fe_st_compute_stress_v(fe_st_t * fe,
			    double r[3][3][NSIMDVL],
			    double dr[3][3][3][NSIMDVL],
			    double h[3][3][NSIMDVL],
			    double s[3][3][NSIMDVL]);

__host__ __device__
int fe_st_bulk_stress(fe_st_t * fe, int index, double sbulk[3][3]);

__host__ __device__
int fe_st_grad_stress(fe_st_t * fe, int index, double sgrad[3][3]);

/* Function of the parameters only */
__host__ int fe_st_dimensionless_field_strength(const fe_st_param_t * param,
						double * e0);

__host__ __device__
int fe_st_amplitude_compute(const fe_st_param_t * param, double * a);

/*
__host__ __device__
int fe_lc_q_uniaxial(fe_lc_param_t * param, const double n[3], double q[3][3]);
__host__ int fe_lc_scalar_ops(double q[3][3], double qs[NQAB]);
*/
__host__ __device__
int fe_st_r_isotropic(fe_st_param_t * param, double r[3][3]);

#endif
 
