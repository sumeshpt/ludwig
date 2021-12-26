/*****************************************************************************
 *
 *  lb_data.h
 *
 *  LB distribution data structure implementation.
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2014-2021 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifndef LB_DATA_H
#define LB_DATA_H

#include <stdint.h>

#include "pe.h"
#include "coords.h"
#include "lb_data_options.h"
#include "lb_model.h"

#include "io_harness.h"
#include "halo_swap.h"

#ifdef _D2Q9_
#include "d2q9.h"

enum {NDIM     = NDIM9,
      NVEL     = NVEL9,
      CVXBLOCK = CVXBLOCK9,
      CVYBLOCK = CVYBLOCK9,
      CVZBLOCK = CVZBLOCK9};
#endif

#ifdef _D3Q15_
#include "d3q15.h"

enum {NDIM     = NDIM15,
      NVEL     = NVEL15,
      CVXBLOCK = CVXBLOCK15,
      CVYBLOCK = CVYBLOCK15,
      CVZBLOCK = CVZBLOCK15};
#endif

#ifdef _D3Q19_

#include "d3q19.h"

enum {NDIM     = NDIM19,
      NVEL     = NVEL19,
      CVXBLOCK = CVXBLOCK19,
      CVYBLOCK = CVYBLOCK19,
      CVZBLOCK = CVZBLOCK19};
#endif

typedef struct lb_collide_param_s lb_collide_param_t;
typedef struct lb_halo_s lb_halo_t;
typedef struct lb_data_s lb_t;

struct lb_collide_param_s {
  int8_t isghost;                      /* switch for ghost modes */
  int8_t cv[NVEL][3];
  int nsite;
  int ndist;
  int nvel;
  double rho0;
  double eta_shear;
  double var_shear;
  double eta_bulk;
  double var_bulk;
  double rna[NVEL];                    /* reciprocal of normaliser[p] */
  double rtau[NVEL];
  double wv[NVEL];
  double ma[NVEL][NVEL];
  double mi[NVEL][NVEL];
};

struct lb_data_s {

  int ndim;
  int nvel;
  int ndist;             /* Number of distributions (default one) */
  int nsite;             /* Number of lattice sites (local) */

  pe_t * pe;             /* parallel environment */
  cs_t * cs;             /* coordinate system */

  lb_model_t model;      /* Current LB model information */
  halo_swap_t * halo;    /* halo swap driver */
  io_info_t * io_info;   /* Distributions */ 
  io_info_t * io_rho;    /* Fluid density (here; could be hydrodynamics...) */

  double * f;            /* Distributions */
  double * fprime;       /* used in propagation only */

  lb_collide_param_t * param;   /* Collision parameters REFACTOR THIS */
  lb_relaxation_enum_t nrelax;  /* Relaxation scheme */
  lb_halo_enum_t haloscheme;    /* halo scheme */

  /* MPI data types for halo swaps; these are comupted at runtime
   * to conform to the model selected at compile time */

  MPI_Datatype plane_xy_full;
  MPI_Datatype plane_xz_full;
  MPI_Datatype plane_yz_full;
  MPI_Datatype plane_xy_reduced[2];
  MPI_Datatype plane_xz_reduced[2];
  MPI_Datatype plane_yz_reduced[2];
  MPI_Datatype plane_xy[2];
  MPI_Datatype plane_xz[2];
  MPI_Datatype plane_yz[2];
  MPI_Datatype site_x[2];
  MPI_Datatype site_y[2];
  MPI_Datatype site_z[2];

  lb_t * target;              /* copy of this structure on target */ 
};

/* Data storage: A rank two object */

#include "memory.h"

#define LB_ADDR(nsites, ndist, nvel, index, n, p) \
  addr_rank2(nsites, ndist, nvel, index, n, p)

/* Number of hydrodynamic modes */
enum {NHYDRO = 1 + NDIM + NDIM*(NDIM+1)/2};

/* Labels to locate relaxation times in array[NVEL] */
/* Bulk viscosity is XX in stress */
/* Shear is XY in stress */

enum {LB_TAU_BULK = 1 + NDIM + XX, LB_TAU_SHEAR = 1 + NDIM + XY};

#define LB_CS2_DOUBLE(cs2)   const double cs2 = (1.0/3.0)
#define LB_RCS2_DOUBLE(rcs2) const double rcs2 = 3.0

typedef enum lb_dist_enum_type{LB_RHO = 0, LB_PHI = 1} lb_dist_enum_t;
typedef enum lb_mode_enum_type{LB_GHOST_ON = 0, LB_GHOST_OFF = 1} lb_mode_enum_t;

__host__ int lb_data_create(pe_t * pe, cs_t * cs,
			    const lb_data_options_t * opts, lb_t ** lb);
__host__ int lb_free(lb_t * lb);
__host__ int lb_memcpy(lb_t * lb, tdpMemcpyKind flag);
__host__ int lb_collide_param_commit(lb_t * lb);
__host__ int lb_halo(lb_t * lb);
__host__ int lb_halo_swap(lb_t * lb, lb_halo_enum_t flag);
__host__ int lb_halo_via_copy(lb_t * lb);
__host__ int lb_halo_via_struct(lb_t * lb);
__host__ int lb_halo_set(lb_t * lb, lb_halo_enum_t halo);
__host__ int lb_io_info(lb_t * lb, io_info_t ** io_info);
__host__ int lb_io_info_set(lb_t * lb, io_info_t * io_info, int fin, int fout);
__host__ int lb_io_rho_set(lb_t *lb, io_info_t * io_rho, int fin, int fout);

__host__ int lb_io_info_commit(lb_t * lb, io_info_args_t args);

__host__ __device__ int lb_ndist(lb_t * lb, int * ndist);
__host__ __device__ int lb_f(lb_t * lb, int index, int p, int n, double * f);
__host__ __device__ int lb_f_set(lb_t * lb, int index, int p, int n, double f);
__host__ __device__ int lb_0th_moment(lb_t * lb, int index, lb_dist_enum_t nd,
				      double * rho);
/* These  could be __host__ __device__ pending removal of
 * static constants */

__host__ int lb_init_rest_f(lb_t * lb, double rho0);
__host__ int lb_1st_moment(lb_t * lb, int index, lb_dist_enum_t nd, double g[3]);
__host__ int lb_2nd_moment(lb_t * lb, int index, lb_dist_enum_t nd, double s[3][3]);
__host__ int lb_0th_moment_equilib_set(lb_t * lb, int index, int n, double rho);
__host__ int lb_1st_moment_equilib_set(lb_t * lb, int index, double rho, double u[3]);

/* Halo */

#include "cs_limits.h"

struct lb_halo_s {

  MPI_Comm comm;                  /* coords: Cartesian communicator */
  int nbrrank[3][3][3];           /* coords: neighbour rank look-up */
  int nlocal[3];                  /* coords: local domain size */

  lb_model_t map;                 /* Communication map 2d or 3d */
  int tagbase;                    /* send/recv tag */
  int full;                       /* All velocities at each site required. */
  int count[27];                  /* halo: item data count per direction */
  cs_limits_t slim[27];           /* halo: send data region (rectangular) */
  cs_limits_t rlim[27];           /* halo: recv data region (rectangular) */
  double * send[27];              /* halo: send buffer per direction */
  double * recv[27];              /* halo: recv buffer per direction */
  MPI_Request request[2*27];      /* halo: array of requests */

};

int lb_halo_create(const lb_t * lb, lb_halo_t * h, int full);
int lb_halo_free(lb_t * lb, lb_halo_t * h);

#endif
