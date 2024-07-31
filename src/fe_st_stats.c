/****************************************************************************
 *
 *  fe_st_stats.c
 *
 *  Statistics for shape tensor free energy including surface
 *  free energy terms.
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2017-2023 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 ****************************************************************************/

#include <assert.h>
#include <math.h>

#include "fe_st_stats.h"
#include "util.h"


static int fe_st_wall(cs_t * cs, wall_t * wall, fe_st_t * fe, double * fs);
static int fe_st_colloid(fe_st_t * fe, cs_t * cs, colloids_info_t * cinfo,
			 map_t * map, double * fs);

/* additional forward declarations */

int fe_st_wallx(cs_t * cs, fe_st_t * fe, double * fs);
int fe_st_wally(cs_t * cs, fe_st_t * fe, double * fs);
int fe_st_wallz(cs_t * cs, fe_st_t * fe, double * fs);

int colloids_q_boundary(fe_st_param_t * param,
			const double nhat[3], double rs[3][3],
			  double r0[3][3], int map_status);

__host__ int shape_tensor_fs(fe_st_param_t * feparam, const double dn[3],
			   double rs[3][3], char status, double * fs);

static int fe_st_bulk_grad(fe_st_t * fe, cs_t * cs, map_t * map, double * fbg);

__host__ int shape_tensor_fbg(fe_st_param_t * feparam, double r[3][3], 
			   double dr[3][3][3], double * fbg);


#define NFE_STAT 5

/****************************************************************************
 *
 *  fe_st_stats_info
 *
 ****************************************************************************/

int fe_st_stats_info(pe_t * pe, cs_t * cs, fe_st_t * fe,
		     wall_t * wall, map_t * map,
		     colloids_info_t * cinfo, int step) {

  int ic, jc, kc, index;
  int nlocal[3];
  int status;
  int ncolloid;

  double fed;
  double fe_local[NFE_STAT];
  double fe_total[NFE_STAT];

  MPI_Comm comm;

  assert(pe);
  assert(cs);
  assert(fe);
  assert(map);

  pe_mpi_comm(pe, &comm);
  cs_nlocal(cs, nlocal);
  colloids_info_ntotal(cinfo, &ncolloid);

  fe_local[0] = 0.0; /* Total free energy (fluid all sites) */
  fe_local[1] = 0.0; /* Fluid only free energy */
  fe_local[2] = 0.0; /* Volume of fluid */
  fe_local[3] = 0.0; /* surface free energy */
  fe_local[4] = 0.0; /* other wall free energy (walls only) */

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = cs_index(cs, ic, jc, kc);
	map_status(map, index, &status);

	fe_st_fed(fe, index, &fed);
	fe_local[0] += fed;

	if (status == MAP_FLUID) {
	    fe_local[1] += fed;
	    fe_local[2] += 1.0;
	}
      }
    }
  }

  /* I M P O R T A N T */
  /* The regression test output is sensitive to the form of
   * this output. If you change this, you need to update
   * all the test logs when satisfied on correctness. */

  if (wall_present(wall)) {

    fe_st_wall(cs, wall, fe, fe_local + 3);

    MPI_Reduce(fe_local, fe_total, NFE_STAT, MPI_DOUBLE, MPI_SUM, 0, comm);

    pe_info(pe, "\nFree energies - timestep f v f/v f_s1 fs_s2 redshift\n");
    pe_info(pe, "[fe] %14d %17.10e %17.10e %17.10e %17.10e %17.10e %17.10e\n",
	    step, fe_total[1], fe_total[2], fe_total[1]/fe_total[2],
	    fe_total[3], fe_total[4], fe->param->redshift);
  }
  else if (ncolloid > 0) {

    fe_st_colloid(fe, cs, cinfo, map, fe_local + 3);

    MPI_Reduce(fe_local, fe_total, NFE_STAT, MPI_DOUBLE, MPI_SUM, 0, comm);

    pe_info(pe, "\nFree energies - timestep f v f/v f_s a f_s/a\n");

    if (fe_total[4] > 0.0) {
      /* Area > 0 means the free energy is available */
      pe_info(pe, "[fe] %14d %17.10e %17.10e %17.10e %17.10e %17.10e %17.10e\n",
	      step, fe_total[1], fe_total[2], fe_total[1]/fe_total[2],
	      fe_total[3], fe_total[4], fe_total[3]/fe_total[4]);
    }
    else {
      pe_info(pe, "[fe] %14d %17.10e %17.10e %17.10e %17.10e\n",
	      step, fe_total[1], fe_total[2], fe_total[1]/fe_total[2],
	      fe_total[3]);
    }
  }
  else {

    fe_st_bulk_grad(fe, cs, map, fe_local + 3);

    MPI_Reduce(fe_local, fe_total, NFE_STAT, MPI_DOUBLE, MPI_SUM, 0, comm);

    pe_info(pe, "\nFree energies - timestep f v f/v f_bulk/v f_grad/v redshift\n");
    pe_info(pe, "[fe] %14d %17.10e %17.10e %17.10e %17.10e %17.10e %17.10e\n", step, 
	fe_total[1], fe_total[2], fe_total[1]/fe_total[2], fe_total[3]/fe_total[2], fe_total[4]/fe_total[2], fe->param->redshift);
  }

  return 0;
}

/*****************************************************************************
 *
 *  fe_st_wall
 *
 *  Return f_s for bottom wall and top wall (and could add an area).
 *
 *****************************************************************************/

static int fe_st_wall(cs_t * cs, wall_t * wall, fe_st_t * fe, double * fs) {

  int iswall[3];

  assert(cs);
  assert(wall);
  assert(fe);

  wall_present_dim(wall, iswall);

  if (iswall[X]) fe_st_wallx(cs, fe, fs);
  if (iswall[Y]) fe_st_wally(cs, fe, fs);
  if (iswall[Z]) fe_st_wallz(cs, fe, fs);

  return 0;
}

/*****************************************************************************
 *
 *  fe_st_wallx
 *
 *  Return f_s for bottom wall and top wall (and could add an area).
 *
 *****************************************************************************/

int fe_st_wallx(cs_t * cs, fe_st_t * fe, double * fs) {

  int ic, jc, kc, index;
  int nlocal[3];
  int mpisz[3];
  int mpicoords[3];

  double dn[3];
  double rs[3][3];
  double fes;

  assert(cs);
  assert(fe);
  assert(fs);

  fs[0] = 0.0;
  fs[1] = 0.0;

  cs_nlocal(cs, nlocal);
  cs_cartsz(cs, mpisz);
  cs_cart_coords(cs, mpicoords);

  dn[Y] = 0.0;
  dn[Z] = 0.0;

  if (mpicoords[X] == 0) {

    ic = 1;
    dn[X] = +1.0;

    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

        index = cs_index(cs, ic, jc, kc);
	field_tensor(fe->r, index, rs);
	shape_tensor_fs(fe->param, dn, rs, MAP_BOUNDARY, &fes);
	fs[0] += fes;
      }
    }
  }

  if (mpicoords[X] == mpisz[X] - 1) {

    ic = nlocal[X];
    dn[X] = -1;

    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

        index = cs_index(cs, ic, jc, kc);
	field_tensor(fe->r, index, rs);
	shape_tensor_fs(fe->param, dn, rs, MAP_BOUNDARY, &fes);
	fs[1] += fes;
      }
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  fe_st_wally
 *
 *  Return f_s for bottom wall and top wall (and could add an area).
 *
 *****************************************************************************/

int fe_st_wally(cs_t * cs, fe_st_t * fe, double * fs) {

  int ic, jc, kc, index;
  int nlocal[3];
  int mpisz[3];
  int mpicoords[3];

  double dn[3];
  double rs[3][3];
  double fes;

  assert(cs);
  assert(fe);
  assert(fs);

  fs[0] = 0.0;
  fs[1] = 0.0;

  cs_nlocal(cs, nlocal);
  cs_cartsz(cs, mpisz);
  cs_cart_coords(cs, mpicoords);

  dn[X] = 0.0;
  dn[Z] = 0.0;

  if (mpicoords[Y] == 0) {

    jc = 1;
    dn[Y] = +1.0;

    for (ic = 1; ic <= nlocal[X]; ic++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

        index = cs_index(cs, ic, jc, kc);
	field_tensor(fe->r, index, rs);
	shape_tensor_fs(fe->param, dn, rs, MAP_BOUNDARY, &fes);
	fs[0] += fes;
      }
    }
  }

  if (mpicoords[Y] == mpisz[Y] - 1) {

    jc = nlocal[Y];
    dn[Y] = -1;

    for (ic = 1; ic <= nlocal[X]; ic++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

        index = cs_index(cs, ic, jc, kc);
	field_tensor(fe->r, index, rs);
	shape_tensor_fs(fe->param, dn, rs, MAP_BOUNDARY, &fes);
	fs[1] += fes;
      }
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  fe_st_wallz
 *
 *  Return f_s for bottom wall and top wall (and could add an area).
 *
 *****************************************************************************/

int fe_st_wallz(cs_t * cs, fe_st_t * fe, double * fs) {

  int ic, jc, kc, index;
  int nlocal[3];
  int mpisz[3];
  int mpicoords[3];

  double dn[3];
  double rs[3][3];
  double fes;

  fs[0] = 0.0;
  fs[1] = 0.0;

  assert(cs);
  assert(fe);
  assert(fs);

  cs_nlocal(cs, nlocal);
  cs_cartsz(cs, mpisz);
  cs_cart_coords(cs, mpicoords);

  dn[X] = 0.0;
  dn[Y] = 0.0;

  if (mpicoords[Z] == 0) {

    kc = 1;
    dn[Z] = +1.0;

    for (ic = 1; ic <= nlocal[X]; ic++) {
      for (jc = 1; jc <= nlocal[Y]; jc++) {

        index = cs_index(cs, ic, jc, kc);
	field_tensor(fe->r, index, rs);
	shape_tensor_fs(fe->param, dn, rs, MAP_BOUNDARY, &fes);
	fs[0] += fes;
      }
    }
  }

  if (mpicoords[Z] == mpisz[Z] - 1) {

    kc = nlocal[Z];
    dn[Z] = -1;

    for (ic = 1; ic <= nlocal[X]; ic++) {
      for (jc = 1; jc <= nlocal[Y]; jc++) {

        index = cs_index(cs, ic, jc, kc);
	field_tensor(fe->r, index, rs);
	shape_tensor_fs(fe->param, dn, rs, MAP_BOUNDARY, &fes);
	fs[1] += fes;
      }
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  colloids_s_boundary
 *
 *  Produce an estimate of the surface order parameter R_ab for
 *  normal or planar anchoring.
 *
 *  This will depend on the outward surface normal nhat, and in the
 *  case of planar anchoring may depend on the estimate of the
 *  existing order parameter at the surface Qs_ab.
 *
 *  This planar anchoring idea follows e.g., Fournier and Galatola
 *  Europhys. Lett. 72, 403 (2005).
 *
 *****************************************************************************/

int colloids_s_boundary(fe_st_param_t * param,
			const double nhat[3], double rs[3][3],
			double r0[3][3], int map_status) {
//  int ia, ib, ic, id;
//  int anchoring;
//
//  double qtilde[3][3];
//  double amp;
//  KRONECKER_DELTA_CHAR(d);
//
//  assert(map_status == MAP_COLLOID || map_status == MAP_BOUNDARY);
//
//  anchoring = param->coll.type;
//  if (map_status == MAP_BOUNDARY) anchoring = param->wall.type;
//
//  fe_lc_amplitude_compute(param, &amp);
//
//  if (anchoring == LC_ANCHORING_FIXED) {
//    assert(map_status == MAP_BOUNDARY);
//    for (ia = 0; ia < 3; ia++) {
//      double na = param->wall.nfix[ia];
//      for (ib = 0; ib < 3; ib++) {
//	double nb = param->wall.nfix[ib];
//	q0[ia][ib] = 0.5*amp*(3.0*na*nb - d[ia][ib]);
//      }
//    }
//  }
//
//  if (anchoring == LC_ANCHORING_NORMAL) {
//    for (ia = 0; ia < 3; ia++) {
//      for (ib = 0; ib < 3; ib++) {
//	q0[ia][ib] = 0.5*amp*(3.0*nhat[ia]*nhat[ib] - d[ia][ib]);
//      }
//    }
//  }
//
//  if (anchoring == LC_ANCHORING_PLANAR) {
//
//    /* Planar: use the fluid Q_ab to find ~Q_ab */
//
//    for (ia = 0; ia < 3; ia++) {
//      for (ib = 0; ib < 3; ib++) {
//	qtilde[ia][ib] = qs[ia][ib] + 0.5*amp*d[ia][ib];
//      }
//    }
//
//    for (ia = 0; ia < 3; ia++) {
//      for (ib = 0; ib < 3; ib++) {
//	q0[ia][ib] = 0.0;
//	for (ic = 0; ic < 3; ic++) {
//	  for (id = 0; id < 3; id++) {
//	    q0[ia][ib] += (d[ia][ic] - nhat[ia]*nhat[ic])*qtilde[ic][id]
//	      *(d[id][ib] - nhat[id]*nhat[ib]);
//	  }
//	}
//	/* Return Q^0_ab = ~Q_ab - (1/2) A d_ab */
//	q0[ia][ib] -= 0.5*amp*d[ia][ib];
//      }
//    }
//
//  }

  return 0;
}

/*****************************************************************************
 *
 *  fe_st_colloid
 *
 *  Return f_s for the local colloid surface (and an area).
 *
 *     fs[0] = free energy (integrated over area)
 *     fs[1] = discrete surface area
 *
 *****************************************************************************/

static int fe_st_colloid(fe_st_t * fe, cs_t * cs, colloids_info_t * cinfo,
			 map_t * map, double * fs) {

//  int ic, jc, kc, index, index1;
//  int nhat[3];
//  int nlocal[3];
//  int status;
//
//  double dn[3];
//  double qs[3][3];
//  double fes;
//
//  fs[0] = 0.0;
//  fs[1] = 0.0;
//
//  assert(fe);
//  assert(cs);
//  assert(map);
//  assert(fs);
//
//  cs_nlocal(cs, nlocal);
//
//  for (ic = 1; ic <= nlocal[X]; ic++) {
//    for (jc = 1; jc <= nlocal[Y]; jc++) {
//      for (kc = 1; kc <= nlocal[Z]; kc++) {
//
//        index = cs_index(cs, ic, jc, kc);
//	map_status(map, index, &status);
//	if (status != MAP_FLUID) continue;
//
//	/* This site is fluid. Look at six nearest neighbours... */
//	field_tensor(fe->q, index, qs);
//
//        nhat[Y] = 0;
//        nhat[Z] = 0;
//
//	/* Surface in direction of (ic+1,jc,kc) */
//	index1 = cs_index(cs, ic+1, jc, kc);
//	map_status(map, index1, &status);
//
//	if (status == MAP_COLLOID) {
//          nhat[X] = -1;
//          colloids_q_boundary_normal(cinfo, index, nhat, dn);
//	  blue_phase_fs(fe->param, dn, qs, MAP_COLLOID, &fes);
//	  fs[0] += fes;
//	  fs[1] += 1.0;
//        }
//
//	/* Surface in direction of (ic-1,jc,kc) */
//	index1 = cs_index(cs, ic-1, jc, kc);
//	map_status(map, index1, &status);
//
//        if (status == MAP_COLLOID) {
//          nhat[X] = +1;
//          colloids_q_boundary_normal(cinfo, index, nhat, dn);
//	  blue_phase_fs(fe->param, dn, qs, MAP_COLLOID, &fes);
//	  fs[0] += fes;
//	  fs[1] += 1.0;
//        }
//
//	nhat[X] = 0;
//	nhat[Z] = 0;
//
//	/* Surface in direction of (ic, jc+1,kc) */
//	index1 = cs_index(cs, ic, jc+1, kc);
//	map_status(map, index1, &status);
//
//        if (status == MAP_COLLOID) {
//          nhat[Y] = -1;
//          colloids_q_boundary_normal(cinfo, index, nhat, dn);
//	  blue_phase_fs(fe->param, dn, qs, MAP_COLLOID, &fes);
//	  fs[0] += fes;
//	  fs[1] += 1.0;
//        }
//
//	/* Surface in direction of (ic,jc-1,kc) */
//	index1 = cs_index(cs, ic, jc-1, kc);
//	map_status(map, index1, &status);
//
//        if (status == MAP_COLLOID) {
//          nhat[Y] = +1;
//          colloids_q_boundary_normal(cinfo, index, nhat, dn);
//	  blue_phase_fs(fe->param, dn, qs, MAP_COLLOID, &fes);
//	  fs[0] += fes;
//	  fs[1] += 1.0;
//        }
//
//	nhat[X] = 0;
//	nhat[Y] = 0;
//
//	/* Suface in direction of (ic,jc,kc+1) */
//	index1 = cs_index(cs, ic, jc, kc+1);
//	map_status(map, index1, &status);
//
//        if (status == MAP_COLLOID) {
//          nhat[Z] = -1;
//          colloids_q_boundary_normal(cinfo, index, nhat, dn);
//	  blue_phase_fs(fe->param, dn, qs, MAP_COLLOID, &fes);
//	  fs[0] += fes;
//	  fs[1] += 1.0;
//        }
//
//	/* Surface in direction of (ic,jc,kc-1) */
//	index1 = cs_index(cs, ic, jc, kc-1);
//	map_status(map, index1, &status);
//
//        if (status == MAP_COLLOID) {
//          nhat[Z] = +1;
//          colloids_q_boundary_normal(cinfo, index, nhat, dn);
//	  blue_phase_fs(fe->param, dn, qs, MAP_COLLOID, &fes);
//	  fs[0] += fes;
//	  fs[1] += 1.0;
//        }
//	
//      }
//    }
//  }

  return 0;
}

///*
// * Normal anchoring free energy
// * f_s = (1/2) w_1 ( Q_ab - Q^0_ab )^2  with Q^0_ab prefered orientation.
// *
// * Planar anchoring free energy (Fournier and Galatola EPL (2005).
// * f_s = (1/2) w_1 ( Q^tilde_ab - Q^tidle_perp_ab )^2
// *     + (1/2) w_2 ( Q^tidle^2 - S_0^2 )^2
// *
// * so w_2 must be zero for normal anchoring.
// */
//
/*****************************************************************************
 *
 *  shape_tensor_fs
 *
 *  Compute and return surface free energy area density given
 *    outward normal nhat[3]
 *    fluid R_ab rs
 *    site map status
 *
 *****************************************************************************/

__host__ int shape_tensor_fs(fe_st_param_t * feparam, const double dn[3],
			   double rs[3][3], char status,
			   double * fs) {

//  int ia, ib;
//  double w1, w2;
//  double q0[3][3];
//  double qtilde;
//  double amplitude;
//  double f1, f2, s0;
//  KRONECKER_DELTA_CHAR(d);
//
//  assert(status == MAP_BOUNDARY || status == MAP_COLLOID);
//
//  colloids_q_boundary(feparam, dn, qs, q0, status);
//
//  w1 = feparam->coll.w1;
//  w2 = feparam->coll.w2;
//
//  if (status == MAP_BOUNDARY) {
//    w1 = feparam->wall.w1;
//    w2 = feparam->wall.w2;
//  }
//
//  fe_lc_amplitude_compute(feparam, &amplitude);
//  s0 = 1.5*amplitude;  /* Fournier & Galatola S_0 = (3/2)A */
//
//  /* Free energy density */
//
//  f1 = 0.0;
//  f2 = 0.0;
//  for (ia = 0; ia < 3; ia++) {
//    for (ib = 0; ib < 3; ib++) {
//      f1 += (qs[ia][ib] - q0[ia][ib])*(qs[ia][ib] - q0[ia][ib]);
//      qtilde = qs[ia][ib] + 0.5*amplitude*d[ia][ib];
//      f2 += (qtilde*qtilde - s0*s0)*(qtilde*qtilde - s0*s0);
//    }
//  }
//
//  *fs = 0.5*w1*f1 + 0.5*w2*f2;

  return 0;
}

/*****************************************************************************
 *
 *  fe_st_bulk_grad
 *
 *****************************************************************************/

static int fe_st_bulk_grad(fe_st_t * fe,  cs_t * cs, map_t * map, double * fbg) {

  int ic, jc, kc, index;
  int nlocal[3];
  int status;

  double febg[2];
  double r[3][3];
  double h[3][3];
  double dr[3][3][3];
  double dsr[3][3];

  assert(fe);
  assert(cs);
  assert(map);
  assert(fbg);

  cs_nlocal(cs, nlocal);

  fbg[0] = 0.0;
  fbg[1] = 0.0;

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

        index = cs_index(cs, ic, jc, kc);
	map_status(map, index, &status);
	if (status != MAP_FLUID) continue;

	field_tensor(fe->r, index, r);
	field_grad_tensor_grad(fe->dr, index, dr);
	field_grad_tensor_delsq(fe->dr, index, dsr);
	fe_st_compute_h(fe, r, dr, dsr, h);

	shape_tensor_fbg(fe->param, r, dr, febg);
	fbg[0] += febg[0];
	fbg[1] += febg[1];
	
      }
    }
  }

  return 0;
}

/*****************************************************************************
 *
 *  shape_tensor_fbg
 *
 *  This computes statistics for the total, bulk and gradient free energy.
 *
 *****************************************************************************/

__host__ int shape_tensor_fbg(fe_st_param_t * feparam, double r[3][3],
                           double dr[3][3][3], double * febg) {

  //int ia;
  double k;
  //double lambda0;
  double trR;
  double *lambda;
  //double I2R;
  //double detR;

  //  double redshift, rredshift;
  //  const double r3 = 1.0/3.0;
  //  LEVI_CIVITA_CHAR(e);

  /* Use current redshift.*/
  //  redshift = feparam->redshift;
  //  rredshift = feparam->rredshift;

  k = feparam->k;
//  lambda0 = feparam->lambda0;
  
   /* Tr(R_ab) */
//   trR = 0.0;
//   for (ia = 0; ia < 3; ia++) {
//     trR += r[ia][ia];
//   }
  
  /* Version I*/
  /* ln(det(R_ab)) */
  /* 
  int twod;
  twod = feparam->twod;
  double detR, detaR, lndetaR;
   if(twod==0) {
     detR = r[0][0]*(r[1][1]*r[2][2] - r[2][1]*r[1][2]);
     detR -= r[0][1]*(r[1][0]*r[2][2]-r[2][0]*r[1][2]);
     detR += r[0][2]*(r[1][0]*r[2][1]-r[2][0]*r[1][1]);
     detaR = detR/(lambda0*lambda0*lambda0);
   }
   else {
     detR = r[0][0]*r[2][2] - r[0][2]*r[2][0];
     detaR = detR/(lambda0*lambda0);
   }
   lndetaR = log(detaR);
   febg[0] = 0.5*k*(trR - lambda0*lndetaR);
  */

  /* Version II*/
  /* Contribution bulk */
  //febg[0] = k*log(trR/lambda0);
  /* Version III*/
  //lambda = feparam->lambda;
  //I2R = lambda[0]*lambda[1] + lambda[1]*lambda[2] + lambda[2]*lambda[0];
  //detR= lambda[0]*lambda[1]*lambda[2];
  //febg[0] = k*log(I2R/pow(detR,2.0/3.0));
  /* Version IV*/
  lambda = feparam->lambda;
  trR = r[0][0]+r[1][1];
  febg[0] = k*log(trR/lambda[0]);

  /* Contribution gradient kapp0 and kappa1 */
  febg[1] = 0.0;

  return 0;
}

