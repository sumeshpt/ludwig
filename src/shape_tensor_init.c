/*****************************************************************************
 *
 *  shape_tensor_init.c
 *
 *  Various initial states for shape tensor
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2012-2022 The University of Edinburgh
 *
 *****************************************************************************/

#include <assert.h>
#include <math.h>
#include <stdlib.h>

#include "pe.h"
#include "util.h"
#include "coords.h"
#include "field_grad.h"
#include "shape_tensor.h"
#include "shape_tensor_init.h"
#include "shape_tensor_beris_edwards.h"
#include "util_shapetensor.h"
#include "ran.h"
#include "noise.h"

#define DEFAULT_SEED 13

//typedef struct rotation_s {
//  double m0[3][3];
//  double m1[3][3];
//  double m2[3][3];
//} rotation_t;

//static int rotation_create(rotation_t * rot, int, int, int, const double a[3]);
//static int rotate_inplace(const rotation_t * rot, double r[3]);
//
//void blue_phase_M_rot(double M[3][3], int dim, double alpha);
//
/*****************************************************************************
 *
 *  shape_tensor_isotropic_init
 *
 *  Initialise an isotropic shape tensor.
 *
 *****************************************************************************/
/*
int shape_tensor_isotropic_init(cs_t * cs, fe_st_param_t * param, field_t * fr) {

  int ic, jc, kc;
  int nlocal[3];
  int noffset[3];
  int index;

  double r[3][3];

  assert(cs);
  assert(fr);

  cs_nlocal(cs, nlocal);
  cs_nlocal_offset(cs, noffset);

  fe_st_r_isotropic(param, r);
  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = cs_index(cs, ic, jc, kc);
	field_tensor_set(fr, index, r);
      }
    }
  }

  return 0;
}
*/
/*****************************************************************************
 *
 *  shape_tensor_isotropic_init
 *
 *  Initialise an isotropic shape tensor isotropically.
 *
 *****************************************************************************/

int shape_tensor_isotropic_init(cs_t * cs, fe_st_param_t * param, field_t * fr, int nside) {

  int ic, jc, kc;
  int nlocal[3];
  int noffset[3];
  int index;
  int twod = 0;
  
  double r[3][3];
  double lambda0 = 0.0;
  double stini[3][3] = {0}; 

  assert(cs);
  assert(fr);
  assert(nside);

  twod = param->twod;
  if(twod > 0) {shapetensor_polygon(nside, &lambda0);}
  else {shapetensor_polyhedron(nside, stini);}

  cs_nlocal(cs, nlocal);
  cs_nlocal_offset(cs, noffset);

  fe_st_r_isotropic(param, r);
  if(twod > 0) {
    for (ic = 0; ic < 3; ic++) {r[ic][ic] = lambda0*r[ic][ic];}
    r[1][1]=0.0;
  }
  else {
    r[0][0] = stini[0][0];
    r[1][1] = stini[1][1];
    r[2][2] = stini[2][2];
  }
  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = cs_index(cs, ic, jc, kc);
	field_tensor_set(fr, index, r);
      }
    }
  }

  return 0;
}

///*****************************************************************************

///*****************************************************************************
// *
// *  blue_phase_O8M_init
// *
// *  Initialisation of BP I
// *
// *  An Euler rotation is allowed (a default {0, 0, 0} is 'no action').
// *  Input angles are in degrees.
// *
// *****************************************************************************/
//
//int blue_phase_O8M_init(cs_t * cs, fe_lc_param_t * param, field_t * fq,
//			const double euler_angles[3]) {
//
//  int ntotal[3], nlocal[3], noffset[3];
//
//  double root2;
//  double q0;
//  double amplitude0;
//  rotation_t rot = {0};
//
//  assert(cs);
//  assert(fq);
//
//  cs_ntotal(cs, ntotal);
//  cs_nlocal(cs, nlocal);
//  cs_nlocal_offset(cs, noffset);
//
//  root2 = sqrt(2.0);
//  q0    = param->q0;
//  amplitude0 = param->amplitude0;
//
//  /* Set up rotation matrices with negative angles. Clockwise rotation */
//  /* of arguments leads to counterclockwise rotation of the Q-tensor.  */
//  /* So we add a sign here. */
//
//  {
//    double angles[3]  = {0}; /* radians */
//    PI_DOUBLE(pi);
//
//    angles[0] = -1.0*pi*euler_angles[0]/180.0;
//    angles[1] = -1.0*pi*euler_angles[1]/180.0;
//    angles[2] = -1.0*pi*euler_angles[2]/180.0;
//
//    rotation_create(&rot, Z, X, Z, angles);
//  }
//
//  for (int ic = 1; ic <= nlocal[X]; ic++) {
//    double x = noffset[X] + ic;
//    for (int jc = 1; jc <= nlocal[Y]; jc++) {
//      double y = noffset[Y] + jc;
//      for (int kc = 1; kc <= nlocal[Z]; kc++) {
//	double z = noffset[Z] + kc;
//	double r[3] = {0};
//
//	/* Rotate around the centre */
//	r[X] = x - 0.5*ntotal[X];
//	r[Y] = y - 0.5*ntotal[Y];
//	r[Z] = z - 0.5*ntotal[Z];
//
//	rotate_inplace(&rot, r);
//
//	r[X] += 0.5*ntotal[X];
//	r[Y] += 0.5*ntotal[Y];
//	r[Z] += 0.5*ntotal[Z];
//
//	{
//	  int index   = cs_index(cs, ic, jc, kc);
//	  double cosx = cos(root2*q0*r[X]);
//	  double cosy = cos(root2*q0*r[Y]);
//	  double cosz = cos(root2*q0*r[Z]);
//	  double sinx = sin(root2*q0*r[X]);
//	  double siny = sin(root2*q0*r[Y]);
//	  double sinz = sin(root2*q0*r[Z]);
//	  double q[3][3] = {0};
//
//	  q[X][X] = amplitude0*( -2.0*cosy*sinz +       sinx*cosz + cosx*siny);
//	  q[X][Y] = amplitude0*(root2*cosy*cosz + root2*sinx*sinz - sinx*cosy);
//	  q[X][Z] = amplitude0*(root2*cosx*cosy + root2*sinz*siny - cosx*sinz);
//	  q[Y][X] = q[X][Y];
//	  q[Y][Y] = amplitude0*( -2.0*sinx*cosz +       siny*cosx + cosy*sinz);
//	  q[Y][Z] = amplitude0*(root2*cosz*cosx + root2*siny*sinx - siny*cosz);
//	  q[Z][X] = q[X][Z];
//	  q[Z][Y] = q[Y][Z];
//	  q[Z][Z] = - q[X][X] - q[Y][Y];
//
//	  field_tensor_set(fq, index, q);
//	}
//	/* Next site */
//      }
//    }
//  }
//
//  return 0;
//}
//
///*****************************************************************************
// *
// *  blue_phase_O2_init
// *
// *  This initialisation is for BP II.
// *
// *  An Euler rotation is allowed (a default {0, 0, 0} is 'no action').
// *  Input angles are in degrees.
// *
// *****************************************************************************/
//
//int blue_phase_O2_init(cs_t * cs, fe_lc_param_t * param, field_t * fq,
//		       const double euler_angles[3]) {
//
//  int ntotal[3], nlocal[3], noffset[3];
//
//  double q0;
//  double amplitude0;
//  rotation_t rot = {0};
//
//  assert(cs);
//  assert(fq);
//
//  cs_ntotal(cs, ntotal);
//  cs_nlocal(cs, nlocal);
//  cs_nlocal_offset(cs, noffset);
//
//  q0 = param->q0;
//  amplitude0 = param->amplitude0;
//
//  /* Set up rotation matrices with negative angles. Clockwise rotation */
//  /* of arguments leads to counterclockwise rotation of the Q-tensor.  */
//  /* Hence the factor of -1.0 below. */
//
//  {
//    double angles[3] = {0}; /* radians */
//    PI_DOUBLE(pi);
//
//    angles[0] = -1.0*pi*euler_angles[0]/180.0;
//    angles[1] = -1.0*pi*euler_angles[1]/180.0;
//    angles[2] = -1.0*pi*euler_angles[2]/180.0;
//
//    rotation_create(&rot, Z, X, Z, angles);
//  }
//
//  for (int ic = 1; ic <= nlocal[X]; ic++) {
//    double x = noffset[X] + ic;
//    for (int jc = 1; jc <= nlocal[Y]; jc++) {
//      double y = noffset[Y] + jc;
//      for (int kc = 1; kc <= nlocal[Z]; kc++) {
//	double z = noffset[Z] + kc;
//	double r[3] = {0};
//
//	r[X] = x - 0.5*ntotal[X];
//	r[Y] = y - 0.5*ntotal[Y];
//	r[Z] = z - 0.5*ntotal[Z];
//
//	rotate_inplace(&rot, r);
//
//	r[X] += 0.5*ntotal[X];
//	r[Y] += 0.5*ntotal[Y];
//	r[Z] += 0.5*ntotal[Z];
//
//	{
//	  int index = cs_index(cs, ic, jc, kc);
//	  double cosx = cos(2.0*q0*r[X]);
//	  double cosy = cos(2.0*q0*r[Y]);
//	  double cosz = cos(2.0*q0*r[Z]);
//	  double sinx = sin(2.0*q0*r[X]);
//	  double siny = sin(2.0*q0*r[Y]);
//	  double sinz = sin(2.0*q0*r[Z]);
//	  double q[3][3] = {0};
//
//	  q[X][X] = amplitude0*(cosz - cosy);
//	  q[X][Y] = amplitude0*sinz;
//	  q[X][Z] = amplitude0*siny;
//	  q[Y][X] = q[X][Y];
//	  q[Y][Y] = amplitude0*(cosx - cosz);
//	  q[Y][Z] = amplitude0*sinx;
//	  q[Z][X] = q[X][Z];
//	  q[Z][Y] = q[Y][Z];
//	  q[Z][Z] = - q[X][X] - q[Y][Y];
//
//	  field_tensor_set(fq, index, q);
//	}
//	/* Next site */
//      }
//    }
//  }
//
//  return 0;
//}
//
///*****************************************************************************
// *
// *  blue_phase_H2D_init
// *
// *  This initialisation is for 2D hexagonal BP.
// *
// *****************************************************************************/
//
//int blue_phase_H2D_init(cs_t * cs, fe_lc_param_t * param, field_t * fq) {
//
//  int ic, jc, kc;
//  int nlocal[3];
//  int noffset[3];
//  int index;
//
//  double q[3][3];
//  double x, y;
//  double r3;
//  double q0;
//  double amplitude0;
//
//  assert(cs);
//  assert(fq);
//
//  r3 = sqrt(3.0);
//  q0 = param->q0;
//  amplitude0 = param->amplitude0;
//
//  cs_nlocal(cs, nlocal);
//  cs_nlocal_offset(cs, noffset);
//
//  for (ic = 1; ic <= nlocal[X]; ic++) {
//    x = noffset[X] + ic;
//    for (jc = 1; jc <= nlocal[Y]; jc++) {
//      y = noffset[Y] + jc;
//      for (kc = 1; kc <= nlocal[Z]; kc++) {
//
//	index = cs_index(cs, ic, jc, kc);
//
//	q[X][X] = amplitude0*(-1.5*   cos(q0*x)*cos(q0*r3*y));
//	q[X][Y] = amplitude0*(-0.5*r3*sin(q0*x)*sin(q0*r3*y));
//	q[X][Z] = amplitude0*(     r3*cos(q0*x)*sin(q0*r3*y));
//	q[Y][X] = q[X][Y];
//	q[Y][Y] = amplitude0*(-cos(2.0*q0*x) - 0.5*cos(q0*x)*cos(q0*r3*y));
//	q[Y][Z] = amplitude0*(-sin(2.0*q0*x) -     sin(q0*x)*cos(q0*r3*y));
//	q[Z][X] = q[X][Z];
//	q[Z][Y] = q[Y][Z];
//	q[Z][Z] = - q[X][X] - q[Y][Y];
//
//	field_tensor_set(fq, index, q);
//      }
//    }
//  }
//
//  return 0;
//}
//
///*****************************************************************************
// *
// *  blue_phase_H3DA_init
// *
// *  This initialisation is for 3D hexagonal BP A.
// *
// *****************************************************************************/
//
//int blue_phase_H3DA_init(cs_t * cs, fe_lc_param_t * param, field_t * fq) {
//
//  int ic, jc, kc;
//  int nlocal[3];
//  int noffset[3];
//  int index;
//
//  double q[3][3];
//  double x, y, z;
//  double r3;
//  double q0;
//  double amplitude0;
//  double ltot[3];
//
//  assert(cs);
//  assert(fq);
//
//  r3 = sqrt(3.0);
//  q0 = param->q0;
//  amplitude0 = param->amplitude0;
//
//  cs_ltot(cs, ltot);
//  cs_nlocal(cs, nlocal);
//  cs_nlocal_offset(cs, noffset);
//
//  for (ic = 1; ic <= nlocal[X]; ic++) {
//    x = noffset[X] + ic;
//    for (jc = 1; jc <= nlocal[Y]; jc++) {
//      y = noffset[Y] + jc;
//      for (kc = 1; kc <= nlocal[Z]; kc++) {
//	z = noffset[Z] + kc;
//
//	index = cs_index(cs, ic, jc, kc);
//
//	q[X][X] = amplitude0*(-1.5*cos(q0*x)*cos(q0*r3*y)
//			       + 0.25*cos(q0*ltot[X]/ltot[Z]*z)); 
//	q[X][Y] = amplitude0*(-0.5*r3*sin(q0*x)*sin(q0*r3*y)
//			       + 0.25*sin(q0*ltot[X]/ltot[Z]*z));
//	q[X][Z] = amplitude0*(r3*cos(q0*x)*sin(q0*r3*y));
//	q[Y][X] = q[X][Y];
//	q[Y][Y] = amplitude0*(-cos(2.0*q0*x)-0.5*cos(q0*x)*cos(q0*r3*y)
//			       -0.25*cos(q0*ltot[X]/ltot[Z]*z));
//	q[Y][Z] = amplitude0*(-sin(2.0*q0*x)-sin(q0*x)*cos(q0*r3*y));
//	q[Z][X] = q[X][Z];
//	q[Z][Y] = q[Y][Z];
//	q[Z][Z] = - q[X][X] - q[Y][Y];
//
//	field_tensor_set(fq, index, q);
//      }
//    }
//  }
//
//  return 0;
//}
//
///*****************************************************************************
// *
// *  blue_phase_H3DB_init
// *
// *  This initialisation is for 3D hexagonal BP B.
// *
// *****************************************************************************/
//
//int blue_phase_H3DB_init(cs_t * cs, fe_lc_param_t * param, field_t * fq) {
//
//  int ic, jc, kc;
//  int nlocal[3];
//  int noffset[3];
//  int index;
//
//  double q[3][3];
//  double x, y, z;
//  double r3;
//  double q0;
//  double amplitude0;
//  double ltot[3];
//
//  assert(cs);
//  assert(fq);
//
//  r3 = sqrt(3.0);
//  q0 = param->q0;
//  amplitude0 = param->amplitude0;
//
//  cs_ltot(cs, ltot);
//  cs_nlocal(cs, nlocal);
//  cs_nlocal_offset(cs, noffset);
//
//  for (ic = 1; ic <= nlocal[X]; ic++) {
//    x = noffset[X] + ic;
//    for (jc = 1; jc <= nlocal[Y]; jc++) {
//      y = noffset[Y] + jc;
//      for (kc = 1; kc <= nlocal[Z]; kc++) {
//	z = noffset[Z] + kc;
//
//	index = cs_index(cs, ic, jc, kc);
//
//	q[X][X] = amplitude0*(1.5*cos(q0*x)*cos(q0*r3*y)
//			       + 0.25*cos(q0*ltot[X]/ltot[Z]*z)); 
//	q[X][Y] = amplitude0*(0.5*r3*sin(q0*x)*sin(q0*r3*y)
//			       + 0.25*sin(q0*ltot[X]/ltot[Z]*z));
//	q[X][Z] = amplitude0*(-r3*cos(q0*x)*sin(q0*r3*y));
//	q[Y][X] = q[X][Y];
//	q[Y][Y] = amplitude0*(cos(2.0*q0*x) + 0.5*cos(q0*x)*cos(q0*r3*y)
//			       - 0.25*cos(q0*ltot[X]/ltot[Z]*z));
//	q[Y][Z] = amplitude0*(sin(2.0*q0*x) + sin(q0*x)*cos(q0*r3*y));
//	q[Z][X] = q[X][Z];
//	q[Z][Y] = q[Y][Z];
//	q[Z][Z] = - q[X][X] - q[Y][Y];
//
//	field_tensor_set(fq, index, q);
//      }
//    }
//  }
//
//  return 0;
//}
//
///*****************************************************************************
// *
// *  blue_phase_O5_init
// *
// *  This initialisation is for O5.
// *
// *****************************************************************************/
//
//int blue_phase_O5_init(cs_t * cs, fe_lc_param_t * param, field_t * fq) {
//
//  int ic, jc, kc;
//  int nlocal[3];
//  int noffset[3];
//  int index;
//
//  double q[3][3];
//  double x, y, z;
//  double q0;
//  double amplitude0;
//
//  assert(fq);
//
//  cs_nlocal(cs, nlocal);
//  cs_nlocal_offset(cs, noffset);
//
//  q0 = param->q0;
//  amplitude0 = param->amplitude0;
//
//  for (ic = 1; ic <= nlocal[X]; ic++) {
//    x = noffset[X] + ic;
//    for (jc = 1; jc <= nlocal[Y]; jc++) {
//      y = noffset[Y] + jc;
//      for (kc = 1; kc <= nlocal[Z]; kc++) {
//	z = noffset[Z] + kc;
//
//	index = cs_index(cs, ic, jc, kc);
//
//	q[X][X] = amplitude0*
//            (2.0*cos(sqrt(2.0)*q0*y)*cos(sqrt(2.0)*q0*z)-
//                 cos(sqrt(2.0)*q0*x)*cos(sqrt(2.0)*q0*z)-
//                 cos(sqrt(2.0)*q0*x)*cos(sqrt(2.0)*q0*y)); 
//	q[X][Y] = amplitude0*
//            (sqrt(2.0)*cos(sqrt(2.0)*q0*y)*sin(sqrt(2.0)*q0*z)-
//             sqrt(2.0)*cos(sqrt(2.0)*q0*x)*sin(sqrt(2.0)*q0*z)-
//             sin(sqrt(2.0)*q0*x)*sin(sqrt(2.0)*q0*y));
//	q[X][Z] = amplitude0*
//            (sqrt(2.0)*cos(sqrt(2.0)*q0*x)*sin(sqrt(2.0)*q0*y)-
//             sqrt(2.0)*cos(sqrt(2.0)*q0*z)*sin(sqrt(2.0)*q0*y)-
//             sin(sqrt(2.0)*q0*x)*sin(sqrt(2.0)*q0*z));
//	q[Y][X] = q[X][Y];
//	q[Y][Y] = amplitude0*
//            (2.0*cos(sqrt(2.0)*q0*x)*cos(sqrt(2.0)*q0*z)-
//                 cos(sqrt(2.0)*q0*y)*cos(sqrt(2.0)*q0*x)-
//                 cos(sqrt(2.0)*q0*y)*cos(sqrt(2.0)*q0*z));
//	q[Y][Z] = amplitude0*
//            (sqrt(2.0)*cos(sqrt(2.0)*q0*z)*sin(sqrt(2.0)*q0*x)-
//             sqrt(2.0)*cos(sqrt(2.0)*q0*y)*sin(sqrt(2.0)*q0*x)-
//             sin(sqrt(2.0)*q0*y)*sin(sqrt(2.0)*q0*z));
//	q[Z][X] = q[X][Z];
//	q[Z][Y] = q[Y][Z];
//	q[Z][Z] = - q[X][X] - q[Y][Y];
//
//	field_tensor_set(fq, index, q);
//      }
//    }
//  }
//
//  return 0;
//}
//
///*****************************************************************************
// *
// *  blue_phase_DTC_init
// *
// *  This initialisation is with double twist cylinders.
// *
// *****************************************************************************/
//
//int blue_phase_DTC_init(cs_t * cs, fe_lc_param_t * param, field_t * fq) {
//
//  int ic, jc, kc;
//  int nlocal[3];
//  int noffset[3];
//  int index;
//
//  double q[3][3];
//  double x, y;
//  double q0;
//  double amplitude0;
//
//  assert(cs);
//  assert(fq);
//
//  cs_nlocal(cs, nlocal);
//  cs_nlocal_offset(cs, noffset);
//
//  q0 = param->q0;
//  amplitude0 = param->amplitude0;
//
//  for (ic = 1; ic <= nlocal[X]; ic++) {
//    x = noffset[X] + ic;
//    for (jc = 1; jc <= nlocal[Y]; jc++) {
//      y = noffset[Y] + jc;
//      for (kc = 1; kc <= nlocal[Z]; kc++) {
//
//	index = cs_index(cs, ic, jc, kc);
//
//	q[X][X] = -amplitude0*cos(2*q0*y);
//	q[X][Y] = 0.0;
//	q[X][Z] = amplitude0*sin(2.0*q0*y);
//	q[Y][X] = q[X][Y];
//	q[Y][Y] = -amplitude0*cos(2.0*q0*x);
//	q[Y][Z] = -amplitude0*sin(2.0*q0*x);
//	q[Z][X] = q[X][Z];
//	q[Z][Y] = q[Y][Z];
//	q[Z][Z] = - q[X][X] - q[Y][Y];
//
//	field_tensor_set(fq, index, q);
//      }
//    }
//  }
//
//  return 0;
//}
//
///*****************************************************************************
// *
// *  blue_phase_BPIII_init
// *
// *  This initialisation is with Blue Phase III, randomly positioned
// *  and oriented DTC-cylinders in isotropic (0) or cholesteric (1) environment.
// *
// *  NOTE: The rotations are not rigorously implemented; no cross-boundary 
// *        communication is performed. 
// *        Hence, the decomposition must consist of sufficiently large volumes.
// *        
// *****************************************************************************/
//
//int blue_phase_BPIII_init(cs_t * cs, fe_lc_param_t * param, field_t * fq,
//			  const double specs[3]) {
//
//  int ic, jc, kc;
//  int ir, jr, kr; 	/* indices for rotated output */
//  int ia, ib, ik, il, in;
//  int nlocal[3];
//  int ntotal[3];
//  int noffset[3];
//  int index;
//  double q[3][3], q0[3][3];
//  double x, y, z;
//  double *a, *b;	/* rotation angles */
//  double *C;     	/* global coordinates of DTC-centres */
//  int N=2, R=3, ENV=1; 	/* default no. & radius & environment */ 
//  double rc[3];	  	/* distance DTC-centre - site */ 
//  double rc_r[3]; 	/* rotated vector */ 
//  double Mx[3][3], My[3][3]; /* rotation matrices */
//  double phase1, phase2;
//  double n[3]={0.0,0.0,0.0};
//  double q0_pitch;      /* Just q0 scalar */
//  double amplitude0;
//  PI_DOUBLE(pi);
//
//  assert(cs);
//  assert(fq);
//  assert(specs);
//
//  N = (int) specs[0];
//  R = (int) specs[1];
//  ENV = (int) specs[2];
//
//  a = (double*)calloc(N, sizeof(double));
//  b = (double*)calloc(N, sizeof(double));
//  C = (double*)calloc(3*N, sizeof(double));
//
//  cs_nlocal(cs, nlocal);
//  cs_ntotal(cs, ntotal);
//  cs_nlocal_offset(cs, noffset);
//
//  q0_pitch = param->q0;
//  amplitude0 = param->amplitude0;
//
//    /* Initialise random rotation angles and centres in serial */
//    /* to get the same random numbers on all processes */
//    for(in = 0; in < N; in++){
//
//      a[in] = 2.0*pi * ran_serial_uniform();
//      b[in] = 2.0*pi * ran_serial_uniform();
//      C[3*in]   = ntotal[X] * ran_serial_uniform(); 
//      C[3*in+1] = ntotal[Y] * ran_serial_uniform(); 
//      C[3*in+2] = ntotal[Z] * ran_serial_uniform(); 
//
//    }
//
//  /* Setting environment configuration */
//  for (ic = 1; ic <= nlocal[X]; ic++) {
//    for (jc = 1; jc <= nlocal[Y]; jc++) {
//      y = noffset[Y] + jc;
//      for (kc = 1; kc <= nlocal[Z]; kc++) {
//
//	if (ENV == 0){
//
//	  phase1 = pi*(0.5 - ran_parallel_uniform());
//	  phase2 = pi*(0.5 - ran_parallel_uniform());
//
//	  n[X] = cos(phase1)*sin(phase2);
//	  n[Y] = sin(phase1)*sin(phase2);
//	  n[Z] = cos(phase2);
//
//	  fe_lc_q_uniaxial(param, n, q);
//
//	 /* The amplitude of the orderparameter is hardwired */
//	 /* for the random and isotropic background configuration */     
//	  for (ia = 0; ia < 3; ia++) {
//	    for (ib = 0; ib < 3; ib++) {
//	      q[ia][ib] *= 1.0e-6;
//	    }
//	  }
//
//	}
//	if (ENV == 1){
//
//	  /* cholesteric helix along y-direction */
//	  n[X] = cos(q0_pitch*y);
//	  n[Y] = 0.0;
//	  n[Z] = -sin(q0_pitch*y);
//
//	  fe_lc_q_uniaxial(param, n, q);
//
//	  for (ia = 0; ia < 3; ia++) {
//	    for (ib = 0; ib < 3; ib++) {
//	      q[ia][ib] *= amplitude0;
//	    }
//	  }
//	  
//	}
//
//	index = cs_index(cs, ic, jc, kc);
//	field_tensor_set(fq, index, q);
//      }
//    }
//  }
//
//  /* Replace configuration inside DTC-domains */
//  /* by sweeping through all local sites */
//  for(in = 0; in<N; in++){
//
//    blue_phase_M_rot(Mx,0,a[in]);
//    blue_phase_M_rot(My,1,b[in]);
//
//    for (ic = 1; ic <= nlocal[X]; ic++) {
//      x = noffset[X] + ic;
//      for (jc = 1; jc <= nlocal[Y]; jc++) {
//	y = noffset[Y] + jc;
//	for (kc = 1; kc <= nlocal[Z]; kc++) {
//	  z = noffset[Z] + kc;
//
//	  rc[X] = x - C[3*in];
//	  rc[Y] = y - C[3*in+1];
//	  rc[Z] = z - C[3*in+2];
//
//	  /* If current site is in ROI perform double */
//	  /* rotation around local x- and y-axis */
//	  if(rc[0]*rc[0] + rc[1]*rc[1] + rc[2]*rc[2] < R*R){
//
//	    for(ia=0; ia<3; ia++){
//	      rc_r[ia] = 0.0;
//	      for(ik=0; ik<3; ik++){
//		for(il=0; il<3; il++){
//		  rc_r[ia] += My[ia][ik] * Mx[ik][il] * rc[il];
//		}
//	      }
//	    }
//
//	    /* DTC symmetric wrt local z-axis */
//	    q0[X][X] = -amplitude0*cos(2*q0_pitch*rc[Y]);
//	    q0[X][Y] = 0.0;
//	    q0[X][Z] = amplitude0*sin(2.0*q0_pitch*rc[Y]);
//	    q0[Y][X] = q[X][Y];
//	    q0[Y][Y] = -amplitude0*cos(2.0*q0_pitch*rc[X]);
//	    q0[Y][Z] = -amplitude0*sin(2.0*q0_pitch*rc[X]);
//	    q0[Z][X] = q[X][Z];
//	    q0[Z][Y] = q[Y][Z];
//	    q0[Z][Z] = - q[X][X] - q[Y][Y];
//
//	    /* Transform order parameter tensor */ 
//            /* Determine local output index */
//            ir = (int)(C[3*in] + rc_r[X] - noffset[X]);
//            jr = (int)(C[3*in+1] + rc_r[Y] - noffset[Y]);
//            kr = (int)(C[3*in+2] + rc_r[Z] - noffset[Z]);
//
//	    /* Replace if index is in local domain */
//	    if((1 <= ir && ir <= nlocal[X]) &&  
//	       (1 <= jr && jr <= nlocal[Y]) &&  
//               (1 <= kr && kr <= nlocal[Z]))
//	    {
//   /*******************************************************************
//   * NOTE: The individual componentes of the tensor order parameter are
//   *       not transformed, i.e. rotated, as this leads to considerable 
//   *       instabilities in the calculation of the gradients.
//   *       BPIII emerges more reliably from an unrotated OP.
//   *******************************************************************/
//
//	      q[X][X] = q0[X][X];
//	      q[X][Y] = q0[X][Y];
//	      q[X][Z] = q0[X][Z];
//	      q[Y][X] = q[X][Y];
//	      q[Y][Y] = q0[Y][Y];
//	      q[Y][Z] = q0[Y][Z];
//	      q[Z][X] = q[X][Z];
//	      q[Z][Y] = q[Y][Z];
//	      q[Z][Z] = - q[X][X] - q[Y][Y];
//
//	      index = cs_index(cs, ir, jr, kr);
//	      field_tensor_set(fq, index, q);
//	    }
//
//	  }
//
//	}
//      }
//    }
//
//  }
//
//  free(a);
//  free(b);
//  free(C);
//
//  return 0;
//}
//
//
///*****************************************************************************
// *
// *  blue_phase_twist_init
// *
// *  Initialise a uniaxial helix in the indicated helical axis.
// *  Uses the current free energy parameters
// *     q0 (P=2pi/q0)
// *     amplitude
// *
// *****************************************************************************/
//
//int blue_phase_twist_init(cs_t * cs, fe_lc_param_t * param, field_t * fq,
//			  int helical_axis) {
//
//  int ic, jc, kc;
//  int nlocal[3];
//  int noffset[3];
//  int index;
//
//  double n[3];
//  double q[3][3];
//  double x, y, z;
//  double q0;
//
//  assert(cs);
//  assert(param);
//  assert(fq);
//  assert(helical_axis == X || helical_axis == Y || helical_axis == Z);
//
//  cs_nlocal(cs, nlocal);
//  cs_nlocal_offset(cs, noffset);
//
//  q0 = param->q0;
//
//  n[X] = 0.0;
//  n[Y] = 0.0;
//  n[Z] = 0.0;
// 
//  for (ic = 1; ic <= nlocal[X]; ic++) {
//
//    if (helical_axis == X) {
//      x = noffset[X] + ic;
//      n[Y] = cos(q0*x);
//      n[Z] = sin(q0*x);
//    }
//
//    for (jc = 1; jc <= nlocal[Y]; jc++) {
//
//      if (helical_axis == Y) {
//	y = noffset[Y] + jc;
//	n[X] = cos(q0*y);
//	n[Z] = -sin(q0*y);
//      }
//
//      for (kc = 1; kc <= nlocal[Z]; kc++) {
//	
//	index = cs_index(cs, ic, jc, kc);
//
//	if (helical_axis == Z) {
//	  z = noffset[Z] + kc;
//	  n[X] = cos(q0*z);
//	  n[Y] = sin(q0*z);
//	}
//
//	fe_lc_q_uniaxial(param, n, q);
//	field_tensor_set(fq, index, q);
//      }
//    }
//  }
//
//  return 0;
//}
//
///*****************************************************************************
// *
// *  blue_phase_nematic_init
// *
// *  Initialise a uniform uniaxial nematic.
// *
// *  The inputs are the amplitude A and the vector n_a (which we explicitly
// *  convert to a unit vector here).
// *
// *****************************************************************************/
//
//int blue_phase_nematic_init(cs_t * cs, fe_lc_param_t * param, field_t * fq,
//			    const double n[3]) {
//
//  int ic, jc, kc;
//  int nlocal[3];
//  int noffset[3];
//  int ia, index;
//
//  double nhat[3];
//  double q[3][3];
//
//  assert(cs);
//  assert(fq);
//  assert(n);
//  assert(modulus(n) > 0.0);
//
//  cs_nlocal(cs, nlocal);
//  cs_nlocal_offset(cs, noffset);
//
//  for (ia = 0; ia < 3; ia++) {
//    nhat[ia] = n[ia] / modulus(n);
//  }
//
//  fe_lc_q_uniaxial(param, nhat, q);
//
//  for (ic = 1; ic <= nlocal[X]; ic++) {
//    for (jc = 1; jc <= nlocal[Y]; jc++) {
//      for (kc = 1; kc <= nlocal[Z]; kc++) {
//
//	index = cs_index(cs, ic, jc, kc);
//	field_tensor_set(fq, index, q);
//      }
//    }
//  }
//
//  return 0;
//}
//
///*****************************************************************************
// *
// *  blue_phase_active_nematic_init
// *
// *  Initialise a uniform uniaxial nematic with a kink in the centre.
// *
// *  This is like the above initialisation in blue_phase_nematic_init
// *  apart from the symmetry-breaking kinks in the middle of the system.
// *
// *****************************************************************************/
//
//int blue_phase_active_nematic_init(cs_t * cs, fe_lc_param_t * param,
//				   field_t * fq, const double n[3]) {
//
//  int ic, jc, kc;
//  int nlocal[3];
//  int ntotal[3];
//  int noffset[3];
//  int ia, index;
//
//  double nhat[3];
//  double q[3][3];
//
//  double nkink1[3], nkink2[3];
//  double qkink1[3][3], qkink2[3][3];
//
//  double ang;
//  PI_DOUBLE(pi);
//
//  ang = pi/180.0*10.0;
//
//  assert(cs);
//  assert(modulus(n) > 0.0);
//
//  cs_nlocal(cs, nlocal);
//  cs_ntotal(cs, ntotal);
//  cs_nlocal_offset(cs, noffset);
//
//  for (ia = 0; ia < 3; ia++) {
//    nhat[ia] = n[ia] / modulus(n);
//  }
//
//  /* 2 kinks depending on the primary alignment */
//
//  /* Kink for primary alignment along x */	 
//  if (nhat[0] == 1.0) {
//    nkink1[0] = nhat[0]*sin(ang);
//    nkink1[1] = nhat[1];
//    nkink1[2] = nhat[0]*cos(ang);
//
//    nkink2[0] = -nhat[0]*sin(ang);
//    nkink2[1] =  nhat[1];
//    nkink2[2] =  nhat[0]*cos(ang);
//  }
//  /* Kink for primary alignment along y */	 
//
//  if (nhat[1] == 1.0) {
//    nkink1[0] = nhat[0];
//    nkink1[1] = nhat[1]*sin(ang);
//    nkink1[2] = nhat[1]*cos(ang);
//
//    nkink2[0] =  nhat[0];
//    nkink2[1] = -nhat[1]*sin(ang);
//    nkink2[2] =  nhat[1]*cos(ang);
//  }
//
//  fe_lc_q_uniaxial(param, nhat, q);
//  fe_lc_q_uniaxial(param, nkink1, qkink1);
//  fe_lc_q_uniaxial(param, nkink2, qkink2);
//
//  for (ic = 1; ic <= nlocal[X]; ic++) {
//    int ix = noffset[X] + ic;
//    for (jc = 1; jc <= nlocal[Y]; jc++) {
//      int iy = noffset[Y] + jc;
//      for (kc = 1; kc <= nlocal[Z]; kc++) {
//	int iz = noffset[Z] + kc;
//
//	index = cs_index(cs, ic, jc, kc);
//	field_tensor_set(fq, index, q);
//
//        /* If alignment along x region around 
//	   z = ntotal[Z]/2 is being replaced */
// 
//	if(nhat[0] == 1.0) {
//	  if (iz == ntotal[Z]/2 || iz == (ntotal[Z]-1)/2) {
//	    if (ix <= ntotal[X]/2) {
//	      field_tensor_set(fq, index, qkink1);
//	    }
//	    else {
//	      field_tensor_set(fq, index, qkink2);
//	    }
//	  }
//	}
//
//        /* If alignment along y region around 
//	   z=ntotal[Z]/2 is being replaced */
//
//	if(nhat[1] == 1.0){
//	  if (iz == ntotal[Z]/2 || iz == (ntotal[Z]-1)/2) {
//	    if (iy <= ntotal[Y]/2) {
//	      field_tensor_set(fq, index, qkink1);
//	    }
//	    else {
//	      field_tensor_set(fq, index, qkink2);
//	    }
//	  }
//	}
//
//      }
//    }
//  }
//
//
//  return 0;
//}
//
///*****************************************************************************
// *
// *  lc_active_nematic_init_q2d
// *
// *  Intended for quasi-2d systems in which the z-direction is Lz = 1).
// *
// *  For istrip == X, the director looks like
// *
// *       - - - - - -     ^ 
// *       / / / \ \ \     |
// *       - - - - - -     | Y   ---> X
// *
// *  For istrip == Y, swap the labels in the above picture.
// *
// *****************************************************************************/
//
//int lc_active_nematic_init_q2d(cs_t * cs, fe_lc_param_t * param, field_t * fq,
//			       int istrip) {
//
//  int ic, jc, kc;
//  int nlocal[3];
//  int ntotal[3];
//  int noffset[3];
//  int index;
//
//  double nhat[3];
//  double q[3][3];
//
//  double nkink1[3], nkink2[3];
//  double qkink1[3][3], qkink2[3][3];
//
//  double ang;
//  PI_DOUBLE(pi);
//
//  assert(cs);
//  assert(param);
//  assert(fq);
//  assert(istrip == X || istrip == Y);
//
//  ang = pi/180.0*10.0;
//
//  cs_nlocal(cs, nlocal);
//  cs_ntotal(cs, ntotal);
//  cs_nlocal_offset(cs, noffset);
//
//  if (istrip == X) {
//    nhat[X] = 1.0;
//    nhat[Y] = 0.0;
//    nhat[Z] = 0.0;
//    nkink1[X] = sin(ang);
//    nkink1[Y] = cos(ang);
//    nkink1[Z] = 0.0;
//    nkink2[X] = -sin(ang);
//    nkink2[Y] =  cos(ang);
//    nkink2[Z] = 0.0;
//  }
//
//  if (istrip == Y) {
//    nhat[X] = 0.0;
//    nhat[Y] = 1.0;
//    nhat[Z] = 0.0;
//    nkink1[X] = cos(ang);
//    nkink1[Y] = sin(ang);
//    nkink1[Z] = 0.0;
//    nkink2[X] =  cos(ang);
//    nkink2[Y] = -sin(ang);
//    nkink2[Z] =  0.0;
//  }
//
//  fe_lc_q_uniaxial(param, nhat, q);
//  fe_lc_q_uniaxial(param, nkink1, qkink1);
//  fe_lc_q_uniaxial(param, nkink2, qkink2);
//
//  for (ic = 1; ic <= nlocal[X]; ic++) {
//    int ix = noffset[X] + ic;
//    for (jc = 1; jc <= nlocal[Y]; jc++) {
//      int iy = noffset[Y] + jc;
//      for (kc = 1; kc <= nlocal[Z]; kc++) {
//
//	index = cs_index(cs, ic, jc, kc);
//
//	/* Background */
//	field_tensor_set(fq, index, q);
//
//        /* Central strip parallel to X */
// 
//	if (istrip == X) {
//	  if (iy == ntotal[Y]/2 || iy == (ntotal[Y]-1)/2) {
//	    if (ix <= ntotal[X]/2) {
//	      field_tensor_set(fq, index, qkink1);
//	    }
//	    else {
//	      field_tensor_set(fq, index, qkink2);
//	    }
//	  }
//	}
//
//        /* Central strip parallel to Y */
//
//	if (istrip == Y) {
//	  if (ix == ntotal[X]/2 || ix == (ntotal[X]-1)/2) {
//	    if (iy <= ntotal[Y]/2) {
//	      field_tensor_set(fq, index, qkink1);
//	    }
//	    else {
//	      field_tensor_set(fq, index, qkink2);
//	    }
//	  }
//	}
//
//	/* Next site */
//      }
//    }
//  }
//
//  return 0;
//}
//
///*****************************************************************************
// *
// *  blue_phase_chi_edge
// *  Setting  chi edge disclination
// *  Using the current free energy parameter q0 (P=2pi/q0)
// *
// *****************************************************************************/
//
//int blue_phase_chi_edge(cs_t * cs, fe_lc_param_t * param, field_t * fq, int N,
//			double z0, double x0) {
//
//  int ic, jc, kc;
//  int nlocal[3];
//  int noffset[3];
//  int index;
//
//  double q[3][3];
//  double x, z;
//  double n[3];
//  double theta;
//  double q0;
//
//  assert(cs);
//  assert(fq);
//
//  cs_nlocal(cs, nlocal);
//  cs_nlocal_offset(cs, noffset);
//
//  q0 = param->q0;
//
//  for (ic = 1; ic <= nlocal[X]; ic++) {
//    x = noffset[X] + ic;
//    for (jc = 1; jc <= nlocal[Y]; jc++) {
//      for (kc = 1; kc <= nlocal[Z]; kc++) {
//	z = noffset[Z] + kc;
//	
//	index = cs_index(cs, ic, jc, kc);
//
//	theta = 1.0*N/2.0*atan2((1.0*z-z0),(1.0*x-x0)) + q0*(z-z0);	
//	n[X] = cos(theta);
//	n[Y] = sin(theta);
//	n[Z] = 0.0;
//
//	fe_lc_q_uniaxial(param, n, q);
//	field_tensor_set(fq, index, q);
//      }
//    }
//  }
//
//  return 0;
//}
//
///*****************************************************************************
// *
// *  blue_phase_random_q_init
// *
// *  Set a decomposition-independent random initial Q tensor
// *  based on the initial order amplitude0 and a randomly
// *  chosen director at each site.
// *
// *****************************************************************************/
//
//int blue_phase_random_q_init(cs_t * cs, fe_lc_param_t * param, field_t * fq) {
//
//  int ic, jc, kc, index;
//  int nlocal[3];
//  int seed = DEFAULT_SEED;
//
//  double n[3];            /* random director */
//  double q[3][3];         /* resulting unixial q */
//  double phase1, phase2;
//
//  double ran1, ran2;
//  noise_t * rng = NULL;
//  PI_DOUBLE(pi);
//
//  assert(fq);
//
//  cs_nlocal(cs, nlocal);
//
//  noise_create(fq->pe, cs, &rng);
//  noise_init(rng, seed);
//
//  for (ic = 1; ic <= nlocal[X]; ic++) {
//    for (jc = 1; jc <= nlocal[Y]; jc++) {
//      for (kc = 1; kc <= nlocal[Z]; kc++) {
//	    
//	index = cs_index(cs, ic, jc, kc);
//
//	noise_uniform_double_reap(rng, index, &ran1);
//	noise_uniform_double_reap(rng, index, &ran2);
//
//	phase1 = 2.0*pi*(0.5 - ran1);
//	phase2 = acos(2.0*ran2 - 1.0);
//
//	n[X] = cos(phase1)*sin(phase2);
//	n[Y] = sin(phase1)*sin(phase2);
//	n[Z] = cos(phase2);
//
//	fe_lc_q_uniaxial(param, n, q);
//	field_tensor_set(fq, index, q);
//      }
//    }
//  }
//
//  noise_free(rng);
//
//  return 0;
//}
//
///*****************************************************************************
// *
// *  blue_phase_random_q_2d
// *
// *  Set a decomposition-independent random initial Q tensor in
// *  two dimensions (x,y).
// *
// *****************************************************************************/
//
//int blue_phase_random_q_2d(cs_t * cs, fe_lc_param_t * param, field_t * fq) {
//
//  int seed = DEFAULT_SEED;
//  int nlocal[3] = {0};
//
//  noise_t * rng = NULL;
//  PI_DOUBLE(pi);
//
//  assert(fq);
//
//  cs_nlocal(cs, nlocal);
//
//  noise_create(fq->pe, cs, &rng);
//  noise_init(rng, seed);
//
//  for (int ic = 1; ic <= nlocal[X]; ic++) {
//    for (int jc = 1; jc <= nlocal[Y]; jc++) {
//      for (int kc = 1; kc <= nlocal[Z]; kc++) {
//
//	int index = cs_index(cs, ic, jc, kc);
//	double ran1 = 0.0;
//	double phase1 = 0.0;
//	double n[3] = {0};
//	double q[3][3] = {0};
//
//	noise_uniform_double_reap(rng, index, &ran1);
//
//	phase1 = 2.0*pi*(0.5 - ran1);
//
//	n[X] = cos(phase1);
//	n[Y] = sin(phase1);
//	n[Z] = 0.0;
//
//	fe_lc_q_uniaxial(param, n, q);
//	field_tensor_set(fq, index, q);
//      }
//    }
//  }
//
//  noise_free(rng);
//
//  return 0;
//}
//
///*****************************************************************************
// *
// *  blue_phase_random_q_rectangle
// *
// *  Within the limits of the rectanular box provided, set the initial
// *  Q tensor to a random value. The idea here is to 'melt' the order
// *  set previously (e.g., to cholesteric), but only in a small region.
// *
// *  We should then have amplitude of order a0 << amplitude0; we use
// *  a0 = 1.0e-06. 
// * 
// *****************************************************************************/
//
//int blue_phase_random_q_rectangle(cs_t * cs, fe_lc_param_t * param,
//				  field_t * fq, int rmin[3],
//				  int rmax[3]) {
//
//  int ic, jc, kc, index;
//  int ia, ib;
//  int nlocal[3];
//  int noffset[3];
//  int seed = DEFAULT_SEED;
//
//  double n[3];
//  double q[3][3];
//  double phase1, phase2;
//  double a0 = 0.01;             /* Initial amplitude of order in 'box' */
//
//  double ran1, ran2;
//
//  noise_t * rng = NULL;
//  PI_DOUBLE(pi);
//  KRONECKER_DELTA_CHAR(d);
//
//  assert(cs);
//  assert(fq);
//
//  cs_nlocal(cs, nlocal);
//  cs_nlocal_offset(cs, noffset);
//
//  noise_create(fq->pe, cs, &rng);
//  noise_init(rng, seed);
//
//  /* Adjust min, max to allow for parallel offset of box */
//
//  rmin[X] -= noffset[X]; rmax[X] -= noffset[X];
//  rmin[Y] -= noffset[Y]; rmax[Y] -= noffset[Y];
//  rmin[Z] -= noffset[Z]; rmax[Z] -= noffset[Z];
//
//  for (ic = 1; ic <= nlocal[X]; ic++) {
//    for (jc = 1; jc <= nlocal[Y]; jc++) {
//      for (kc = 1; kc <= nlocal[Z]; kc++) {
//
//	if (ic < rmin[X] || ic > rmax[X] || jc < rmin[Y] || jc > rmax[Y] ||
//	    kc < rmin[Z] || kc > rmax[Z]) continue;
//
//	index = cs_index(cs, ic, jc, kc);
//
//	noise_uniform_double_reap(rng, index, &ran1);
//	noise_uniform_double_reap(rng, index, &ran2);
//
//	phase1 = 2.0*pi*(0.5 - ran1);
//	phase2 = acos(2.0*ran2 - 1.0);
//	    
//	n[X] = cos(phase1)*sin(phase2);
//	n[Y] = sin(phase1)*sin(phase2);
//	n[Z] = cos(phase2);
//
//	/* Uniaxial approximation using a0 */
//	for (ia = 0; ia < 3; ia++) {
//	  for (ib = 0; ib < 3; ib++) {
//	    q[ia][ib] = 0.5*a0*(3.0*n[ia]*n[ib] - d[ia][ib]);
//	  }
//	}
//	field_tensor_set(fq, index, q);
//
//      }
//    }
//  }
//
//  noise_free(rng);
//
//  return 0;
//}
//
//
///****************************************************************************
// *
// *  M_rot
// *
// *  Matrix for rotation around specified axis
// *
// ****************************************************************************/
//
//void blue_phase_M_rot(double M[3][3], int dim, double alpha){
//
//  assert(dim == X || dim == Y || dim == Z);
//
//  if(dim==0){
//    M[0][0] = 1.0;
//    M[0][1] = 0.0;
//    M[0][2] = 0.0;
//    M[1][0] = 0.0;
//    M[1][1] = cos(alpha);
//    M[1][2] = -sin(alpha);
//    M[2][0] = 0.0;
//    M[2][1] = sin(alpha);
//    M[2][2] = cos(alpha);
//  }
//
//  if(dim==1){
//    M[0][0] = cos(alpha);
//    M[0][1] = 0.0;
//    M[0][2] = -sin(alpha);
//    M[1][0] = 0.0;
//    M[1][1] = 1.0;
//    M[1][2] = 0.0;
//    M[2][0] = sin(alpha);
//    M[2][1] = 0.0;
//    M[2][2] = cos(alpha);
//  }
//
//  if(dim==2){
//    M[0][0] = cos(alpha);
//    M[0][1] = -sin(alpha);
//    M[0][2] = 0.0;
//    M[1][0] = sin(alpha);
//    M[1][1] = cos(alpha);
//    M[1][2] = 0.0;
//    M[2][0] = 0.0;
//    M[2][1] = 0.0;
//    M[2][2] = 1.0;
//  }
//
//  return;
//}
//
///*****************************************************************************
// *
// *  rotation_create
// *
// *  A convenience to help compute a standard Euler-style rotation.
// *
// *  (d0, d1, d2) is a set of coordinate axes e.g., (Z, X, Z), and
// *  theta are the Euler angles in radians.
// *
// *  The sequence of rotations (Z,X,Z) follows the standard Euler angles:
// *  1) rotation around z-axis obtaining frame x'-y'-z
// *  2) rotation around x'-axis obtaining frame x'-y''-z'
// *  3) rotation around z'-axis obtaining final frame
// *
// *****************************************************************************/
//
//static int rotation_create(rotation_t * rot, int d0, int d1, int d2,
//			   const double theta[3]) {
//
//  assert(rot);
//
//  blue_phase_M_rot(rot->m0, d0, theta[0]);
//  blue_phase_M_rot(rot->m1, d1, theta[1]);
//  blue_phase_M_rot(rot->m2, d2, theta[2]);
//
//  return 0;
//}
//
///*****************************************************************************
// *
// *  rotate_inplace
// *
// *****************************************************************************/
//
//static int rotate_inplace(const rotation_t * rot, double r[3]) {
//
//  double r0[3] = {r[X], r[Y], r[Z]};
//
//  assert(rot);
//
//  for (int ik = 0; ik < 3; ik++) {
//    double rrot = 0.0;
//    for (int il = 0; il < 3; il++) {
//      for (int im = 0; im < 3; im++) {
//	for (int in = 0; in < 3; in++) {
//	  rrot += rot->m2[ik][il]*rot->m1[il][im]*rot->m0[im][in]*r0[in];
//	}
//      }
//    }
//    r[ik] = rrot;
//  }
//
//  return 0;
//}
//
//
///*****************************************************************************
// *
// *  blue_phase_cf1_init
// *
// *  Initialise a cholesteric finger of the first kind.
// *  Uses the current free energy parameters
// *     q0 (P=2pi/q0)
// *
// *  See also P. Ribiere, S. Pirkl, P. Oswald, Phys. Rev. A 44, 8198--8209 (1991). 
// *
// *****************************************************************************/
//
//int blue_phase_cf1_init(cs_t * cs, fe_lc_param_t * param, field_t * fq,
//			int axis) {
//
//  int ic, jc, kc;
//  int nlocal[3];
//  int noffset[3];
//  int ntotal[3];
//  int index;
//
//  double n[3];
//  double q[3][3];
//  double q0;
//  double alpha, alpha0, beta;
//  PI_DOUBLE(pi);
//
//  assert(fq);
//  assert(axis == X || axis == Y || axis == Z);
//
//  cs_nlocal(cs, nlocal);
//  cs_nlocal_offset(cs, noffset);
//  cs_ntotal(cs, ntotal);
//
//  q0 = param->q0;
//  alpha0 = 0.5*pi; 
//
//  n[X] = 0.0;
//  n[Y] = 0.0;
//  n[Z] = 0.0;
// 
//  for (ic = 1; ic <= nlocal[X]; ic++) {
//    for (jc = 1; jc <= nlocal[Y]; jc++) {
//      for (kc = 1; kc <= nlocal[Z]; kc++) {
//
//	index = cs_index(cs, ic, jc, kc);
//
//	if (axis == X) {
//
//	    alpha = alpha0*sin(pi*(noffset[Z]+kc)/ntotal[Z]);
//	    beta  = -2.0*(pi*(noffset[Z]+kc)/ntotal[Z]-0.5*pi);
//
//	    n[X]  =  cos(beta)* sin(alpha)*sin(q0*(noffset[Y]+jc)) -
//		     cos(alpha)*sin(beta)*sin(alpha)*cos(q0*(noffset[Y]+jc)) +
//		     sin(alpha)*sin(beta)*cos(alpha);
//	    n[Y]  = -sin(beta)*sin(alpha)*sin(q0*(noffset[Y]+jc)) -
//		     cos(alpha)*cos(beta)*sin(alpha)*cos(q0*(noffset[Y]+jc)) +
//		     sin(alpha)*cos(beta)*cos(alpha);
//	    n[Z]  =  sin(alpha)*sin(alpha)*cos(q0*(noffset[Y]+jc)) +
//		     cos(alpha)*cos(alpha);
//
//	}
//
//	if (axis == Y) {
//
//	    alpha = alpha0*sin(pi*(noffset[X]+ic)/ntotal[X]);
//	    beta  = -2.0*(pi*(noffset[X]+ic)/ntotal[X]-0.5*pi);
//
//	    n[Y]  =  cos(beta)* sin(alpha)*sin(q0*(noffset[Z]+kc)) -
//		     cos(alpha)*sin(beta)*sin(alpha)*cos(q0*(noffset[Z]+kc)) +
//		     sin(alpha)*sin(beta)*cos(alpha);
//	    n[Z]  = -sin(beta)*sin(alpha)*sin(q0*(noffset[Z]+kc)) -
//		     cos(alpha)*cos(beta)*sin(alpha)*cos(q0*(noffset[Z]+kc)) +
//		     sin(alpha)*cos(beta)*cos(alpha);
//	    n[X]  =  sin(alpha)*sin(alpha)*cos(q0*(noffset[Z]+kc)) +
//		     cos(alpha)*cos(alpha);
//
//	}
//
//	if (axis == Z) {
//
//	    alpha = alpha0*sin(pi*(noffset[Y]+jc)/ntotal[Y]);
//	    beta  = -2.0*(pi*(noffset[Y]+jc)/ntotal[Y]-0.5*pi);
//
//	    n[Z]  =  cos(beta)* sin(alpha)*sin(q0*(noffset[X]+ic)) -
//		     cos(alpha)*sin(beta)*sin(alpha)*cos(q0*(noffset[X]+ic)) +
//		     sin(alpha)*sin(beta)*cos(alpha);
//	    n[X]  = -sin(beta)*sin(alpha)*sin(q0*(noffset[X]+ic)) -
//		     cos(alpha)*cos(beta)*sin(alpha)*cos(q0*(noffset[X]+ic)) +
//		     sin(alpha)*cos(beta)*cos(alpha);
//	    n[Y]  =  sin(alpha)*sin(alpha)*cos(q0*(noffset[X]+ic)) +
//		     cos(alpha)*cos(alpha);
//
//	}
//
//	fe_lc_q_uniaxial(param, n, q);
//	field_tensor_set(fq, index, q);
//      }
//    }
//  }
//
//  return 0;
//}
//
///*****************************************************************************
// *
// *  blue_phase_random_cf1_init
// *
// *  Initialise a cholesteric finger of the first kind.
// *  In addtion to blue_phase_cf1_init a traceless symmetric
// *  random fluctuation is put on the Q-tensor.
// *
// *  Uses the current free energy parameters
// *     q0 (pitch = 2pi/q0)
// *
// *  See also P. Ribiere, S. Pirkl, P. Oswald, Phys. Rev. A 44, 8198--8209 (1991). 
// *
// *****************************************************************************/
//
//int blue_phase_random_cf1_init(cs_t * cs, fe_lc_param_t * param, field_t * fq,
//			       int axis) {
//
//  int ic, jc, kc;
//  int nlocal[3];
//  int noffset[3];
//  int ntotal[3];
//  int index;
//
//  double n[3];
//  double q[3][3];
//  double q0;
//  double alpha, alpha0, beta;
//
//  int ia, ib, id;
//  double tmatrix[3][3][NQAB];
//  double chi[NQAB], chi_qab[3][3];
//  double var = 1e-1; /* variance of random fluctuation */
//  int seed = DEFAULT_SEED;
//
//  noise_t * rng = NULL;
//  PI_DOUBLE(pi);
//
//  assert(cs);
//  assert(fq);
//  assert(axis == X || axis == Y || axis == Z);
//
//  cs_nlocal(cs, nlocal);
//  cs_nlocal_offset(cs, noffset);
//  cs_ntotal(cs, ntotal);
//
//  q0 = param->q0;
//  alpha0 = 0.5*pi; 
//
//  n[X] = 0.0;
//  n[Y] = 0.0;
//  n[Z] = 0.0;
//
//  noise_create(fq->pe, cs, &rng);
//  noise_init(rng, seed);
// 
//  for (ic = 1; ic <= nlocal[X]; ic++) {
//    for (jc = 1; jc <= nlocal[Y]; jc++) {
//      for (kc = 1; kc <= nlocal[Z]; kc++) {
//
//	index = cs_index(cs, ic, jc, kc);
//
//	if (axis == X) {
//
//	    alpha = alpha0*sin(pi*(noffset[Z]+kc)/ntotal[Z]);
//	    beta  = -2.0*(pi*(noffset[Z]+kc)/ntotal[Z]-0.5*pi);
//
//	    n[X]  =  cos(beta)* sin(alpha)*sin(q0*(noffset[Y]+jc)) -
//		     cos(alpha)*sin(beta)*sin(alpha)*cos(q0*(noffset[Y]+jc)) +
//		     sin(alpha)*sin(beta)*cos(alpha);
//	    n[Y]  = -sin(beta)*sin(alpha)*sin(q0*(noffset[Y]+jc)) -
//		     cos(alpha)*cos(beta)*sin(alpha)*cos(q0*(noffset[Y]+jc)) +
//		     sin(alpha)*cos(beta)*cos(alpha);
//	    n[Z]  =  sin(alpha)*sin(alpha)*cos(q0*(noffset[Y]+jc)) +
//		     cos(alpha)*cos(alpha);
//
//	}
//
//	if (axis == Y) {
//
//	    alpha = alpha0*sin(pi*(noffset[X]+ic)/ntotal[X]);
//	    beta  = -2.0*(pi*(noffset[X]+ic)/ntotal[X]-0.5*pi);
//
//	    n[Y]  =  cos(beta)* sin(alpha)*sin(q0*(noffset[Z]+kc)) -
//		     cos(alpha)*sin(beta)*sin(alpha)*cos(q0*(noffset[Z]+kc)) +
//		     sin(alpha)*sin(beta)*cos(alpha);
//	    n[Z]  = -sin(beta)*sin(alpha)*sin(q0*(noffset[Z]+kc)) -
//		     cos(alpha)*cos(beta)*sin(alpha)*cos(q0*(noffset[Z]+kc)) +
//		     sin(alpha)*cos(beta)*cos(alpha);
//	    n[X]  =  sin(alpha)*sin(alpha)*cos(q0*(noffset[Z]+kc)) +
//		     cos(alpha)*cos(alpha);
//
//	}
//
//	if (axis == Z) {
//
//	    alpha = alpha0*sin(pi*(noffset[Y]+jc)/ntotal[Y]);
//	    beta  = -2.0*(pi*(noffset[Y]+jc)/ntotal[Y]-0.5*pi);
//
//	    n[Z]  =  cos(beta)* sin(alpha)*sin(q0*(noffset[X]+ic)) -
//		     cos(alpha)*sin(beta)*sin(alpha)*cos(q0*(noffset[X]+ic)) +
//		     sin(alpha)*sin(beta)*cos(alpha);
//	    n[X]  = -sin(beta)*sin(alpha)*sin(q0*(noffset[X]+ic)) -
//		     cos(alpha)*cos(beta)*sin(alpha)*cos(q0*(noffset[X]+ic)) +
//		     sin(alpha)*cos(beta)*cos(alpha);
//	    n[Y]  =  sin(alpha)*sin(alpha)*cos(q0*(noffset[X]+ic)) +
//		     cos(alpha)*cos(alpha);
//
//	}
//
//	fe_lc_q_uniaxial(param, n, q);
//
//	/* Random fluctuation with specified variance */
//        noise_reap_n(rng, index, NQAB, chi);
//
//	for (id = 0; id < NQAB; id++) {
//	  chi[id] = var*chi[id];
//	}
//
//	beris_edw_tmatrix(tmatrix);
//
//	/* Random fluctuation added to tensor order parameter */
//	for (ia = 0; ia < 3; ia++) {
//	  for (ib = 0; ib < 3; ib++) {
//	    chi_qab[ia][ib] = 0.0;
//	    for (id = 0; id < NQAB; id++) {
//	      chi_qab[ia][ib] += chi[id]*tmatrix[ia][ib][id];
//	    }
//	    q[ia][ib] += chi_qab[ia][ib];
//	  }
//	}
//
//	field_tensor_set(fq, index, q);
//
//      }
//    }
//  }
//
//  noise_free(rng);
//
//  return 0;
//}
