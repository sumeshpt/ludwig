/*****************************************************************************
 *
 *  d3q19.c
 *
 *  D3Q19 definitions and model-dependent code.
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <math.h>

#include "pe.h"
#include "ran.h"
#include "d3q19.h"

#ifdef _D3Q19_

/*****************************************************************************
 *
 *  There are 19 eigenvectors:
 *
 *  rho             (eigenvector implicitly {1})
 *  rho cv[p][X]    (x-component of velocity)
 *  rho cv[p][Y]    (y-component of velocity) 
 *  rho cv[p][Z]    (z-component of velocity)
 *  q[p][X][X]      (xx component of deviatoric stress)
 *  q[p][X][Y]      (xy component of deviatoric stress)
 *  q[p][X][Z]      (xz ...
 *  q[p][Y][Y]      (yy ...
 *  q[p][Y][Z]      (yz ...
 *  q[p][Z][Z]      (zz ...
 *  chi1[p]         (1st ghost mode)
 *  chi2[p]         (2nd ghost mode)
 *  jchi1[p][X]     (x-component of ghost current chi1[p]*rho*cv[p][X])
 *  jchi1[p][Y]     (y-component of ghost current chi1[p]*rho*cv[p][Y])
 *  jchi1[p][Z]     (z-component of ghost current chi1[p]*rho*cv[p][Z])
 *  jchi2[p][X]     (x-component of ghost current chi2[p]*rho*cv[p][X])
 *  jchi2[p][Y]     (y-component of ghost current chi2[p]*rho*cv[p][Y])
 *  jchi2[p][Z]     (z-component of ghost current chi2[p]*rho*cv[p][Z])
 *  chi3[p]         (3rd ghost mode)
 *
 *  The associated quadrature weights are:
 *
 *  wv[p]
 *
 *  Note that q[p][i][j], jchi1[p][i], and jchi2[p][i] are computed
 *  at run time.
 *
 *****************************************************************************/


const int cv[NVEL][3] = {{ 0,  0,  0},
			 { 1,  1,  0}, { 1,  0,  1}, { 1,  0,  0},
			 { 1,  0, -1}, { 1, -1,  0}, { 0,  1,  1},
			 { 0,  1,  0}, { 0,  1, -1}, { 0,  0,  1},
			 { 0,  0, -1}, { 0, -1,  1}, { 0, -1,  0},
			 { 0, -1, -1}, {-1,  1,  0}, {-1,  0,  1},
			 {-1,  0,  0}, {-1,  0, -1}, {-1, -1,  0}};

const double chi1[NVEL] = {0.0, -2.0,  1.0,  1.0, 
                                 1.0, -2.0,  1.0, 
                                 1.0,  1.0, -2.0,
                                -2.0,  1.0,  1.0,
                                 1.0, -2.0,  1.0,
                                 1.0,  1.0, -2.0};

const double chi2[NVEL] = {0.0,  0.0, -1.0,  1.0,
                                -1.0,  0.0,  1.0,
                                -1.0,  1.0,  0.0,
                                 0.0,  1.0, -1.0,
                                 1.0,  0.0, -1.0,
                                 1.0, -1.0,  0.0};

const double chi3[NVEL] = {1.0,  1.0,  1.0, -2.0,
                                 1.0,  1.0,  1.0,
                                -2.0,  1.0, -2.0,
                                -2.0,  1.0, -2.0,
                                 1.0,  1.0,  1.0,
                                -2.0,  1.0,  1.0};

#define w0 (12.0/36.0)
#define w1  (2.0/36.0)
#define w2  (1.0/36.0)

const double wv[NVEL] = {w0,
			 w2, w2, w1, w2, w2, w2, w1, w2, w1,
			 w1, w2, w1, w2, w2, w2, w1, w2, w2}; 

static double jchi1[NVEL][3];
static double jchi2[NVEL][3];

static double var_chi1;
static double var_jchi1;
static double var_chi2;
static double var_jchi2;
static double var_chi3;

/*****************************************************************************
 *
 *  init_ghosts
 *
 *  Initialise the D3Q15 ghost modes jchi1 and jchi2.
 *
 *****************************************************************************/

void init_ghosts(const double kT) {

  double tau_ghost = 1.0;
  double var;
  int    i, p;

  /* Set the ghost current eigenvectors jchi1 */

  for (p = 0; p < NVEL; p++) {
    for (i = 0; i < 3; i++) {
      jchi1[p][i] = chi1[p]*cv[p][i];
      jchi2[p][i] = chi2[p]*cv[p][i];
    }
  }

  /* These are the variances for fluctuations in ghosts */

  var = sqrt((tau_ghost + tau_ghost - 1.0)/(tau_ghost*tau_ghost));

  var_chi1  = sqrt(kT)*sqrt(4.0/3.0)*var;
  var_jchi1 = sqrt(kT)*sqrt(2.0/3.0)*var;
  var_chi2  = sqrt(kT)*sqrt(4.0/9.0)*var;
  var_jchi2 = sqrt(kT)*sqrt(2.0/9.0)*var;
  var_chi3 =  sqrt(kT)*sqrt(2.0/1.0)*var;

  return;
}

/*****************************************************************************
 *
 *  get_ghosts
 *
 *  Work out the model-dependent part of the distribution for the
 *  collision stage.
 *
 *****************************************************************************/

void get_ghosts(double fghost[]) {

  const double c3r4 = (3.0/4.0);   /* Normaliser for chi1 mode */
  const double c3r2 = (3.0/2.0);   /* Normaliser for jchi1 mode */
  const double c3   = (9.0/4.0);   /* Normaliser for chi2 mode */
  const double c9r2 = (9.0/2.0);   /* Normaliser for jchi2 mode */
  const double r2   = (1.0/2.0);   /* Normaliser for chi3 mode */

  double chi1hat;
  double chi2hat;
  double chi3hat;
  double jchi1hat[3];
  double jchi2hat[3];

  int    i, p;

  /* Set fluctuating parts and and tot up the ghost projection. */

  chi1hat      = ran_parallel_gaussian()*var_chi1;
  jchi1hat[0]  = ran_parallel_gaussian()*var_jchi1;
  jchi1hat[1]  = ran_parallel_gaussian()*var_jchi1;
  jchi1hat[2]  = ran_parallel_gaussian()*var_jchi1;
  chi2hat      = ran_parallel_gaussian()*var_chi2;
  jchi2hat[0]  = ran_parallel_gaussian()*var_jchi2;
  jchi2hat[1]  = ran_parallel_gaussian()*var_jchi2;
  jchi2hat[2]  = ran_parallel_gaussian()*var_jchi2;
  chi3hat      = ran_parallel_gaussian()*var_chi3;

  for (p = 0; p < NVEL; p++) {

    fghost[p] = c3r4*chi1hat*chi1[p];

    for (i = 0; i < 3; i++) {
      fghost[p] += c3r2*jchi1hat[i]*jchi1[p][i];
      fghost[p] += c9r2*jchi2hat[i]*jchi2[p][i];
    }

    fghost[p] += c3*chi2[p]*chi2hat;
    fghost[p] += r2*chi3[p]*chi3hat;
  }

  return;
}


#endif
