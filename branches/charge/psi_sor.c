/*****************************************************************************
 *
 *  psi_sor.c
 *
 *  A solution of the Poisson equation
 *
 *  via a simple SOR method following Press et al.
 *
 *****************************************************************************/

#include <assert.h>
#include <float.h>
#include <math.h>
#include <mpi.h>

#include "pe.h"
#include "coords.h"
#include "psi_s.h"
#include "psi_sor.h"

/*****************************************************************************
 *
 *  psi_sor_poisson
 *
 *
 *  First attempt. Uniform permeativity. The differencing is a seven
 *  point stencil for \nabla^2 \psi. So
 *
 *  epsilon [ psi(i+1,j,k) - 2 psi(i,j,k) + psi(i-1,j,k)
 *          + psi(i,j+1,k) - 2 psi(i,j,k) + psi(i,j-1,k)
 *          + psi(i,j,k+1) - 2 psi(i,j,k) + psi(i,j,k-1) ] = rho_elec(i,j,k)
 *
 *****************************************************************************/

int psi_sor_poisson(psi_t * obj) {

  int ic, jc, kc, index;
  int nhalo;
  int n, niteration;           /* Relaxation iterations */
  int pass;                    /* Red/black iteration */
  int kst;                     /* Start kc index for red/black iteration */
  int nlocal[3];
  int xs, ys, zs;              /* Memory strides */

  double rho_elec;             /* Right-hand side */
  double residual;             /* Residual at given point */
  double rnorm[2];             /* Initial and current norm of residual */
  double rnorm_local[2];       /* Local values */

  double epsilon;              /* Uniform permeativity */
  double dpsi;

  double omega;                /* Over-relaxation parameter 1 < omega < 2 */
  double radius;               /* Spectral radius of Jacobi iteration */
  double tol_abs;              /* Absolute tolerance */
  double tol_rel;              /* Relative tolerance */

  MPI_Comm comm;               /* Cartesian communicator */

  nhalo = coords_nhalo();
  coords_nlocal(nlocal);
  comm = cart_comm();

  assert(nhalo >= 1);

  zs = 1;
  ys = zs*(nlocal[Z] + 2*nhalo);
  xs = ys*(nlocal[Y] + 2*nhalo);

  /* Compute initial norm of the residual */

  niteration = 1000;
  epsilon = 1.0;
  rnorm_local[0] = 0.0;

  for (ic = 1; ic <= nlocal[X]; ic++) {
    for (jc = 1; jc <= nlocal[Y]; jc++) {
      for (kc = 1; kc <= nlocal[Z]; kc++) {

	index = coords_index(ic, jc, kc);

	psi_rho_elec(obj, index, &rho_elec);
	dpsi = obj->psi[index + xs] + obj->psi[index - xs]
	     + obj->psi[index + ys] + obj->psi[index - ys]
	     + obj->psi[index + zs] + obj->psi[index - zs]
	     - 6.0*obj->psi[index]; 
	rnorm_local[0] += fabs(epsilon*dpsi - rho_elec);
      }
    }
  }

  /* Iterate to solution */

  omega = 1.0;
  radius = 0.999;
  tol_abs = 10.0*DBL_EPSILON;
  tol_rel = 1.000*FLT_EPSILON;

  for (n = 0; n < niteration; n++) {

    /* Compute current normal of the residual */

    rnorm_local[1] = 0.0;

    for (pass = 0; pass < 2; pass++) {

      for (ic = 1; ic <= nlocal[X]; ic++) {
	for (jc = 1; jc <= nlocal[Y]; jc++) {
	  kst = 1 + (jc + pass) % 2;
	  for (kc = kst; kc <= nlocal[Z]; kc += 2) {

	    index = coords_index(ic, jc, kc);

	    psi_rho_elec(obj, index, &rho_elec);
	    dpsi = obj->psi[index + xs] + obj->psi[index - xs]
	         + obj->psi[index + ys] + obj->psi[index - ys]
	         + obj->psi[index + zs] + obj->psi[index - zs]
	      - 6.0*obj->psi[index];
	    residual = epsilon*dpsi - rho_elec;
	    obj->psi[index] -= omega*residual / (-6.0*epsilon);
	    rnorm_local[1] += fabs(residual);
	  }
	}
      }

      /* Recompute relation parameter and next pass */

      if (n == 0 && pass == 0) {
	omega = 1.0 / (1.0 - 0.5*radius*radius);
      }
      else {
	omega = 1.0 / (1.0 - 0.25*radius*radius*omega);
      }
      assert(1.0 < omega);
      assert(omega < 2.0);

      psi_halo_psi(obj);
    }

    /* Compare residual and iterate again if necessary */

    MPI_Allreduce(rnorm_local, rnorm, 2, MPI_DOUBLE, MPI_SUM, comm);
    printf("Iteration %4d rnorm %14.7e\n", n, rnorm[1]);
    if (rnorm[1] < tol_abs || rnorm[1] < tol_rel*rnorm[0]) break;
  }

  return 0;
}
