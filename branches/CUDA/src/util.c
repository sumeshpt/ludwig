/*****************************************************************************
 *
 *  util.c
 *
 *  Utility functions, including vectors.
 *
 *  Little / big endian stuff based on suggestions by Harsha S.
 *  Adiga from IBM.
 *
 *  $Id$
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *  (c) 2010 The University of Edinburgh
 *
 *****************************************************************************/

#include <assert.h>
#include <math.h>
#include <float.h>
#include <stdlib.h>

#include "pe.h"
#include "coords.h"
#include "util.h"

#define c0 0.0
#define c1 1.0

const double pi_ = 3.1415926535897932385;
const double r3_ = (1.0/3.0);
const double d_[3][3]    = {{c1, c0, c0}, {c0, c1, c0}, {c0, c0, c1}};
const double e_[3][3][3] = {{{c0, c0, c0}, { c0, c0, c1}, {c0,-c1, c0}},
			    {{c0, c0,-c1}, { c0, c0, c0}, {c1, c0, c0}},
			    {{c0, c1, c0}, {-c1, c0, c0}, {c0, c0, c0}}}; 

static void util_swap(int ia, int ib, double a[3], double b[3][3]);

/***************************************************************************
 *
 *  is_bigendian
 *
 *  Byte order for this 4-byte int is 00 00 00 01 for big endian (most
 *  significant byte stored first).
 *
 ***************************************************************************/

int is_bigendian() {

  const int i = 1;

  return (*(char *) &i == 0);
}

/****************************************************************************
 *
 *  reverse_byte_order_double
 *
 *  Reverse the bytes in the char argument to make a double.
 *
 *****************************************************************************/

double reverse_byte_order_double(char * c) {

  double result;
  char * p = (char *) &result;
  int b;

  for (b = 0; b < sizeof(double); b++) {
    p[b] = c[sizeof(double) - (b + 1)];
  }

  return result;
}

/*****************************************************************************
 *
 *  dot_product
 *
 *****************************************************************************/

double dot_product(const double a[3], const double b[3]) {

  return (a[X]*b[X] + a[Y]*b[Y] + a[Z]*b[Z]);
}

/*****************************************************************************
 *
 *  cross_product
 *
 *****************************************************************************/

void cross_product(const double a[3], const double b[3], double result[3]) {

  result[X] = a[Y]*b[Z] - a[Z]*b[Y];
  result[Y] = a[Z]*b[X] - a[X]*b[Z];
  result[Z] = a[X]*b[Y] - a[Y]*b[X];

  return;
}

/*****************************************************************************
 *
 *  modulus
 *
 *****************************************************************************/

double modulus(const double a[3]) {

  return sqrt(a[X]*a[X] + a[Y]*a[Y] + a[Z]*a[Z]);
}

/*****************************************************************************
 *
 *  rotate_vector
 *
 *  Rotate the vector v around the unit axis of rotation \hat{w}
 *  by an angle of \theta, where \theta = |w|. (For example, w
 *  might be an angular velocity.)
 *
 *  The rotated vector is computed via
 *      v' = (1 - cos \theta)(\hat{w}.v) \hat{w} + cos \theta v +
 *           (\hat{w} x v) sin \theta      
 *
 *  For theta positive this gives rotations in the correct sense
 *  in the right-handed coordinate system.
 *
 ****************************************************************************/

void rotate_vector(double v[3], const double w[3]) {

  double what[3], vrot[3];
  double theta, ct, st;
  double vdotw;

  theta = sqrt(w[X]*w[X] + w[Y]*w[Y] + w[Z]*w[Z]);

  if (theta == 0.0) {
    /* There is no rotation. */
   }
  else {
    /* Work out the unit axis of rotation */

    what[X] = w[X] / theta;
    what[Y] = w[Y] / theta;
    what[Z] = w[Z] / theta;

    /* Rotation */

    st = sin(theta);
    ct = cos(theta);
    vdotw = v[X]*what[X] + v[Y]*what[Y] + v[Z]*what[Z];

    vrot[X] = ct*v[X] + st*(what[Y]*v[Z] - what[Z]*v[Y]);
    vrot[Y] = ct*v[Y] + st*(what[Z]*v[X] - what[X]*v[Z]);
    vrot[Z] = ct*v[Z] + st*(what[X]*v[Y] - what[Y]*v[X]);
    v[X] = (1.0 - ct)*vdotw*what[X] + vrot[X];
    v[Y] = (1.0 - ct)*vdotw*what[Y] + vrot[Y];
    v[Z] = (1.0 - ct)*vdotw*what[Z] + vrot[Z];
  }

  return;
}

/*****************************************************************************
 *
 *  imin, imax, dmin, dmax
 *
 *  minimax functions
 *
 *****************************************************************************/

int imin(const int i, const int j) {
  return ((i < j) ? i : j);
}

int imax(const int i, const int j) {
  return ((i > j) ? i : j);
}

double dmin(const double a, const double b) {
  return ((a < b) ? a : b);
}

double dmax(const double a, const double b) {
  return ((a > b) ? a : b);
}

/*****************************************************************************
 *
 *  util_jacobi_sort
 *
 *  Returns sorted eigenvalues and eigenvectors, highest eigenvalue first.
 *
 *  Returns zero on success.
 *
 *****************************************************************************/

int util_jacobi_sort(double a[3][3], double vals[3], double vecs[3][3]) {

  int ifail;

  ifail = util_jacobi(a, vals, vecs);

  /* And sort */

  if (vals[X] < vals[Y]) util_swap(X, Y, vals, vecs);
  if (vals[X] < vals[Z]) util_swap(X, Z, vals, vecs);
  if (vals[Y] < vals[Z]) util_swap(Y, Z, vals, vecs);

  return ifail;
}

/*****************************************************************************
 *
 *  util_jacobi
 *
 *  Find the eigenvalues and eigenvectors of a 3x3 symmetric matrix a.
 *  This routine from Press et al. (page 467). The eigenvectors are
 *  returned as the columns of vecs[nrow][ncol].
 *
 *  Returns 0 on success. Garbage out usually means garbage in!
 *
 *****************************************************************************/

int util_jacobi(double a[3][3], double vals[3], double vecs[3][3]) {

  int iterate, ia, ib, ic;
  double tresh, theta, tau, t, sum, s, h, g, c;
  double b[3], z[3];

  const int maxjacobi = 50;    /* Maximum number of iterations */

  for (ia = 0; ia < 3; ia++) {
    for (ib = 0; ib < 3; ib++) {
      vecs[ia][ib] = d_[ia][ib];
    }
    vals[ia] = a[ia][ia];
    b[ia] = a[ia][ia];
    z[ia] = 0.0;
  }

  for (iterate = 1; iterate <= maxjacobi; iterate++) {
    sum = 0.0;

    for (ia = 0; ia < 2; ia++) {
      for (ib = ia + 1; ib < 3; ib++) {
	sum += fabs(a[ia][ib]);
      }
    }

    if (sum < DBL_MIN) return 0;

    if (iterate < 4)
      tresh = 0.2*sum/(3*3);
    else
      tresh = 0.0;

    for (ia = 0; ia < 2; ia++) {
      for (ib = ia + 1; ib < 3; ib++) {

	g = 100.0*fabs(a[ia][ib]);

	if (iterate > 4 && (fabs(vals[ia]) + g) == fabs(vals[ia]) &&
	    (fabs(vals[ib]) + g) == fabs(vals[ib])) {
	  a[ia][ib] = 0.0;
	}
	else if (fabs(a[ia][ib]) > tresh) {
	  h = vals[ib] - vals[ia];
	  if ((fabs(h) + g) == fabs(h)) {
	    t = (a[ia][ib])/h;
	  }
	  else {
	    theta = 0.5*h/a[ia][ib];
	    t = 1.0/(fabs(theta) + sqrt(1.0 + theta*theta));
	    if (theta < 0.0) t = -t;
	  }

	  c = 1.0/sqrt(1 + t*t);
	  s = t*c;
	  tau = s/(1.0 + c);
	  h = t*a[ia][ib];
	  z[ia] -= h;
	  z[ib] += h;
	  vals[ia] -= h;
	  vals[ib] += h;
	  a[ia][ib] = 0.0;

	  for (ic = 0; ic <= ia - 1; ic++) {
	    assert(ic < 3);
	    g = a[ic][ia];
	    h = a[ic][ib];
	    a[ic][ia] = g - s*(h + g*tau);
	    a[ic][ib] = h + s*(g - h*tau);
	  }
	  for (ic = ia + 1; ic <= ib - 1; ic++) {
	    assert(ic < 3);
	    g = a[ia][ic];
	    h = a[ic][ib];
	    a[ia][ic] = g - s*(h + g*tau);
	    a[ic][ib] = h + s*(g - h*tau);
	  }
	  for (ic = ib + 1; ic < 3; ic++) {
	    g = a[ia][ic];
	    h = a[ib][ic];
	    a[ia][ic] = g - s*(h + g*tau);
	    a[ib][ic] = h + s*(g - h*tau);
	  }
	  for (ic = 0; ic < 3; ic++) {
	    g = vecs[ic][ia];
	    h = vecs[ic][ib];
	    vecs[ic][ia] = g - s*(h + g*tau);
	    vecs[ic][ib] = h + s*(g - h*tau);
	  }
	}
      }
    }

    for (ia = 0; ia < 3; ia++) {
      b[ia] += z[ia];
      vals[ia] = b[ia];
      z[ia] = 0.0;
    }
  }

  return -1;
}

/*****************************************************************************
 *
 *  util_swap
 *
 *  Intended for a[3] eigenvalues and b[nrow][ncol] column eigenvectors.
 *
 *****************************************************************************/

static void util_swap(int ia, int ib, double a[3], double b[3][3]) {

  int ic;
  double tmp;

  tmp = a[ia];
  a[ia] = a[ib];
  a[ib] = tmp;

  for (ic = 0; ic < 3; ic++) {
    tmp = b[ic][ia];
    b[ic][ia] = b[ic][ib];
    b[ic][ib] = tmp;
  }

  return;
}

/*****************************************************************************
 *
 *  util_matrix_create
 *
 *****************************************************************************/

int util_matrix_create(int m, int n, double *** p) {

  int ifail = 0;
  int i;
  double ** matrix = NULL;

  matrix = calloc(m, sizeof(double *));
  if (matrix == NULL) return -1;

  for (i = 0; i < m; i++) {
    matrix[i] = calloc(n, sizeof(double));
    if (matrix[i] == NULL) ifail += 1;
  }

  *p = matrix;

  return ifail;
}

/*****************************************************************************
 *
 *  util_matrix_free
 *
 *****************************************************************************/

int util_matrix_free(int m, double *** p) {

  int i;
  assert(p);

  for (i = 0; i < m; i++) {
    free((*p)[i]);
  }
  free(*p);
  *p = NULL;
 
  return 0;
}

/*****************************************************************************
 *
 *  util_matrix_invert
 *
 *  For n x n matrix, compute and return inverse. This is the same
 *  as the Gauss Jordan routine, but we don't bother with a RHS.
 *
 *  This is done in place.
 *
 *****************************************************************************/

int util_matrix_invert(int n, double ** a) {

  int i, j, k, ia, ib;
  int irow, icol;

  int * indexcol = NULL;
  int * indexrow = NULL;
  int * ipivot = NULL;

  double rpivot, tmp;

  assert(a);

  indexcol = calloc(n, sizeof(int));
  indexrow = calloc(n, sizeof(int));
  ipivot = calloc(n, sizeof(int));

  if (indexcol == NULL) return -3;
  if (indexrow == NULL) return -3;
  if (ipivot == NULL) return -3;

  icol = -1;
  irow = -1;

  for (j = 0; j < n; j++) {
    ipivot[j] = -1;
  }

  for (i = 0; i < n; i++) {
    tmp = 0.0;
    for (j = 0; j < n; j++) {
      if (ipivot[j] != 0) {
	for (k = 0; k < n; k++) {

	  if (ipivot[k] == -1) {
	    if (fabs(a[j][k]) >= tmp) {
	      tmp = fabs(a[j][k]);
	      irow = j;
	      icol = k;
	    }
	  }
	}
      }
    }

    assert(icol != -1);
    assert(irow != -1);

    ipivot[icol] += 1;

    if (irow != icol) {
      for (ia = 0; ia < n; ia++) {
	tmp = a[irow][ia];
	a[irow][ia] = a[icol][ia];
	a[icol][ia] = tmp;
      }
    }

    indexrow[i] = irow;
    indexcol[i] = icol;

    if (a[icol][icol] == 0.0) {
      free(ipivot);
      free(indexrow);
      free(indexcol);
      return -1;
    }

    rpivot = 1.0/a[icol][icol];
    a[icol][icol] = 1.0;

    for (ia = 0; ia < n; ia++) {
      a[icol][ia] *= rpivot;
    }

    for (ia = 0; ia < n; ia++) {
      if (ia != icol) {
	tmp = a[ia][icol];
	a[ia][icol] = 0.0;
	for (ib = 0; ib < n; ib++) {
	  a[ia][ib] -= a[icol][ib]*tmp;
	}
      }
    }
  }

  /* Recover the inverse. */

  for (i = n - 1; i >= 0; i--) {
    if (indexrow[i] != indexcol[i]) {
      for (j = 0; j < n; j++) {
	tmp = a[j][indexrow[i]];
	a[j][indexrow[i]] = a[j][indexcol[i]];
	a[j][indexcol[i]] = tmp;
      }
    }
  }

  free(ipivot);
  free(indexrow);
  free(indexcol);

  return 0;
}
