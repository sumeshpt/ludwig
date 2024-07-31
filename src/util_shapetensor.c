/*****************************************************************************
 *
 *  util_shapetensor.c
 *
 *  Some shape tensor operations.
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2023 The University of Edinburgh
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>
#include <math.h>
#include <stdio.h>

#include "util.h"
#include "util_shapetensor.h"

/*****************************************************************************
 *
 *  shapetensor_polyhedron
 *
 *****************************************************************************/

int shapetensor_polyhedron(const int n, double stini[3][3]) {

  assert(n > 0);
  assert(stini);
  PI_DOUBLE(pi);

  int ia;

  double theta = 2*pi/n;
  double area = 1.0;
  double volume = 1.0;
  double height = volume/area;

  double l;
  int ifail = -1;
  double eigenvalue[3];
  double eigenvector[3][3];

  double x1[3] = {0.0, 0.0, 0.0};
  double x2[3] = {0.0, 0.0, 0.0};
  double x3[3], s[3], snew[3];
  double x1h[3], x2h[3];
  copyvecvector(x1,x1h);
  x1h[2]+=height;
  double sout[3][3];
  double rout[3][3] = {0};
  /*Length of the side*/
  l = length_polygon(area, n);
  x2[0] = l;
  for(ia = 0; ia < n; ia++) {
    /*Strand on the polygon, face 1*/
    subvecvector(x1, x2, s);
    outervecvec(s, s, sout); 
    addplustensor((const double (*)[])sout, rout);
    rotatevector(theta, s, snew); 
    addvecvector(x2, snew, x3);
    /*Strand on the polygon, face 2*/
    copyvecvector(x2,x2h);
    x2h[2]+=height;
    subvecvector(x1h, x2h, s);
    outervecvec(s, s, sout); 
    addplustensor((const double (*)[])sout, rout);
    /*Strand on the sides*/
    subvecvector(x2, x2h, s);
    outervecvec(s, s, sout); 
    addplustensor((const double (*)[])sout, rout);
    /*Renaming to facilitate the loop*/
    copyvecvector(x2,x1);
    copyvecvector(x3,x2);
    copyvecvector(x2h,x1h);
  }
  dividetensscalar(rout,(double) n*3);
  ifail = util_jacobi_sort(rout, eigenvalue, eigenvector);
  for(int ia = 0; ia < 3; ia++) {
    for(int ib = 0; ib < 3; ib++) {
      stini[ia][ib] = rout[ia][ib];
    }
  }

  return ifail;
}

/*****************************************************************************
 *
 *  shapetensor_polygon
 *
 *****************************************************************************/

int shapetensor_polygon(const int n, double *lambda1) {

  assert(n > 0);
  assert(lambda1);
  PI_DOUBLE(pi);

  int ia;

  double theta = 2*pi/n;
  double area = 1.0;

  double l;
  int ifail = -1;
  double eigenvalue[3];
  double eigenvector[3][3];

  double x1[3] = {0.0, 0.0, 0.0};
  double x2[3] = {0.0, 0.0, 0.0};
  double x3[3], s[3], snew[3];
  double sout[3][3];
  double rout[3][3] = {0};
  l = length_polygon(area, n);
  x2[0] = l;
  for(ia = 0; ia < n; ia++) {
    subvecvector(x1, x2, s);
    outervecvec(s, s, sout); 
    addplustensor((const double (*)[])sout, rout);
    rotatevector(theta, s, snew); 
    addvecvector(x2, snew, x3);
    copyvecvector(x2,x1);
    copyvecvector(x3,x2);
  }
  dividetensscalar(rout,(double) n);
  ifail = util_jacobi_sort(rout, eigenvalue, eigenvector);
  *lambda1 = eigenvalue[0];

  return ifail;
}

/*****************************************************************************
 *
 *  length of side of a polygon of a given area
 *
 *****************************************************************************/

double length_polygon(const double area, const int n) {

  assert(n > 0);
  assert(area);

  PI_DOUBLE(pi);
  double l = 0.0;
 
  switch (n) {
    case 4:
      l = sqrt(area);
      break;
    case 5:
      l = (sqrt(4.0*area/sqrt(5.0*(5.0+2.0*sqrt(5.0)))));
      break;
    case 6:
      l = sqrt(2.0*area/(3.0*sqrt(3.0))); 
      break;
    case 7:
      l = sqrt(4.0*area/7.0*(tan(pi/7.0)));
      break;
    default:
      printf("No default case!.");
      break;
  }

  return l;
}

/*****************************************************************************
 *
 *  Rotating a vector by an angle 
 *
 *****************************************************************************/

void rotatevector(const double theta, double a[3], double *b) {

  double m[3][3];

  /*Create rotation vector*/
  m[0][0] = cos(theta);
  m[0][1] =-sin(theta);
  m[0][2] = 0.0;
  m[1][0] = sin(theta);
  m[1][1] = cos(theta);
  m[1][2] = 0.0;
  m[2][0] = 0.0;
  m[2][1] = 0.0;
  m[2][2] = 1.0;

  mulmxvector(m,a,b);

  return;
}

/*****************************************************************************
 *
 *  Multiplying a matrix by a vector
 *
 *****************************************************************************/

void mulmxvector(const double m[3][3], const double a[3], double *b) {
 
  int ia, ib;
  for(ia = 0; ia < 3; ia++){
    b[ia] = 0.0;
    for(ib = 0; ib < 3; ib++) {
      b[ia] += m[ia][ib]*a[ib];
    }
  }

  return;
}

/*****************************************************************************
 *
 *  Subtracting a vector from another vector
 *
 *****************************************************************************/

void subvecvector(const double a[3], const double b[3], double *c) {
 
  int ia;
  for(ia = 0; ia < 3; ia++){
    c[ia] = b[ia] - a[ia];
  }

  return;
}

/*****************************************************************************
 *
 *  Adding a vector to another vector
 *
 *****************************************************************************/

void addvecvector(const double a[3], const double b[3], double *c) {
 
  int ia;
  for(ia = 0; ia < 3; ia++){
    c[ia] = b[ia] + a[ia];
  }

  return;
}

/*****************************************************************************
 *
 *  Outer product of a vector with another vector
 *
 *****************************************************************************/

void outervecvec(const double a[3], const double b[3], double c[3][3]) {
 
  int ia, ib;
  for(ia = 0; ia < 3; ia++){
    for(ib = 0; ib < 3; ib++){
      c[ia][ib] = a[ia]*b[ib];
    }
  }

  return;
}

/*****************************************************************************
 *
 *  Adding a tensor to another tensor
 *
 *****************************************************************************/

void addplustensor(const double a[3][3], double c[3][3]) {
 
  int ia,ib;
  for(ia = 0; ia < 3; ia++){
    for(ib = 0; ib < 3; ib++){
      c[ia][ib] += a[ia][ib];
    }
  }

  return;
}

/*****************************************************************************
 *
 *  Copying a vector to another vector
 *
 *****************************************************************************/

void copyvecvector(const double a[3], double *c) {
 
  int ia;
  for(ia = 0; ia < 3; ia++){
    c[ia] = a[ia];
  }

  return;
}

/*****************************************************************************
 *
 *  Dividing a tensor by a scalar
 *
 *****************************************************************************/

void dividetensscalar(double a[3][3], double c) {
 
  int ia, ib;
  for(ia = 0; ia < 3; ia++){
    for(ib = 0; ib < 3; ib++){
      a[ia][ib] = a[ia][ib]/c;
    }
  }

  return;
}

