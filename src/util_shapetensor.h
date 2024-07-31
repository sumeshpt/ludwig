/*****************************************************************************
 *
 *  util_shapetensor.h
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2023 The University of Edinburgh
 *
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#ifndef LUDWIG_UTIL_SHAPETENSOR_H
#define LUDWIG_UTIL_SHAPETENSOR_H

int shapetensor_polygon(const int n, double *a);
int shapetensor_polyhedron(const int n, double a[3][3]);
double length_polygon(const double area, const int n);
void rotatevector(const double theta, double a[3], double *b);
void mulmxvector(const double m[3][3], const double a[3], double *b);
void subvecvector(const double a[3], const double b[3], double *c);
void addvecvector(const double a[3], const double b[3], double *c);
void outervecvec(const double a[3], const double b[3], double c[3][3]);
void addplustensor(const double a[3][3], double c[3][3]);
void copyvecvector(const double a[3], double *c);
void dividetensscalar(double a[3][3], double c); 
//void   util_vector_normalise(int n, double * a);
//void   util_vector_copy(int n, const double * a, double * b);

#endif
