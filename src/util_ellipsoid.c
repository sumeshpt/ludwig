/*****************************************************************************
 *
 *  util_ellipsoid.c
 *
 *  Utility functions for ellipsoids.
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2010-2022 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>

#include <ctype.h>
#include <string.h>

#include "util.h"
#include "util_ellipsoid.h"

/*****************************************************************************
 *
 *  Orthonormalise a vector b to a given vector a
 *
 *****************************************************************************/

 __host__ __device__ void orthonormalise_vector_b_to_a(double *a, double *b){

  double proj,mag;
  /*projecting b onto a*/
  proj = a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
  b[0] = b[0] - proj*a[0];
  b[1] = b[1] - proj*a[1];
  b[2] = b[2] - proj*a[2];
  /*Normalising b */
  mag = sqrt(b[0]*b[0]+b[1]*b[1]+b[2]*b[2]);
  b[0]=b[0]/mag;
  b[1]=b[1]/mag;
  b[2]=b[2]/mag;
  return ;
}


/*****************************************************************************
 *
 *  Normalise a vector a unit vector
 *
 *****************************************************************************/

__host__ __device__ void normalise_unit_vector(double *a ,const int n){
  
  double magsum = 0.0;
  for(int i = 0; i < n; i++) {magsum+=a[i]*a[i];}
  double mag = sqrt(magsum);
  for(int i = 0; i < n; i++) {a[i]=a[i]/mag;}
  return ; 
}

/*****************************************************************************
 *
 *  matrix_product
 *
 *****************************************************************************/

__host__ __device__
void matrix_product(const double a[3][3], const double b[3][3], double result[3][3]) {

  for(int i = 0; i < 3; i++){
    for(int j = 0; j < 3; j++){
      result[i][j]=0.0;
      for(int k=0; k < 3; k++){
        result[i][j]+=a[i][k]*b[k][j];
      }   
    }   
  }
  return;
}

/*****************************************************************************
 *
 *  matrix_transpose
 *
 *****************************************************************************/

__host__ __device__
void matrix_transpose(const double a[3][3], double result[3][3]) {

  for(int i = 0; i < 3; i++){
    for(int j = 0; j < 3; j++){
      result[i][j]=a[j][i];
    }   
  }
  return;
}

/*****************************************************************************
 *
 *  Printing vector on the screen
 *
 ****************************************************************************/
__host__ __device__ void print_vector_onscreen(const double *a, const int n){
  for(int i = 0; i < n; i++) printf("%22.15e, ",a[i]);
  printf("\n");
  return ;
  }
/*****************************************************************************
 *
 *  Printing matrix on the screen
 *
 ****************************************************************************/
__host__ __device__ void print_matrix_onscreen(const double a[3][3]){
  for(int i = 0; i < 3; i++) {
    for(int j = 0; j < 3; j++) {
      printf("%22.15e, ",a[i][j]);
    }
    printf("\n");
  }
  return ;
  }
/*****************************************************************************
 *
 *  Multiplying quaternions
 *
 ****************************************************************************/
__host__ __device__ void quaternion_product(const double a[4], const double b[4], double result[4]) {
  result[1]= a[0]*b[1] - a[3]*b[2] + a[2]*b[3] + a[1]*b[0];
  result[2]= a[3]*b[1] + a[0]*b[2] - a[1]*b[3] + a[2]*b[0];
  result[3]=-a[2]*b[1] + a[1]*b[2] + a[0]*b[3] + a[3]*b[0];
  result[0]=-a[1]*b[1] - a[2]*b[2] - a[3]*b[3] + a[0]*b[0];
  return;
  }
/*****************************************************************************
 *
 *  Determining a quaternion from the angular velocity
 *
 ****************************************************************************/
__host__ __device__ void quaternion_from_omega(const double omega[3], const double f, double qbar[4]) {

 double omag;
 omag=sqrt(omega[0]*omega[0]+omega[1]*omega[1]+omega[2]*omega[2]);
 if(omag>1e-12) {
   qbar[0]=cos(omag*f);
   for(int i = 0; i < 3; i++) {qbar[i+1]=sin(omag*f)*(omega[i]/omag);}
 }
 else {
  qbar[0]=1.0;
  for (int i = 1; i < 4; i++) {qbar[i] = 0.0;}
 }
  
  return;
  }

/*****************************************************************************
 *
 *  Calculate the moment of inerita tensor from the specified quaternion
 *
 ****************************************************************************/
__host__ __device__ void inertia_tensor_quaternion(const double q[4], const double moment[3], double mI[3][3]) {

  /*Construting the moment of inertia tensor in the principal coordinates*/
  double mIP[3][3];
  for(int i = 0; i < 3; i++) {
    for(int j = 0; j < 3; j++) {
      mIP[i][j] = 0.0;
    }
  }
  for(int i = 0; i < 3; i++) {mIP[i][i] = moment[i];}
  /*Rotating it to the body frame, eqn 13*/
  double Mi[3],Mdi[3],Mdd[3][3];
  /*First column transform and fill as first row*/
  for(int i = 0; i < 3; i++) {Mi[i] = mIP[i][0];}
  rotate_tobodyframe_quaternion(q,Mi,Mdi);
  for(int i = 0; i < 3; i++) {Mdd[0][i] = Mdi[i];}
  /*Second column transform and fill as second row*/
  for(int i = 0; i < 3; i++) {Mi[i] = mIP[i][1];}
  rotate_tobodyframe_quaternion(q,Mi,Mdi);
  for(int i = 0; i < 3; i++) {Mdd[1][i] = Mdi[i];}
  /*Third column transform and fill as third row*/
  for(int i = 0; i < 3; i++) {Mi[i] = mIP[i][2];}
  rotate_tobodyframe_quaternion(q,Mi,Mdi);
  for(int i = 0; i < 3; i++) {Mdd[2][i] = Mdi[i];}
  /*Repeat the entire procedure*/
  /*First column transform and fill as first row*/
  for(int i = 0; i < 3; i++) {Mi[i] = Mdd[i][0];}
  rotate_tobodyframe_quaternion(q,Mi,Mdi);
  for(int i = 0; i < 3; i++) {mI[0][i] = Mdi[i];}
  /*Second column transform and fill as second row*/
  for(int i = 0; i < 3; i++) {Mi[i] = Mdd[i][1];}
  rotate_tobodyframe_quaternion(q,Mi,Mdi);
  for(int i = 0; i < 3; i++) {mI[1][i] = Mdi[i];}
  /*Third column transform and fill as third row*/
  for(int i = 0; i < 3; i++) {Mi[i] = Mdd[i][2];}
  rotate_tobodyframe_quaternion(q,Mi,Mdi);
  for(int i = 0; i < 3; i++) {mI[2][i] = Mdi[i];}
  return;
  }
/*****************************************************************************
 *
 *  Rotate to body frame by quaternion
 *
 ****************************************************************************/
__host__ __device__ void rotate_tobodyframe_quaternion(const double q[4], const double a[3], double b[3]) {
   double s1, s2, s3[3];
   s1=(2.0*q[0]*q[0]-1.0);
   s2=dot_product(&q[1],a);
   cross_product(&q[1],a,s3);
   for(int i = 0; i < 3; i++) {b[i]=s1*a[i] + 2.0*s2*q[i+1] + 2.0*q[0]*s3[i];}
 return;
 }
/*****************************************************************************
 *
 *  Rotate to world frame by quaternion
 *
 ****************************************************************************/
/*__host__ __device__ void rotate_toworldframe_quaternion(const double q[4], const double a[3], double b[3]) {
   double qstar[4];
   qstar[0]=q[0];
   for(int i = 1; i < 4; i++) {qstar[i] = -q[i];}
   rotate_tobodyframe_quaternion(qstar,a,b);
 return;
 }

*/
/*****************************************************************************
 *
 *  Determining Euler angles from quaternions
 *
 ****************************************************************************/
__host__ __device__ void eulerangles_from_quaternions(const double *q, double *phi, double *theta, double *psi) {

  double st,ct,sp,cp,ss,cs;
  st=2.0*sqrt((q[1]*q[1]+q[2]*q[2])*(1.0-q[1]*q[1]-q[2]*q[2]));
  ct=1-2.0*(q[1]*q[1]+q[2]*q[2]);
  if(fabs(st)>1.0e-12) {
    sp=2.0*(q[1]*q[3]+q[2]*q[0])/st;
    cp=2.0*(q[1]*q[0]-q[2]*q[3])/st;
    ss=2.0*(q[1]*q[3]-q[2]*q[0])/st;
    cs=2.0*(q[1]*q[0]+q[2]*q[3])/st;
    *phi = atan2(sp,cp);
    *theta = atan2(st,ct);
    *psi = atan2(ss,cs);
  }
  else {
    st=sqrt(q[1]*q[1]+q[2]*q[2]);
    ct=sqrt(q[3]*q[3]+q[0]*q[0]);
    sp=sqrt(q[2]*q[2]+q[3]*q[3]);
    cp=sqrt(q[1]*q[1]+q[0]*q[0]);
    *phi=2*atan2(sp,cp);
    *theta=2*atan2(st,ct);
    *psi = 0.0;
  }
  return;
  }


/*****************************************************************************
 *
 *  Determining quaternions from Euler angles
 *
 ****************************************************************************/
__host__ __device__ void quaternions_from_eulerangles(const double phi, const double theta, const double psi, double q[4]){

  q[0]=cos(theta/2.0)*cos((phi+psi)/2.0);
  q[1]=sin(theta/2.0)*cos((phi-psi)/2.0);
  q[2]=sin(theta/2.0)*sin((phi-psi)/2.0);
  q[3]=cos(theta/2.0)*sin((phi+psi)/2.0);
  return;
  }

/*****************************************************************************
 *
 *  Copy a vector to another
 *
 ****************************************************************************/
__host__ __device__ void copy_vectortovector(const double a[3], double b[3], int n){
  for(int i=0; i < n; i++){
    b[i]=a[i];
  }
  return ;
  }

/*****************************************************************************
*
*  Jeffery's predictions for a spheroid
*
*****************************************************************************/
__host__ __device__ void Jeffery_omega_predicted(double const r, double const quater[4], double const gammadot, double opred[3], double angpred[2]) {

  double beta;
  double phi1,the1;
  double v1[3]={1.0,0.0,0.0};
  double v2[3]={0.0,1.0,0.0};
  double v3[3]={0.0,0.0,1.0};
  double p[3],pdot[3];
  double pdoty,phiar;
  double pcpdot[3];
  double pxj,pyj,pzj,pxdotj,pydotj,pzdotj;
  double op[3]={0.0,0.0,0.0};
  double omp;

  beta=(r*r-1.0)/(r*r+1.0);
  /*Determining p, the orientation of the long axis*/
  rotate_tobodyframe_quaternion(quater,v1,p);
  /*Determing pdot in Guazzeli's convention*/
  pdoty=p[0]*v2[0]+p[1]*v2[1]+p[2]*v2[2];
  phiar=(p[0]-pdoty*v2[0])*v3[0]+
        (p[1]-pdoty*v2[1])*v3[1]+
        (p[2]-pdoty*v2[2])*v3[2];
  the1=acos(-pdoty);
  phi1=acos(phiar);
  angpred[0]=phi1;
  angpred[1]=the1;
  pxj= sin(the1)*sin(phi1);
  pyj= sin(the1)*cos(phi1);
  pzj=-cos(the1);
  pxdotj= gammadot*((beta+1.0)*pyj/2.0-beta*pxj*pxj*pyj);
  pydotj= gammadot*((beta-1.0)*pxj/2.0-beta*pyj*pyj*pxj);
  pzdotj=-gammadot*beta*pxj*pyj*pzj;
  /*Determing pdot in Ludwig's convention*/
  pdot[0]=pxdotj;
  pdot[1]=-pzdotj;
  pdot[2]=pydotj;
  /*Determing the spinning velocity*/
  op[1]= gammadot/2.0;
  omp=dot_product(op,p);
  /*Determining the tumbling velocity*/
  cross_product(p,pdot,pcpdot);
  /*Determining the total angular velocity*/
  for(int i = 0; i < 3; i++) {opred[i]=omp*p[i]+pcpdot[i];}

  return ;
  }

/*****************************************************************************
*
*  Calculating Euler angles from vectors
*
*****************************************************************************/
__host__ __device__ void euler_from_vectors(double a[3], double b[3], double *euler) {

  double c[3],r[3][3];
  normalise_unit_vector(a, 3);
  orthonormalise_vector_b_to_a(a, b);
  cross_product(a,b,c);
  dcm_from_vectors(a,b,c,r);
  euler_from_dcm(r,&euler[0],&euler[1],&euler[2]);

 return;
}
/*****************************************************************************
*
*  Calculating Euler angles from Direction Cosine Matrix
*
*****************************************************************************/
__host__ __device__ void euler_from_dcm(double const r[3][3], double *phi, double *theta, double *psi) {

  *theta=acos(r[2][2]);
  if(fabs(fabs(r[2][2])-1.0)>1e-12) {
    *phi=atan2(r[2][0],-r[2][1]);
    *psi=atan2(r[0][2],r[1][2]);
  }
  else {
    *phi=atan2(r[0][1],r[0][0]);
    *psi = 0.0;
  }
return;
}
/*****************************************************************************
*
*  Calculating Direction Cosine Matrix from a given set of orientation vectors
*
*****************************************************************************/
__host__ __device__ void dcm_from_vectors(double const a[3], double const b[3], double const c[3], double r[3][3]) {

double v1[3]={1.0,0.0,0.0};
double v2[3]={0.0,1.0,0.0};
double v3[3]={0.0,0.0,1.0};

r[0][0] = dot_product(v1,a);
r[1][0] = dot_product(v1,b);
r[2][0] = dot_product(v1,c);
r[0][1] = dot_product(v2,a);
r[1][1] = dot_product(v2,b);
r[2][1] = dot_product(v2,c);
r[0][2] = dot_product(v3,a);
r[1][2] = dot_product(v3,b);
r[2][2] = dot_product(v3,c);

return;
}

/*****************************************************************************
*
*  Settling velocity of a prolate spheroid, Leal pg 559
*
*****************************************************************************/
__host__ __device__ void settling_velocity_prolate(double const r, double const f, double const mu, double const ela, double U[2]) {

double ecc,logecc;
double cfterm1,cfterm21,cf1br,cf1;
double cfterm2,cfterm22,cf2br,cf2;
double dcoef;
double rinv;
PI_DOUBLE(pi);
rinv = 1.0/r;
ecc = sqrt(1.0 - (rinv*rinv));
logecc = log((1.0+ecc)/(1.0-ecc));
cfterm1 = -2.0*ecc;
cfterm21 = 1.0 + ecc*ecc;
cf1br = cfterm1 + cfterm21*logecc;
cf1 = (8.0/3.0)*(ecc*ecc*ecc)/cf1br;
cfterm2 = 2.0*ecc;
cfterm22 = 3.0*ecc*ecc - 1.0;
cf2br = cfterm2 + cfterm22*logecc;
cf2 = (16.0/3.0)*(ecc*ecc*ecc)/cf2br;
dcoef = 6.0*pi*mu*ela;
U[0] = f/(dcoef*cf1);
U[1] = f/(dcoef*cf2);
return;
}

/*****************************************************************************
 *
 *  Calculate the mass of an ellipsoid
 *
 ****************************************************************************/
__host__ __device__ double mass_ellipsoid(const double dim[3], const double density){

  PI_DOUBLE(pi);
  return (4.0/3.0)*pi*(dim[0]*dim[1]*dim[2]);

}

/*****************************************************************************
 *
 *  Calculate the unsteady term dIij/dt
 *
 ****************************************************************************/
__host__ __device__ void unsteady_mI(const double q[4], const double I[3], const double omega[3], double F[3][3]){

double Ixx, Iyy, Izz;
double ox, oy, oz;

Ixx=I[0];
Iyy=I[1];
Izz=I[2];
ox=omega[0];
oy=omega[1];
oz=omega[2];

F[0][0] = 4.0*Izz*(q[0]*q[2] + q[1]*q[3])*(oy*q[0]*q[0] + 2.0*oz*q[0]*q[1] - oy*q[1]*q[1] - oy*q[2]*q[2] - 2.0*oz*q[2]*q[3] + oy*q[3]*q[3]) + 4.0*Iyy*(q[1]*q[2] - q[0]*q[3])*(-(oz*q[0]*q[0]) + 2.0*oy*q[0]*q[1] + oz*q[1]*q[1] - oz*q[2]*q[2] + 2.0*oy*q[2]*q[3] + oz*q[3]*q[3]) - 4.0*Ixx*(-1.0 + 2.0*q[0]*q[0] + 2.0*q[1]*q[1])* (q[1]*(oz*q[2] - oy*q[3]) + q[0]*(oy*q[2] + oz*q[3]));

F[0][1] = -(Iyy*(-1.0 + 2.0*q[0]*q[0] + 2.0*q[2]*q[2])*(oz*q[0]*q[0] - 2.0*oy*q[0]*q[1] - oz*q[1]*q[1] + oz*q[2]*q[2] - 2.0*oy*q[2]*q[3] - oz*q[3]*q[3])) + 4.0*Iyy*(q[1]*q[2] - q[0]*q[3])* (q[2]*(oz*q[1] - ox*q[3]) - q[0]*(ox*q[1] + oz*q[3])) - 4.0*Ixx*(q[1]*q[2] + q[0]*q[3])*(q[1]*(oz*q[2] - oy*q[3]) + q[0]*(oy*q[2] + oz*q[3])) + Ixx*(-1.0 + 2.0*q[0]*q[0] + 2.0*q[1]*q[1])* (oz*q[0]*q[0] + oz*q[1]*q[1] + 2.0*ox*q[0]*q[2] - 2.0*ox*q[1]*q[3] - oz*(q[2]*q[2] + q[3]*q[3])) - 2.0*Izz*(q[0]*q[0]*q[0]*(oy*q[1] + ox*q[2]) + q[0]*q[0]*(2.0*oz*q[1]*q[1] + ox*q[1]*q[3] - q[2]*(2.0*oz*q[2] + oy*q[3])) + q[3]*(-(ox*q[1]*q[1]*q[1]) + q[1]*q[1]*(oy*q[2] - 2.0*oz*q[3]) + ox*q[1]*(-q[2]*q[2] + q[3]*q[3]) + q[2]*(oy*q[2]*q[2] + 2.0*oz*q[2]*q[3] - oy*q[3]*q[3])) - q[0]*(oy*q[1]*q[1]*q[1] + ox*q[1]*q[1]*q[2] + ox*q[2]*(q[2]*q[2] - q[3]*q[3]) + q[1]*(oy*q[2]*q[2] + 8*oz*q[2]*q[3] - oy*q[3]*q[3])));

F[0][2] = -4.0*Izz*(q[0]*q[2] + q[1]*q[3])*(q[0]*(ox*q[1] + oy*q[2]) + (oy*q[1] - ox*q[2])*q[3]) +  Izz*(-1.0 + 2.0*q[0]*q[0] + 2.0*q[3]*q[3])*(oy*q[0]*q[0] + 2.0*oz*q[0]*q[1] - oy*q[1]*q[1] - oy*q[2]*q[2] - 2.0*oz*q[2]*q[3] + oy*q[3]*q[3]) + 4.0*Ixx*(q[0]*q[2] - q[1]*q[3])*(q[1]*(oz*q[2] - oy*q[3]) + q[0]*(oy*q[2] + oz*q[3])) - Ixx*(-1.0 + 2.0*q[0]*q[0] + 2.0*q[1]*q[1])*(oy*q[0]*q[0] + oy*q[1]*q[1] - 2.0*ox*q[1]*q[2] - 2.0*ox*q[0]*q[3] - oy*(q[2]*q[2] + q[3]*q[3])) - 2.0*Iyy*(q[0]*q[0]*q[0]*(oz*q[1] + ox*q[3]) + q[0]*q[0]*(-2.0*oy*q[1]*q[1] - ox*q[1]*q[2] + q[3]*(oz*q[2] + 2.0*oy*q[3])) + q[2]*(ox*q[1]*q[1]*q[1] + q[1]*q[1]*(2.0*oy*q[2] - oz*q[3]) + ox*q[1]*(-q[2]*q[2] + q[3]*q[3]) + q[3]*(oz*q[2]*q[2] - 2.0*oy*q[2]*q[3] - oz*q[3]*q[3])) - q[0]*(oz*q[1]*q[1]*q[1] + ox*q[1]*q[1]*q[3] + ox*q[3]*(-q[2]*q[2] + q[3]*q[3]) + q[1]*(-(oz*q[2]*q[2]) + 8*oy*q[2]*q[3] + oz*q[3]*q[3])));

F[1][0] = -(Iyy*(-1.0 + 2.0*q[0]*q[0] + 2.0*q[2]*q[2])*(oz*q[0]*q[0] - 2.0*oy*q[0]*q[1] - oz*q[1]*q[1] + oz*q[2]*q[2] - 2.0*oy*q[2]*q[3] - oz*q[3]*q[3])) + 4.0*Iyy*(q[1]*q[2] - q[0]*q[3])*(q[2]*(oz*q[1] - ox*q[3]) - q[0]*(ox*q[1] + oz*q[3])) - 4.0*Ixx*(q[1]*q[2] + q[0]*q[3])*  (q[1]*(oz*q[2] - oy*q[3]) + q[0]*(oy*q[2] + oz*q[3])) + Ixx*(-1.0 + 2.0*q[0]*q[0] + 2.0*q[1]*q[1])*(oz*q[0]*q[0] + oz*q[1]*q[1] + 2.0*ox*q[0]*q[2] - 2.0*ox*q[1]*q[3] - oz*(q[2]*q[2] + q[3]*q[3])) - 2.0*Izz*(q[0]*q[0]*q[0]*(oy*q[1] + ox*q[2]) + q[0]*q[0]*(2.0*oz*q[1]*q[1] + ox*q[1]*q[3] - q[2]*(2.0*oz*q[2] + oy*q[3])) + q[3]*(-(ox*q[1]*q[1]*q[1]) + q[1]*q[1]*(oy*q[2] - 2.0*oz*q[3]) + ox*q[1]*(-q[2]*q[2] + q[3]*q[3]) + q[2]*(oy*q[2]*q[2] + 2.0*oz*q[2]*q[3] - oy*q[3]*q[3])) -q[0]*(oy*q[1]*q[1]*q[1] + ox*q[1]*q[1]*q[2] + ox*q[2]*(q[2]*q[2] - q[3]*q[3]) + q[1]*(oy*q[2]*q[2] + 8*oz*q[2]*q[3] - oy*q[3]*q[3])));

F[1][1] = -4.0*Iyy*(-1.0 + 2.0*q[0]*q[0] + 2.0*q[2]*q[2])*(q[2]*(-(oz*q[1]) + ox*q[3]) + q[0]*(ox*q[1] + oz*q[3])) + 4.0*Izz*(q[0]*q[1] - q[2]*q[3])*(ox*q[0]*q[0] - ox*q[1]*q[1] - 2.0*oz*q[0]*q[2] - 2.0*oz*q[1]*q[3] + ox*(-q[2]*q[2] + q[3]*q[3])) + 4.0*Ixx*(q[1]*q[2] + q[0]*q[3])*(oz*q[0]*q[0] + oz*q[1]*q[1] + 2.0*ox*q[0]*q[2] - 2.0*ox*q[1]*q[3] - oz*(q[2]*q[2] + q[3]*q[3]));

F[1][2] = 4.0*Izz*(q[0]*q[1] - q[2]*q[3])*(q[0]*(ox*q[1] + oy*q[2]) + (oy*q[1] - ox*q[2])*q[3]) - 4.0*Iyy*(q[0]*q[1] + q[2]*q[3])*(q[2]*(-(oz*q[1]) + ox*q[3]) + q[0]*(ox*q[1] + oz*q[3])) + Iyy*(-1.0 + 2.0*q[0]*q[0] + 2.0*q[2]*q[2])*(ox*q[0]*q[0] - ox*q[1]*q[1] - 2.0*oy*q[1]*q[2] + 2.0*oy*q[0]*q[3] + ox*(q[2]*q[2] - q[3]*q[3])) - Izz*(-1.0 + 2.0*q[0]*q[0] + 2.0*q[3]*q[3])*(ox*q[0]*q[0] - ox*q[1]*q[1] - 2.0*oz*q[0]*q[2] - 2.0*oz*q[1]*q[3] + ox*(-q[2]*q[2] + q[3]*q[3])) - 2.0*Ixx*(q[0]*q[0]*q[0]*(oz*q[2] + oy*q[3]) + q[0]*q[0]*(q[1]*(oy*q[2] - oz*q[3]) + 2.0*ox*(q[2]*q[2] - q[3]*q[3])) + q[0]*(-8*ox*q[1]*q[2]*q[3] + q[1]*q[1]*(oz*q[2] + oy*q[3]) - (oz*q[2] + oy*q[3])*(q[2]*q[2] + q[3]*q[3])) + q[1]*(q[1]*q[1]*(oy*q[2] - oz*q[3]) + 2.0*ox*q[1]*(-q[2]*q[2] + q[3]*q[3]) - (oy*q[2] - oz*q[3])*(q[2]*q[2] + q[3]*q[3])));

F[2][0] = -4.0*Izz*(q[0]*q[2] + q[1]*q[3])*(q[0]*(ox*q[1] + oy*q[2]) + (oy*q[1] - ox*q[2])*q[3]) + Izz*(-1.0 + 2.0*q[0]*q[0] + 2.0*q[3]*q[3])*(oy*q[0]*q[0] + 2.0*oz*q[0]*q[1] - oy*q[1]*q[1] - oy*q[2]*q[2] - 2.0*oz*q[2]*q[3] + oy*q[3]*q[3]) + 4.0*Ixx*(q[0]*q[2] - q[1]*q[3])*(q[1]*(oz*q[2] - oy*q[3]) + q[0]*(oy*q[2] + oz*q[3])) - Ixx*(-1.0 + 2.0*q[0]*q[0] + 2.0*q[1]*q[1])*(oy*q[0]*q[0] + oy*q[1]*q[1] - 2.0*ox*q[1]*q[2] - 2.0*ox*q[0]*q[3] - oy*(q[2]*q[2] + q[3]*q[3])) - 2.0*Iyy*(q[0]*q[0]*q[0]*(oz*q[1] + ox*q[3]) + q[0]*q[0]*(-2.0*oy*q[1]*q[1] - ox*q[1]*q[2] + q[3]*(oz*q[2] + 2.0*oy*q[3])) + q[2]*(ox*q[1]*q[1]*q[1] + q[1]*q[1]*(2.0*oy*q[2] - oz*q[3]) + ox*q[1]*(-q[2]*q[2] + q[3]*q[3]) + q[3]*(oz*q[2]*q[2] - 2.0*oy*q[2]*q[3] - oz*q[3]*q[3])) - q[0]*(oz*q[1]*q[1]*q[1] + ox*q[1]*q[1]*q[3] + ox*q[3]*(-q[2]*q[2] + q[3]*q[3]) + q[1]*(-(oz*q[2]*q[2]) + 8*oy*q[2]*q[3] + oz*q[3]*q[3])));

F[2][1] = 4.0*Izz*(q[0]*q[1] - q[2]*q[3])*(q[0]*(ox*q[1] + oy*q[2]) + (oy*q[1] - ox*q[2])*q[3]) - 4.0*Iyy*(q[0]*q[1] + q[2]*q[3])*(q[2]*(-(oz*q[1]) + ox*q[3]) + q[0]*(ox*q[1] + oz*q[3])) + Iyy*(-1.0 + 2.0*q[0]*q[0] + 2.0*q[2]*q[2])*(ox*q[0]*q[0] - ox*q[1]*q[1] - 2.0*oy*q[1]*q[2] + 2.0*oy*q[0]*q[3] + ox*(q[2]*q[2] - q[3]*q[3])) - Izz*(-1.0 + 2.0*q[0]*q[0] + 2.0*q[3]*q[3])*(ox*q[0]*q[0] - ox*q[1]*q[1] - 2.0*oz*q[0]*q[2] - 2.0*oz*q[1]*q[3] + ox*(-q[2]*q[2] + q[3]*q[3])) - 2.0*Ixx*(q[0]*q[0]*q[0]*(oz*q[2] + oy*q[3]) + q[0]*q[0]*(q[1]*(oy*q[2] - oz*q[3]) + 2.0*ox*(q[2]*q[2] - q[3]*q[3])) + q[0]*(-8*ox*q[1]*q[2]*q[3] + q[1]*q[1]*(oz*q[2] + oy*q[3]) - (oz*q[2] + oy*q[3])*(q[2]*q[2] + q[3]*q[3])) + q[1]*(q[1]*q[1]*(oy*q[2] - oz*q[3]) + 2.0*ox*q[1]*(-q[2]*q[2] + q[3]*q[3]) - (oy*q[2] - oz*q[3])*(q[2]*q[2] + q[3]*q[3])));

F[2][2] = -4.0*Izz*(q[0]*(ox*q[1] + oy*q[2]) + (oy*q[1] - ox*q[2])*q[3])*(-1.0 + 2.0*q[0]*q[0] + 2.0*q[3]*q[3]) + 4.0*Iyy*(q[0]*q[1] + q[2]*q[3])*(ox*q[0]*q[0] - ox*q[1]*q[1] - 2.0*oy*q[1]*q[2] + 2.0*oy*q[0]*q[3] + ox*(q[2]*q[2] - q[3]*q[3])) + 4.0*Ixx*(q[0]*q[2] - q[1]*q[3])*(oy*q[0]*q[0] + oy*q[1]*q[1] - 2.0*ox*q[1]*q[2] - 2.0*ox*q[0]*q[3] - oy*(q[2]*q[2] + q[3]*q[3])); 

return;
}

/*****************************************************************************/
