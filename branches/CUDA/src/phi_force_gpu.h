/*****************************************************************************
 *
 *  gradient_3d_7pt_fluid_gpu.h
 *
 *  Alan Gray
 *
 *****************************************************************************/

#ifndef _PHI_FORCE_GPU_H
#define _PHI_FORCE_GPU_H

#include "common_gpu.h"

/* expose main routine in this module to outside routines */
extern "C" void phi_force_calculation_gpu(void);




/* declarations for required external (host) routines */
extern "C" double blue_phase_redshift(void);
extern "C" double blue_phase_rredshift(void);
extern "C" double blue_phase_q0(void);
extern "C" double blue_phase_a0(void);
extern "C" double blue_phase_kappa0(void);
extern "C" double blue_phase_kappa1(void);
extern "C" double blue_phase_get_xi(void);
extern "C" double blue_phase_get_zeta(void);
extern "C" double blue_phase_gamma(void);
extern "C" void blue_phase_get_electric_field(double electric_[3]);
extern "C" double blue_phase_get_dielectric_anisotropy(void);
extern "C" void phi_gradients_tensor_gradient(const int index, double dq[3][3][3]);
extern "C" void phi_gradients_tensor_delsq(const int index, double dsq[3][3]);
extern "C" void phi_get_q_tensor(int index, double q[3][3]);
extern "C" void hydrodynamics_add_force_local(const int index, const double force[3]);

/* external variables holding device memory addresses */
extern double * phi_site_d;
extern double * grad_phi_site_d;
extern double * delsq_phi_site_d;
extern double * force_d;

extern double * r3_d;
extern double * d_d;
extern double * e_d;

extern double * electric_d;

extern int * N_d;
extern int * le_index_real_to_buffer_d;

/* forward declarations of device routines */
__global__ void phi_force_calculation_fluid_gpu_d(int nop, 
						  int nhalo, 
						  int N_d[3], 
						  int * le_index_real_to_buffer_d,
						  int nextra,
						  double *phi_site_d,
						  double *grad_phi_site_d,
						  double *delsq_phi_site_d,
						  double *force_d,
						  double redshift_,
						  double rredshift_,
						  double q0_,
						  double a0_,
						  double kappa0_,
						  double kappa1_,
						  double xi_,
						  double zeta_,
						  double gamma_,
						  double electric_[3],
						  double epsilon_,
						  double *r3_d,
						  double *d_d,
						  double *e_d);

__device__ static int get_linear_index_gpu_d(int ii,int jj,int kk,int N[3]);
__device__ static void get_coords_from_index_gpu_d(int *ii,int *jj,int *kk,
						   int index,int N[3]);





/* /\* forward declarations of device routines *\/ */
/* __global__ void phi_force_calculation_fluid_gpu_d(int nop, int nhalo,  */
/* 						     int N[3],  */
/* 						     const double * field_d, */
/* 						     double * grad_d, */
/* 						     double * del2_d, */
/* 						     int * le_index_real_to_buffer_d); */
/* __device__ static int get_linear_index_gpu_d(int ii,int jj,int kk,int N[3]); */
/* __device__ static void get_coords_from_index_gpu_d(int *ii,int *jj,int *kk, */
/* 						   int index,int N[3]); */

/* /\* external variables holding device memory addresses *\/ */
/* extern double * phi_site_d; */
/* extern double * grad_phi_site_d; */
/* extern double * delsq_phi_site_d; */
/* extern int * N_d; */
/* extern int * le_index_real_to_buffer_d; */

#endif