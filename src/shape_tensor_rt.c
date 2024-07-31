/*****************************************************************************
 *
 *  shape_tensor_rt.c
 *
 *  Run time input for shape tensor free energy, and related parameters.
 *  Also relevant Beris Edwards parameters.
 *
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2009-2023 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 *****************************************************************************/

#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "coords.h"
#include "shape_tensor_init.h"
#include "shape_tensor_rt.h"
#include "physics.h"
#include "util_bits.h"
#include "util_shapetensor.h"

//int blue_phase_rt_coll_anchoring(pe_t * pe, rt_t * rt, rt_enum_t rt_err_level,
//				 lc_anchoring_param_t * coll);
//int blue_phase_rt_wall_anchoring(pe_t * pe, rt_t * rt, rt_enum_t rt_err_level,
//				 lc_anchoring_param_t * wall);

/*****************************************************************************
 *
 *  shape_tensor_init_rt
 *
 *  Pick up the shape tensor parameters from the input.
 *
 *****************************************************************************/

__host__ int shape_tensor_init_rt(pe_t * pe, rt_t *rt,
				fe_st_t * fe,
				beris_edwst_t * stbe) {
  int n;
  //int fe_is_lc_st;
  int redshift_update;
//  char method[BUFSIZ] = "s7"; /* This is the default */
//  char type[BUFSIZ] = "none";
//  char type_wall[BUFSIZ] = "none";

//  double epsilon;
  double redshift;
  double xi_default=1.0;/*Default value of xi for shape tensor*/
  int twod_default = 0;
  double stini[3][3] = {0};
//  double zeta0, zeta1, zeta2;
  double stGamma;

  fe_st_param_t fe_param = {0};
  beris_edwst_param_t stbe_param = {0};

  assert(pe);
  assert(rt);
  assert(fe);
  assert(stbe);

  pe_info(pe, "Shape tensor free energy selected.\n");

  {
    char description[BUFSIZ] = {0};
    rt_string_parameter(rt, "free_energy", description, BUFSIZ);
    n = (strcmp(description, "lc_and_shape") == 0);
    if (n != 1) pe_fatal(pe, "Wrong selection of free energy description <value>\n");
  }
  /* PARAMETERS */

  n = rt_double_parameter(rt, "st_k", &fe_param.k);
  if (n != 1) pe_fatal(pe, "Please specify st_k <value>\n");

  fe_param.twod=twod_default;
  n = rt_int_parameter(rt, "st_twod", &fe_param.twod);
  if(n == 1) pe_info(pe, "Shape tensor in %d dimensions selected\n",(twod_default-fe_param.twod)+3);
//  if (n != 1) pe_fatal(pe, "Please specify lc_xi <value>\n");

  n = rt_int_parameter(rt, "st_nside", &fe_param.nside);
  if (n != 1) pe_fatal(pe, "Please specify st_nside <value>\n");
  if(fe_param.twod>twod_default) {
    n = shapetensor_polygon(fe_param.nside, &fe_param.lambda0);
    }
  else {
    n = shapetensor_polyhedron(fe_param.nside, stini);
    fe_param.lambda[0]=stini[0][0];
    fe_param.lambda[1]=stini[1][1];
    fe_param.lambda[2]=stini[2][2];
    }
  if (n != 0) pe_fatal(pe, "Polygon strand tensor calculation failed.\n");

  fe_param.xi=xi_default;

  /* Use a default redshift of 1 */
  redshift = 1.0;
  rt_double_parameter(rt, "st_init_redshift", &redshift);
  fe_param.redshift = redshift;

  redshift_update = 0;
  rt_int_parameter(rt, "st_redshift_update", &redshift_update);
  fe_param.is_redshift_updated = redshift_update;

  pe_info(pe, "\n");
  pe_info(pe, "Shape tensor free energy\n");
  pe_info(pe, "Elastic constant k:         = %14.7e\n", fe_param.k);
  pe_info(pe, "No of sides: polygon/hedron = %d\n", fe_param.nside);
  pe_info(pe, "... gives the eigen values  = %14.7e, %14.7e, %14.7e\n", fe_param.lambda[0],fe_param.lambda[1],fe_param.lambda[2]);
  pe_info(pe, "Is shape tensor 2D          = %d\n", fe_param.twod);

  fe_st_param_set(fe, &fe_param);

  pe_info(pe, "Initial redshift           = %14.7e\n", fe_param.redshift);
  pe_info(pe, "Dynamic redshift update    = %14s\n",
	  redshift_update == 0 ? "no" : "yes");

  /* Beris Edwards */

  pe_info(pe, "\n");
  pe_info(pe, "Using Beris-Edwards solver for shape tensor:\n");

  n = rt_double_parameter(rt, "st_Gamma", &stGamma);

  if (n == 0) {
    pe_fatal(pe, "Please specify diffusion constant st_Gamma in the input\n");
  }
  else {
    stbe_param.Gamma = stGamma;
    stbe_param.xi = fe_param.xi;
    beris_edwst_param_set(stbe, &stbe_param);
    pe_info(pe, "Rotational diffusion const = %14.7e\n", stGamma);
  }

  return 0;
}

/*****************************************************************************
 *
 *  shape_tensor_rt_initial_conditions
 *
 *  There are several choices:
 *
 *****************************************************************************/

__host__ int shape_tensor_rt_initial_conditions(pe_t * pe, rt_t * rt, cs_t * cs,
					      fe_st_t * fe, field_t * r) {

  int n1;
  int n2;
  int nside;
//  int  rmin[3], rmax[3];
  char key1[FILENAME_MAX];
  char key2[FILENAME_MAX];

//  double nhat[3] = {1.0, 0.0, 0.0};
//  double nhat2[3] = {64.0, 3.0, 1.0};

  fe_st_param_t param;
  fe_st_param_t * feparam = &param;

  assert(pe);
  assert(cs);
  assert(fe);
  assert(r);

  fe_st_param(fe, feparam);

  pe_info(pe, "\n");

  n1 = rt_string_parameter(rt, "st_r_initialisation", key1, FILENAME_MAX);
  if (n1 != 1) pe_fatal(pe, "Please specify st_r_initialisation <value>\n");
  
  pe_info(pe, "\n");

  n2 = rt_string_parameter(rt, "st_shape_initialisation", key2, FILENAME_MAX);
  if (n2 != 1) pe_fatal(pe, "Please specify st_shape_initialisation <value>\n");

  pe_info(pe, "\n");

  if (strcmp(key2, "square") == 0) {
    nside = 4;
  }
  else if (strcmp(key2, "pentagon") == 0) {
    nside = 5;
  }
  else if (strcmp(key2, "hexagon") == 0) {
    nside = 6;
  }
  else if (strcmp(key2, "heptagon") == 0) {
    nside = 7;
  }
  else {
    pe_fatal(pe, "Please specify st_shape_initialisation <value>\n");
  }

  //if (strcmp(key1, "isotropic") == 0) {
    //pe_info(pe, "Initialising R_ab to isotropic\n");
    //shape_tensor_isotropic_init(cs, feparam, r);
  //}

  if (strcmp(key1, "isotropic") == 0) {
    pe_info(pe, "Initialising R_ab to be isotropic\n");
    shape_tensor_isotropic_init(cs, feparam, r, nside);
  }

//  if (strcmp(key1, "twist") == 0) {
//    /* This gives cholesteric_z (for backwards compatibility) */
//    pe_info(pe, "Initialising Q_ab to cholesteric\n");
//    pe_info(pe, "Helical axis Z\n");
//    blue_phase_twist_init(cs, feparam, q, Z);
//  }

//  if (strcmp(key1, "cholesteric_x") == 0) {
//    pe_info(pe, "Initialising Q_ab to cholesteric\n");
//    pe_info(pe, "Helical axis X\n");
//    blue_phase_twist_init(cs, feparam, q, X);
//  }

//  if (strcmp(key1, "cholesteric_y") == 0) {
//    pe_info(pe, "Initialising Q_ab to cholesteric\n");
//    pe_info(pe, "Helical axis Y\n");
//    blue_phase_twist_init(cs, feparam, q, Y);
//  }

//  if (strcmp(key1, "cholesteric_z") == 0) {
//    pe_info(pe, "Initialising Q_ab to cholesteric\n");
//    pe_info(pe, "Helical axis Z\n");
//    blue_phase_twist_init(cs, feparam, q, Z);
//  }

//  if (strcmp(key1, "nematic") == 0) {
//    pe_info(pe, "Initialising Q_ab to nematic\n");
//    rt_double_parameter_vector(rt, "lc_init_nematic", nhat);
//    pe_info(pe, "Director:  %14.7e %14.7e %14.7e\n", nhat[X], nhat[Y], nhat[Z]);
//    blue_phase_nematic_init(cs, feparam, q, nhat);
//  }

//  if (strcmp(key1, "active_nematic") == 0) {
//    pe_info(pe, "Initialising Q_ab to active nematic\n");
//    rt_double_parameter_vector(rt, "lc_init_nematic", nhat);
//    pe_info(pe, "Director:  %14.7e %14.7e %14.7e\n", nhat[X], nhat[Y], nhat[Z]);
//    blue_phase_active_nematic_init(cs, feparam, q, nhat);
//  }
//
//  if (strcmp(key1, "active_nematic_q2d_x") == 0) {
//    pe_info(pe, "Initialising Q_ab to quasi-2d with strip parallel to X\n");
//    lc_active_nematic_init_q2d(cs, feparam, q, X);
//  }
//
//  if (strcmp(key1, "active_nematic_q2d_y") == 0) {
//    pe_info(pe, "Initialising Q_ab to quasi-2d with strip parallel to Y\n");
//    lc_active_nematic_init_q2d(cs, feparam, q, Y);
//  }
//
//  if (strcmp(key1, "o8m") == 0) {
//
//    int   is_rot = 0;                   /* Default no rotation. */
//    double angles[3] = {0.0, 0.0, 0.0}; /* Default Euler rotation (degrees) */
//
//    pe_info(pe, "Initialising Q_ab using O8M (BPI)\n");
//    is_rot = rt_double_parameter_vector(rt, "lc_q_init_euler_angles", angles);
//
//    if (is_rot) {
//      pe_info(pe, "... initial conidition to be rotated ...\n");
//      pe_info(pe, "Euler angle (deg): alpha_z = %14.7e\n", angles[0]);
//      pe_info(pe, "Euler angle (deg): beta_x' = %14.7e\n", angles[1]);
//      pe_info(pe, "Euler angle (deg): gamma_z'= %14.7e\n", angles[2]);
//    }
//
//    blue_phase_O8M_init(cs, feparam, q, angles);
//  }
//
//  if (strcmp(key1, "o2") == 0) {
//
//    int   is_rot = 0;                   /* Default no rotation. */
//    double angles[3] = {0.0, 0.0, 0.0}; /* Default Euler rotation (degrees) */
//
//    pe_info(pe, "Initialising Q_ab using O2 (BPII)\n");
//    is_rot = rt_double_parameter_vector(rt, "lc_q_init_euler_angles", angles);
//
//    if (is_rot) {
//      pe_info(pe, "... initial conidition to be rotated ...\n");
//      pe_info(pe, "Euler angle (deg): alpha_z = %14.7e\n", angles[0]);
//      pe_info(pe, "Euler angle (deg): beta_x' = %14.7e\n", angles[1]);
//      pe_info(pe, "Euler angle (deg): gamma_z'= %14.7e\n", angles[2]);
//    }
//
//    blue_phase_O2_init(cs, feparam, q, angles);
//  }
//
//  if (strcmp(key1, "o5") == 0) {
//    pe_info(pe, "Initialising Q_ab using O5\n");
//    blue_phase_O5_init(cs, feparam, q);
//  }
//
//  if (strcmp(key1, "h2d") == 0) {
//    pe_info(pe, "Initialising Q_ab using H2D\n");
//    blue_phase_H2D_init(cs, feparam, q);
//  }
//
//  if (strcmp(key1, "h3da") == 0) {
//    pe_info(pe, "Initialising Q_ab using H3DA\n");
//    blue_phase_H3DA_init(cs, feparam, q);
//  }
//
//  if (strcmp(key1, "h3db") == 0) {
//    pe_info(pe, "Initialising Q_ab using H3DB\n");
//    blue_phase_H3DB_init(cs, feparam, q);
//  }
//
//  if (strcmp(key1, "dtc") == 0) {
//    pe_info(pe, "Initialising Q_ab using DTC\n");
//    blue_phase_DTC_init(cs, feparam, q);
//  }
//
//  if (strcmp(key1, "bp3") == 0) {
//    pe_info(pe, "Initialising Q_ab using BPIII\n");
//    rt_double_parameter_vector(rt, "lc_init_bp3", nhat2);
//    pe_info(pe, "BPIII specifications: N_DTC=%g,  R_DTC=%g,  ", nhat2[0], nhat2[1]);
//    if (nhat2[2] == 0) pe_info(pe, "isotropic environment\n");
//    if (nhat2[2] == 1) pe_info(pe, "cholesteric environment\n");
//    blue_phase_BPIII_init(cs, feparam, q, nhat2);
//  }
//
//  if (strcmp(key1, "cf1_x") == 0) {
//    pe_info(pe, "Initialising Q_ab to cholesteric finger (1st kind)\n");
//    pe_info(pe, "Finger axis X, helical axis Y\n");
//    blue_phase_cf1_init(cs, feparam, q, X);
//  }
//
//  if (strcmp(key1, "cf1_y") == 0) {
//    pe_info(pe, "Initialising Q_ab to cholesteric finger (1st kind)\n");
//    pe_info(pe, "Finger axis Y, helical axis Z\n");
//    blue_phase_cf1_init(cs, feparam, q, Y);
//  }
//
//  if (strcmp(key1, "cf1_z") == 0) {
//    pe_info(pe, "Initialising Q_ab to cholesteric finger (1st kind)\n");
//    pe_info(pe, "Finger axis Z, helical axis X\n");
//    blue_phase_cf1_init(cs, feparam, q, Z);
//  }
//
//  if (strcmp(key1, "cf1_fluc_x") == 0) {
//    pe_info(pe, "Initialising Q_ab to cholesteric finger (1st kind)\n");
//    pe_info(pe, "with added traceless symmetric random fluctuation.\n");
//    pe_info(pe, "Finger axis X, helical axis Y\n");
//    blue_phase_random_cf1_init(cs, feparam, q, X);
//  }
//
//  if (strcmp(key1, "cf1_fluc_y") == 0) {
//    pe_info(pe, "Initialising Q_ab to cholesteric finger (1st kind)\n");
//    pe_info(pe, "with added traceless symmetric random fluctuation.\n");
//    pe_info(pe, "Finger axis Y, helical axis Z\n");
//    blue_phase_random_cf1_init(cs, feparam, q, Y);
//  }
//
//  if (strcmp(key1, "cf1_fluc_z") == 0) {
//    pe_info(pe, "Initialising Q_ab to cholesteric finger (1st kind)\n");
//    pe_info(pe, "with added traceless symmetric random fluctuation.\n");
//    pe_info(pe, "Finger axis Z, helical axis X\n");
//    blue_phase_random_cf1_init(cs, feparam, q, Z);
//  }
//
//  if (strcmp(key1, "random") == 0) {
//    pe_info(pe, "Initialising Q_ab randomly\n");
//    blue_phase_random_q_init(cs, feparam, q);
//  }
//
//  if (strcmp(key1, "random_xy") == 0) {
//    pe_info(pe, "Initialising Q_ab at random in (x,y)\n");
//    blue_phase_random_q_2d(cs, feparam, q);
//  }
//
//  /* Superpose a rectangle of random Q_ab on whatever was above */
//
//  n1 = rt_int_parameter_vector(rt, "lc_q_init_rectangle_min", rmin);
//  n2 = rt_int_parameter_vector(rt, "lc_q_init_rectangle_max", rmax);
//
//  if (n1 == 1 && n2 == 1) {
//    pe_info(pe, "Superposing random rectangle\n");
//    blue_phase_random_q_rectangle(cs, feparam, q, rmin, rmax);
//  }

  return 0;
}

/*****************************************************************************
// *
// *  blue_phase_rt_coll_anchoring
// *
// *  Newer style anchoring input which is documented for colloids.
// *  Normal or planar only for colloids.
// *
// *****************************************************************************/
//
//int blue_phase_rt_coll_anchoring(pe_t * pe, rt_t * rt, rt_enum_t rt_err_level,
//				 lc_anchoring_param_t * coll) {
//
//  assert(pe);
//  assert(rt);
//  assert(coll);
//
//  /* No colloids at all returns 0. */
//
//  int ierr = 0;
//  char atype[BUFSIZ] = {0};
//
//  if (rt_string_parameter(rt, "lc_coll_anchoring", atype, BUFSIZ)) {
//
//    coll->type = lc_anchoring_type_from_string(atype);
//
//    switch (coll->type) {
//    case LC_ANCHORING_NORMAL:
//      ierr += rt_key_required(rt, "lc_coll_anchoring_w1", rt_err_level);
//      rt_double_parameter(rt, "lc_coll_anchoring_w1", &coll->w1);
//      break;
//    case LC_ANCHORING_PLANAR:
//      ierr += rt_key_required(rt, "lc_coll_anchoring_w1", rt_err_level);
//      ierr += rt_key_required(rt, "lc_coll_anchoring_w2", rt_err_level);
//      rt_double_parameter(rt, "lc_coll_anchoring_w1", &coll->w1);
//      rt_double_parameter(rt, "lc_coll_anchoring_w2", &coll->w2);
//      break;
//    default:
//      /* Not valid. */
//      rt_vinfo(rt, rt_err_level, "%s: %s\n",
//	       "Input key `lc_coll_anchoring` had invalid value", atype);
//      ierr += 1;
//    }
//  }
//
//  return ierr;
//}
//
///*****************************************************************************
// *
// *  blue_phase_rt_wall_anchoring
// *
// *  Newer style anchoring input which is documented (unlike the old type).
// *
// *****************************************************************************/
//
//int blue_phase_rt_wall_anchoring(pe_t * pe, rt_t * rt, rt_enum_t rt_err_level,
//				 lc_anchoring_param_t * wall) {
//
//  assert(pe);
//  assert(rt);
//  assert(wall);
//
//  /* No wall at all is fine; return 0. */
//
//  int ierr = 0;
//  char atype[BUFSIZ] = {0};
//
//  if (rt_string_parameter(rt, "lc_wall_anchoring", atype, BUFSIZ)) {
//    wall->type = lc_anchoring_type_from_string(atype);
//
//    switch (wall->type) {
//    case LC_ANCHORING_NORMAL:
//      ierr += rt_key_required(rt, "lc_wall_anchoring_w1", rt_err_level);
//      rt_double_parameter(rt, "lc_wall_anchoring_w1", &wall->w1);
//      break;
//    case LC_ANCHORING_PLANAR:
//      ierr += rt_key_required(rt, "lc_wall_anchoring_w1", rt_err_level);
//      ierr += rt_key_required(rt, "lc_wall_anchoring_w2", rt_err_level);
//      rt_double_parameter(rt, "lc_wall_anchoring_w1", &wall->w1);
//      rt_double_parameter(rt, "lc_wall_anchoring_w2", &wall->w2);
//      break;
//    case LC_ANCHORING_FIXED:
//      ierr += rt_key_required(rt, "lc_wall_anchoring_w1", rt_err_level);
//      ierr += rt_key_required(rt, "lc_wall_fixed_orientation", rt_err_level);
//      rt_double_parameter(rt, "lc_wall_anchoring_w1", &wall->w1);
//      rt_double_parameter_vector(rt, "lc_wall_fixed_orientation", wall->nfix);
//
//      /* Make sure this is a vlaid unit vector here */
//      {
//	double x2 = wall->nfix[X]*wall->nfix[X];
//	double y2 = wall->nfix[Y]*wall->nfix[Y];
//	double z2 = wall->nfix[Z]*wall->nfix[Z];
//	if (fabs(x2 + y2 + z2) < DBL_EPSILON) {
//	  ierr += 1;
//	  rt_vinfo(rt, rt_err_level, "%s'n",
//		   "lc_wall_fixed_orientation must be non-zero\n");
//	}
//	wall->nfix[X] /= sqrt(x2 + y2 + z2);
//	wall->nfix[Y] /= sqrt(x2 + y2 + z2);
//	wall->nfix[Z] /= sqrt(x2 + y2 + z2);
//      }
//      break;
//    default:
//      /* Not valid. */
//      rt_vinfo(rt, rt_err_level, "%s: %s\n",
//	       "Input key `lc_wall_anchoring` had invalid value", atype);
//      ierr += 1;
//    }
//  }
//
//  return ierr;
//}
