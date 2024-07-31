/****************************************************************************
 *
 *  fe_st_stats.h
 *
 *  Edinburgh Soft Matter and Statistical Physics Group and
 *  Edinburgh Parallel Computing Centre
 *
 *  (c) 2017 The University of Edinburgh
 *
 *  Contributing authors:
 *  Kevin Stratford (kevin@epcc.ed.ac.uk)
 *
 ****************************************************************************/

#ifndef LUDWIG_FE_ST_STATS_H
#define LUDWIG_FE_ST_STATS_H

#include "pe.h"
#include "coords.h"
#include "shape_tensor.h"
#include "wall.h"
#include "map.h"
#include "colloids.h"

int fe_st_stats_info(pe_t * pe, cs_t * cs, fe_st_t * fe,
		     wall_t * wall, map_t * map,
		     colloids_info_t * cinfo, int step);
#endif
