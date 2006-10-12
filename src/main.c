/*****************************************************************************
 *
 *  Ludwig
 *
 *  A lattice Boltzmann code for complex fluids.
 *
 *****************************************************************************/

#include <stdio.h>

#include "pe.h"
#include "runtime.h"
#include "ran.h"
#include "timer.h"
#include "coords.h"
#include "control.h"
#include "free_energy.h"
#include "model.h"
#include "bbl.h"

#include "colloids.h"
#include "collision.h"
#include "test.h"
#include "wall.h"
#include "communicate.h"
#include "leesedwards.h"
#include "interaction.h"
#include "propagation.h"

#include "lattice.h"
#include "cio.h"
#include "regsteer.h"

static char rcsid[] = "$Id: main.c,v 1.7 2006-10-12 14:09:18 kevin Exp $";


int main( int argc, char **argv )
{
  char    filename[256];
  int     step;

  /* Initialise the following:
   *    - RealityGrid steering (if required)
   *    - communications (MPI)
   *    - random number generation (serial RNG and parallel fluctuations)
   *    - model fields
   *    - simple walls 
   *    - colloidal particles */

  REGS_init();

  pe_init(argc, argv);
  if (argc > 1) {
    RUN_read_input_file(argv[1]);
  }
  else {
    RUN_read_input_file("input");
  }
  coords_init();
  init_control();

  COM_init( argc, argv );

  TIMER_init();
  TIMER_start(TIMER_TOTAL);

  ran_init();
  RAND_init_fluctuations();
  MODEL_init();
  wall_init();
  COLL_init();

  init_free_energy();

  /* Report initial statistics */

  TEST_statistics();
  TEST_momentum();

  /* Main time stepping loop */

  while (next_step()) {

    step = get_step();
    TIMER_start(TIMER_STEPS);

#ifdef _REGS_
    {
      int stop;
      stop = REGS_control(step);
      if (stop) break;
    }
#endif

    latt_zero_force();
    COLL_update();
    wall_update();

    MODEL_collide_multirelaxation();

    LE_apply_LEBC();
    COM_halo();

    /* Colloid bounce-back applied between collision and
     * propagation steps. */

    bounce_back_on_links();
    wall_bounce_back();

    /* There must be no halo updates between bounce back
     * and propagation, as the halo regions hold active f,g */

    propagation();

    TIMER_stop(TIMER_STEPS);

    /* Configuration dump */

    if (is_config_step()) {
      COM_write_site(get_output_config_filename(step), MODEL_write_site);
      sprintf(filename, "%s%6.6d", "config.cds", step);
      CIO_write_state(filename);
    }

    /* Measurements */

    if (is_measurement_step()) {	  
      info("Wrting phi file at  at step %d!\n", step);
      /*COLL_compute_phi_missing();*/
      sprintf(filename,"phi-%6.6d",step);
      COM_write_site(filename, MODEL_write_phi);
      TIMER_start(TIMER_IO);
      sprintf(filename, "%s%6.6d", "config.cds", step);
      CIO_write_state(filename);
      TIMER_stop(TIMER_IO);
    }

    /* Print progress report */

    if (is_statistics_step()) {

      MISC_curvature();
      TEST_statistics();
      TEST_momentum();
#ifdef _NOISE_
      TEST_fluid_temperature();
#endif

      info("\nCompleted cycle %d\n", step);
    }

    /* Next time step */
  }


  /* Dump the final configuration if required. */

  if (is_config_at_end()) {
    COM_write_site(get_output_config_filename(step), MODEL_write_site);
    sprintf(filename, "%s%6.6d", "config.cds", step);
    CIO_write_state(filename);
  }

  /* Shut down cleanly. Give the timer statistics. Finalise PE. */

  COLL_finish();
  wall_finish();

  TIMER_stop(TIMER_TOTAL);
  TIMER_statistics();

  pe_finalise();
  REGS_finish();

  return 0;
}
