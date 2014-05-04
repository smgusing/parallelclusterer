// hey, how do header files work?

/*
 * 
 *                This source code is part of
 * 
 *                 G   R   O   M   A   C   S
 * 
 *          GROningen MAchine for Chemical Simulations
 * 
 *                        VERSION 3.2.0
 * Written by David van der Spoel, Erik Lindahl, Berk Hess, and others.
 * Copyright (c) 1991-2000, University of Groningen, The Netherlands.
 * Copyright (c) 2001-2004, The GROMACS development team,
 * check out http://www.gromacs.org for more information.

 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 * 
 * If you want to redistribute modifications, please consider that
 * scientific software is very special. Version control is crucial -
 * bugs must be traceable. We will be happy to consider code for
 * inclusion in the official distribution, but derived work must not
 * be called official GROMACS. Details are found in the README & COPYING
 * files - if they are missing, get the official version at www.gromacs.org.
 * 
 * To help us fund GROMACS development, we humbly ask that you cite
 * the papers on the package - you can find them in the top README file.
 * 
 * For more info, check our website at http://www.gromacs.org
 * 
 * And Hey:
 * Gromacs Runs On Most of All Computer Systems
 */

#ifndef _do_fit_h
#define _do_fit_h

#include "typedefs.h"

#ifdef __cplusplus
extern "C" {
#endif


// ================================================================
/* Vector operations. */
inline void set_rvec_to_zero(rvec v);
inline void rvec_inplace_add(rvec v, rvec w);


// ================================================================
/* Removes the center-of-mass from a single frame. */
void remove_frame_center_of_mass(
    // particle positions, number of dimensions to consider
    // (the first ndim dimensions will be considered)
    rvec *object_frame, int ndim,
    // particle masses
    const real *masses, real total_mass,
    // center-of-mass mask (for considering only a subset of atoms in computing the center-of-mass)
    int domain_size, const atom_id *domain_indices,
    // shift mask (for applying the center-of-mass shift to only a subset of atoms)
    int shift_size, const int *shift_indices);


// ================================================================
/*  Removes the center-of-mass from a trajectory (an array of frames)
    by applying remove_frame_center_of_mass to each one,
    in parallel using mp-parallelization. */
void remove_traj_center_of_mass(
    // frames, frame metadata
    rvec *frame_array, int frame_array_size, int number_atoms, int ndim,
    // fitting size (lengths of the following arrays), frame indices, frame weights
    int fitting_size, int *fitting_indices, real *fitting_weights);


// ================================================================
/*  Computes the minimum RSMD with rotation (best least-squares fitting, approximately)
    between trajectories centered on the origin
    in parallel using mp-parallelization.
    
    - distance_onetomany:           Writes the list of RMSDs to a buffer.
    - count_neighbours_onetomany:   Returns a count of the RMSDs within a cutoff. 
    - list_neighbours_onetomany:    Writes the list of globalIDs having minimum RMSD within a cutoff.
*/

real distance(rvec *frame0, rvec *frame1, int frame_size, int number_atoms, int number_dimensions,
              real *fitting_weights, int *rms_indices, real *rms_weights, int rms_size);

void oneToMany_computeRmsd(
    rvec *reference_frame, rvec *frame_array, int frame_array_size, int number_atoms, int number_dimensions,
    real *fitting_weights, int *rms_indices, real *rms_weights, int rms_size, real *rmsd,
    int *mask, real mask_dummy_value);

int oneToMany_countWithinRmsd(
    real cutoff, int *neighbour_counts,
    rvec *reference_frame, rvec *frame_array, int frame_array_size, int number_atoms, int number_dimensions,
    real *fitting_weights, int *rms_indices, real *rms_weights, int rms_size,
    int *mask);

void manyToMany_countWithinRmsd(
    real cutoff, int number_atoms, int number_dimensions, int rms_size,
    int *rms_indices, real *rms_weights, real *fitting_weights,
    rvec *frame_array0, int *count_buffer0, int frame_array_size0,
    rvec *frame_array1, int *count_buffer1, int frame_array_size1);

void distance_onetomany(
    rvec *reference_frame, rvec *frame_array, int frame_array_size, int number_atoms, int ndim,
    real *fitting_weights, int *rms_indices, real *rms_weights, int rms_size, real *rmsd);

/* Old functions. 
int count_neighbours_onetomany(
    rvec *reference_frame, rvec *frame_array, int frame_array_size, int number_atoms, int ndim,
    real *fitting_weights, int *rms_indices, real *rms_weights, int rms_size, real cutoff);
*/
// ================================================================
/* Gromacs function declarations. */

/* Computes the least-rmsd-fit rotation matrix to fit 'x' to 'xp'. */
//void calc_fit_R(int ndim, int natoms, real *w_rls, rvec *xp, rvec *x, matrix R);

/* Computes the root-mean-squared deviaiton between 'x' and 'xp' for atoms in 'index'.*/
real rmsdev_ind(int nind, atom_id index[], real mass[], rvec x[], rvec xp[]);

/* Do a least squares fit of x to xp. Atoms which have zero mass
 * (w_rls[i]) are not taken into account in fitting.
 * This makes is possible to fit eg. on Calpha atoms and orient
 * all atoms. The routine only fits the rotational part,
 * therefore both xp and x should be centered round the origin.
 */
void do_fit_ndim(int ndim,int natoms,real *w_rls,rvec *xp,rvec *x);

#ifdef __cplusplus
}
#endif

#endif
