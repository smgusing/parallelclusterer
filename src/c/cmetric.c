/* -*- mode: c; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4; c-file-style: "stroustrup"; -*-
 *
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
 * GROningen Mixture of Alchemy and Childrens' Stories
 */

 #ifdef HAVE_CONFIG_H
 #include <config.h>
 #endif
 
 #include "maths.h"
 #include "sysstuff.h"
 #include "typedefs.h"
 #include "nrjac.h"
 #include "vec.h"
 #include "txtdump.h"
 #include "cmetric.h"
 #include "smalloc.h"
 #include <omp.h>
                                        
 #define EPS 1.0e-09   

 /*
    Alex:
    - Removed dependencies on some simple GROMACS macros, functions
      by writing equiavlent definitions within the file.
 */

// ================================================================
/* Vector operations. */

/* Sets all fields of v to 0.0. */
inline void set_rvec_to_zero(rvec v) {
   v[0] = 0.0; v[1] = 0.0; v[2] = 0.0; 
}


/* Modifies v in place so that it becomes v+w. */
inline void rvec_inplace_add(rvec v, rvec w) {
    v[0] += w[0]; v[1] += w[1]; v[2] += w[2];
}


/* Removes the center-of-mass from a single frame. */
void removeCenterOfMass(
    rvec *object_frame, int number_dimensions, const real *masses, real total_mass,
    int domain_size, const atom_id *domain_indices, // Find the center of these atoms -> shift_vector.
    int shift_size, const int *shift_indices) // Subtract shift_vector from the position of these atoms.
{
    int i, j, dim;
    rvec mass_vector_accumulator; // position vector multiplied by mass scalar
    real mass;

    set_rvec_to_zero(mass_vector_accumulator);

    // Sum 'mass-vectors'.
    if (domain_indices != NULL) { // If we are given a subset of indices, use only those.
        for(i = 0; i < domain_size; i++) {
            j = domain_indices[i];
            for(dim = 0; dim < number_dimensions; dim++) {
                mass_vector_accumulator[dim] += masses[j] * object_frame[j][dim];
            }
        }
    } else {
        // Otherwise, use 'domain_size' of the first vectors.
        for(j = 0; j < domain_size; j++) {
            for(dim = 0; dim < number_dimensions; dim++) {
                mass_vector_accumulator[dim] += masses[j] * object_frame[j][dim];
            }
        }
    } 

    // Normalize mass-vector by mass. 
    for(dim = 0; dim < number_dimensions; dim++) {
        mass_vector_accumulator[dim] /= total_mass;
    }

    // In order to use inplace_add, convert to 'shift_vector'.
    rvec shift_vector;
    for (dim = 0; dim < number_dimensions; dim++) {
        shift_vector[dim] = -mass_vector_accumulator[dim];
    }
    
    if (shift_indices != NULL) { // If we are given a subset of indices, use only those.
        for(i = 0; i < shift_size; i++) {
            j = shift_indices[i];
            rvec_inplace_add(object_frame[j], shift_vector);
        }
    }
    else { // Otherwise, shift only the first 'shift_size' vectors.
        for(j = 0; j < shift_size; j++) {
            rvec_inplace_add(object_frame[j], shift_vector);
        }
    }
}



/* Apply removeCenterOfMass to an array of frames with parallelization. */
void parallelFor_removeCenterOfMass( // *** Changed Name
    rvec *frame_array, int frame_array_size, int number_atoms, int number_dimensions,
    int fitting_size, int *fitting_indices, real *fitting_weights)
{
    int i;
    rvec *object_frame;
    real total_mass = 0.0;

    // Compute total mass.
    for (i = 0; i < fitting_size; i++){
        total_mass += fitting_weights[fitting_indices[i]];
    }

    #pragma omp parallel for default(shared) private(i, object_frame)
    for (i = 0; i < frame_array_size; i++) {
        object_frame = frame_array + (i * number_atoms);
        removeCenterOfMass(object_frame, number_dimensions, fitting_weights, total_mass, fitting_size,
                           fitting_indices, number_atoms, NULL); // shift_indices == NULL: Shift all atoms.
    }
}



/* Computes the minimum rmsd between two frames allowing for rotation. */
// Assumes that the center of mass of the frames is the origin.
real computeRmsd(rvec *frame0, rvec *frame1, int number_atoms, int number_dimensions,
              real *fitting_weights, int *rms_indices, real *rms_weights, int rms_size)
{
    matrix rotation_matrix;
    
    // Compute the rotation that rotates frame1 to best fit frame0.
    calc_fit_R(number_dimensions, number_atoms, fitting_weights,
               frame0, frame1, rotation_matrix); // GROMACS

    int i, j, n, m;
    real dx, total_mass = 0.0, rmsd_acc = 0.0;
    rvec rotated;

    for (i = 0; i < rms_size; i++) {
        // Get index.
        if (rms_indices) {
            j = rms_indices[i];
        } else {
            j = i;
        }

        for (n = 0; n < number_dimensions; n++) {
            rotated[n] = 0.0;
            for (m = 0; m < number_dimensions; m++) {
                rotated[n] += rotation_matrix[n][m] * frame1[j][m];
            }
        }

        // Find rmsd between frame 0 and rotated frame 1.
        for (n = 0; n < number_dimensions; n++) {
            dx = frame0[j][n] - rotated[n];
            rmsd_acc += rms_weights[j] * dx * dx;
        }

        // Accumulate total mass.
        total_mass += rms_weights[j];
    }

    // Normalize using mass.
    return sqrt(rmsd_acc / total_mass);
}

/* Apply computeRmsd to an array of frames with parallelization. */
void oneToMany_computeRmsd(
    rvec *reference_frame, rvec *frame_array, int frame_array_size, int number_atoms, int number_dimensions,
    real *fitting_weights, int *rms_indices, real *rms_weights, int rms_size, real *rmsd,
    int *mask, real mask_dummy_value)
{
    int i;
    rvec *object_frame; // Points to a frame from frame_array.

    #pragma omp parallel for default(shared) private(i,object_frame) 
    for (i = 0; i < frame_array_size; i++) {
        if ((!mask) || (mask[i] == 0)) {
            object_frame = frame_array + (i * number_atoms);
            rmsd[i] = computeRmsd(reference_frame, object_frame, number_atoms,
                                number_dimensions, fitting_weights, rms_indices, rms_weights, rms_size);
        } else {
            rmsd[i] = mask_dummy_value;
        }
    }
}

/* Count the frames within an array of frames within a cutoff rmsd from the reference frame. */
/* Also updates a 'neighbour counting buffer' count_buffer
   which has indices corresponding to that of frame_array. */
int oneToMany_countWithinRmsd(
    real cutoff, int *count_buffer,
    rvec *reference_frame, rvec *frame_array, int frame_array_size, int number_atoms, int number_dimensions,
    real *fitting_weights, int *rms_indices, real *rms_weights, int rms_size,
    int *mask)
{
    int i, count = 0;
    real rmsd;
    rvec *object_frame; // Points to a frame from frame_array.

    #pragma omp parallel for default(shared) private(i,object_frame) reduction(+:count)
    for (i = 0; i < frame_array_size; i++) {
        if ((!mask) || (mask[i] == 0)) {
            object_frame = frame_array + (i * number_atoms);
            rmsd = computeRmsd(reference_frame, object_frame, number_atoms,
                            number_dimensions, fitting_weights, rms_indices, rms_weights, rms_size);
            if (rmsd <= cutoff) {
                if (count_buffer) count_buffer[i]++;
                count++;
            }
        }
    }

    return count;
}

// ================================================================
// Old versions -- will delete when new ones are working.
// Gurpreet: I can use this one for alignment, no need to delete for now

void distance_onetomany(
    rvec *reference_frame, rvec *frame_array, int frame_array_size,
    int number_atoms, int number_dimensions,real *fitting_weights,
    int *rms_indices, real *rms_weights, int rms_size, real *rmsd)
    // line 0: frames, frame metadata
    // line 1: fitting parameters, rmsd parameters. rmsd write buffer.
    // NOTE:   fitting_weights must contain weights for all atoms.
{
    int i;
    rvec *object_frame; // points to a frame from frame_array.

    #pragma omp parallel for default(shared) private(i,object_frame)
    for (i = 0; i < frame_array_size; i++) {
        object_frame = frame_array + (i * number_atoms);
        do_fit_ndim(number_dimensions, number_atoms, fitting_weights,
        		reference_frame, object_frame); // GROMACS
        rmsd[i] = rmsdev_ind(rms_size, rms_indices, rms_weights,
        		  reference_frame, object_frame ); // GROMACS
    }
} // fit_onetomany()

