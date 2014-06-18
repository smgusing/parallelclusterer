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
 #include "gmx_rmsd_custom1.h"
 #include "smalloc.h"
 #include <omp.h>
                                        
 #define EPS 1.0e-09   

 /*
    Alex:
    - Removed dependencies on some simple GROMACS macros, functions
      by writing equivalent definitions within the file.
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



/* Computes the minimum rmsd between two frames allowing for rotation. */
real computeRmsd(rvec *frame0, rvec *frame1, int number_atoms, int number_dimensions,
              real *fitting_weights, int *rms_indices, real *rms_weights, int rms_size)
{
    matrix rotation_matrix;
    
    int i, j, k, n, m;
    real dx, rmsd_acc = 0.0, rmsdz = 0.0, rmsdxy = 0.0, rmsdxymin,ang,val;
    rvec rotated;
    
    for (i = 0; i < rms_size; i++) {
        // Get index.
        if (rms_indices) {
            j = rms_indices[i];
        } else {
            j = i;
        }
        n=number_dimensions-1;
        dx=frame0[j][n]-frame1[j][n];
        rmsdz+=dx*dx;
        for (n = 0; n < number_dimensions-1; n++) {
            dx = frame0[j][n] - frame1[j][n];
            rmsdxy += dx * dx;
        }   

    }
    rmsdxymin=rmsdxy;

    for (k = 1; k < 7; k++ ) {
       ang=k*0.897597901026; //360.0*pi/(180.0*7.0)
       val=cos(ang);
       rotation_matrix[0][0]=val;
       rotation_matrix[1][1]=val;
       val=sin(ang);
       rotation_matrix[0][1]=val;
       rotation_matrix[1][0]=-val;
       rmsdxy=0.0;
       for (i = 0; i < rms_size; i++) {
           // Get index.
           if (rms_indices) {
               j = rms_indices[i];
           } else {
               j = i;
           }

           for (n = 0; n < number_dimensions-1; n++) {
               rotated[n]=0.0;
               for (m = 0; m < number_dimensions-1; m++) {
                   rotated[n] += rotation_matrix[n][m] * frame1[j][m];
               }
           }

           // Find rmsd between frame 0 and rotated frame 1.
           for (n = 0; n < number_dimensions-1; n++) {
               dx = frame0[j][n] - rotated[n];
               rmsdxy += dx * dx;
           }

       }
       if (rmsdxy<rmsdxymin) rmsdxymin=rmsdxy;
    }

    // Normalize using mass.
    return sqrt( (rmsdxymin + rmsdz) / rms_size);
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

void manytomany_between(
    real cutoff, rvec *traj0, rvec *traj1,
    int traj0_size, int traj1_size,
    int *traj0_idx,  int *traj1_idx,
    int *traj0_count, int *traj1_count,
    int traj0_idxsize, int traj1_idxsize,
    int number_atoms, int number_dimensions,
    real *fitting_weights, int *rms_indices, real *rms_weights, int rms_size )
{
	/*
	 * function to count number of neigbours for traj0 and traj1
	 * using only the frames mentioned in traj0_idx and traj1_idx
	 * The counts are updated in traj0_count and traj1_count.
	 * Note: For correct counting, the count array MUST have zeros in them.
	 * and the size should be same as trajsize
	 */
	int i,j;
    real rmsd;
    rvec *reference_frame,*object_frame; // Points to a frame from frame_array.

    #pragma omp parallel
    {
    	int traj0_count_priv[traj0_size];
    	int traj1_count_priv[traj1_size];
		for (i=0; i<traj0_size; i++)
		{
			traj0_count_priv[i] = 0;
		}
		for (j=0; j<traj1_size; j++)
		{
			traj1_count_priv[j] = 0;
		}

    	#pragma omp for private(reference_frame, object_frame, rmsd, j )
    	for (i = 0; i < traj0_idxsize; i++)
    	{
    		reference_frame =  traj0 + (traj0_idx[i] * number_atoms);

    		for (j = 0; j < traj1_idxsize; j++)
    		{
    			object_frame =  traj1 + (traj1_idx[j] * number_atoms);
    			rmsd = computeRmsd(reference_frame, object_frame, number_atoms,
                        number_dimensions, fitting_weights, rms_indices, rms_weights, rms_size);
                if (rmsd <= cutoff)
                {
                	traj0_count_priv[traj0_idx[i]]++;
                	traj1_count_priv[traj1_idx[j]]++;
                }
    		}
    	}

		#pragma omp critical
    	{
    		for (i=0; i<traj0_size; i++)
    		{
    			traj0_count[i] += traj0_count_priv[i];
    		}
    		for (j=0; j<traj1_size; j++)
    		{
    			traj1_count[j] += traj1_count_priv[j];
    		}
    	}
	}

}


void manytomany_within(
    real cutoff, rvec *traj0,
    int traj0_size,
    int *traj0_idx,
    int *traj0_count,
    int traj0_idxsize,
    int number_atoms, int number_dimensions,
    real *fitting_weights, int *rms_indices, real *rms_weights, int rms_size )
{
	/*
	 * function to count number of neigbours for traj0
	 * using only the frames mentioned in traj0_idx
	 * The counts are updated in traj0_count
	 * Note: For correct counting, the count array MUST have zeros in them.
	 * and the size should be same as trajsize
	 *
	 * Note: Self count is not included.
	 */
	int i,j;
    real rmsd;
    rvec *reference_frame,*object_frame; // Points to a frame from frame_array.

    #pragma omp parallel
    {
    	int traj0_count_priv[traj0_size];
		for (i=0; i<traj0_size; i++)
		{
			traj0_count_priv[i] = 0;
		}

    	#pragma omp for private(reference_frame, object_frame, rmsd, j )
    	for (i = 0; i < traj0_idxsize-1; i++)
    	{
    		reference_frame =  traj0 + (traj0_idx[i] * number_atoms);

    		for (j = i+1; j < traj0_idxsize; j++)
    		{
    			object_frame =  traj0 + (traj0_idx[j] * number_atoms);
    			rmsd = computeRmsd(reference_frame, object_frame, number_atoms,
                        number_dimensions, fitting_weights, rms_indices, rms_weights, rms_size);
                if (rmsd <= cutoff)
                {
                	traj0_count_priv[traj0_idx[i]]++;
                	traj0_count_priv[traj0_idx[j]]++;
                }
    		}
    	}

		#pragma omp critical
    	{
    		for (i=0; i<traj0_size; i++)
    		{
    			traj0_count[i] += traj0_count_priv[i];
    		}
    	}
	}

}


// ================================================================

void distance_onetomany(
    rvec *reference_frame, rvec *frame_array, int frame_array_size,
    int number_atoms, int number_dimensions, real *fitting_weights,
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
        rmsd[i] = computeRmsd(reference_frame, object_frame, number_atoms,
                            number_dimensions, fitting_weights, rms_indices, rms_weights, rms_size);
    }
} // fit_onetomany()

