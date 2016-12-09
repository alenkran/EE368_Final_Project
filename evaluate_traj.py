# Filename: evaulate_traj.py
# Description: Contains functions to evaluate the reconstructed trajectories
# Authors: Christian Choe, Min Cheol Kim

import numpy as np

# This function returns the average RMSD error between each atom between each frame
def get_RMSD_error(md_obj):
	num_frames, num_atoms, num_axis = md_obj.xyz.shape
	recon = np.reshape(md_obj.xyz, (num_frames, num_atoms*num_axis))

	# Construct difference operator
	temp1 = np.identity(num_frames-1)
	temp1 = np.concatenate((temp1, np.zeros((num_frames-1, 1))), axis=1)
	temp2 = -1*np.identity(num_frames-1)
	temp2 = np.concatenate((np.zeros((num_frames-1, 1)), temp2), axis=1)
	difference_operator = temp1+temp2

	# Get difference matrix (each row is the difference between frames)
	difference_matrix = difference_operator.dot(recon)
	return np.sqrt((np.linalg.norm(difference_matrix)**2)/( (num_frames-1)*(num_atoms*num_axis)))
