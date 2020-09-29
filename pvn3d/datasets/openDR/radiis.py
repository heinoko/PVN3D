import numpy as np

radii = []
for i in range(0,10):

	corners = np.loadtxt('./openDR_object_kps/'+str(i+1)+'/corners.txt')

	x_dis = corners[:,0].max() - corners[:,0].min()
	y_dis = corners[:,1].max() - corners[:,1].min()
	z_dis = corners[:,2].max() - corners[:,2].min()

	diag = np.sqrt(x_dis**2 + y_dis**2 + z_dis**2)
	radii.append( diag / 2 )

np.savetxt('./dataset_config/radius.txt', radii)
