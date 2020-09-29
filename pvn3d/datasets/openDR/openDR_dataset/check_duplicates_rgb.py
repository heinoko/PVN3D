import cv2
import numpy as np

mean1 = 0
for i in range(0, 2303):
	
	try:
		a = cv2.imread('./rgb/'+str(i)+'.png', -1)
		b = cv2.imread('./rgb/'+str(i+1)+'.png', -1)
	except Exception as inst:
		print str(inst) 
		continue
	diff = a - b
	mean = abs(np.mean(diff))
	if mean1!=0:
		mean1 = (mean + mean1)/2
	else:
		mean1 += mean

	#print str(i)+'-'+str(i+1)+':'+str(mean)
	if mean < 3:
		print 'Duplicates detected: ' + str(i) +' and ' + str(i+1)
print 'Mean of all images: ' + str(mean1) 

