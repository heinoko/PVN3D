
train_file = open(r"train.txt",'w')
test_file = open(r"test.txt",'w')
train_file.truncate()
test_file.truncate()

j=0
for i in range(0,100):
	
	if j<4:
		train_file.write(str(i)+'\n')
		j+=1
	else:
		test_file.write(str(i)+'\n')
		j=0
	
	#test_file.write(str(i)+'\n')
