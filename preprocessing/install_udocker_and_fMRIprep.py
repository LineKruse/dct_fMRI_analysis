
#Keeping track of time
from datetime import datetime
now = datetime.now()
print('Time now:',now.strftime("%H:%M:%S"))

#Install udocker 
# importing modules
import os
import pip
#Install udocker
now = datetime.now()
print('Starting to install udocker:',now.strftime("%H:%M:%S"))

#Using os library instead of command line
os.system('python -m pip install udocker')
now = datetime.now()
print('Finishing udocker install:',now.strftime("%H:%M:%S"))
#os.system('export UDOCKER_DIR=/work/74471/udocker')


#Pull fmriprep
now = datetime.now()
print('Starting to pull:',now.strftime("%H:%M:%S"))

os.system('udocker pull nipreps/fmriprep:latest')
now = datetime.now()
print('Finishing pull:',now.strftime("%H:%M:%S"))


#Create a container from the pulled image (calling it fprep)
now = datetime.now()
print('Starting to create image:',now.strftime("%H:%M:%S"))

os.system('udocker create --name=fprep22 nipreps/fmriprep:latest')

now = datetime.now()
print('Finishing image creation:',now.strftime("%H:%M:%S"))

import os
import pip
#Performs sanity checks to verify a image available in the local repository.
print(os.popen('udocker verify fprep22').read())

#Prints container metadata. Applies both to container images or to previously extracted containers, accepts both an image or container id as input.
print(os.popen('udocker inspect -p fprep22').read())

#List images available in the local repository, these are images pulled form Docker Hub, and/or load or imported from files.
print(os.popen('udocker images -l').read())

#List extracted containers. These are not processes but containers extracted and available to the executed with udocker run
print(os.popen('udocker ps').read())

# Run fmriprep
#os.system('udocker run -v /path/to/data:/in -v /path/to/output:/out -v /path/to/fslicense:/fs -v /path/to/tmp:/work fprep /in /out participant --participant-label 01 02 --fs-no-reconall --fs-license-file /fs/license.txt -w /work')