import os, math, uuid, sys, random
import numpy as np
import pandas as pd
from numpy.lib.function_base import average
import utils 
import time
from datetime import datetime

import concurrent.futures
import multiprocessing
import shutil

##############################################################################
# df_pickle = "./train_reg_0.04_pdDataframe_26000.pkl"
# df = pd.read_pickle(df_pickle)

samples                     = 1 #len(df)           # no. of datasets to produce
freestream_angle            = math.pi / 8.  # -angle ... angle
freestream_length           = 10.           # len * (1. ... factor)
freestream_length_factor    = 10.           # length factor
number_cores                = 2

#number of elements should be equal to number of samples
list_xvel = "" #df['u_vel']
list_yvel = "" #df['v_vel']

exclude_low_angle = False
low_angle         = math.radians(21)

airfoil_database  = "airfoil_database/"
airfoil_choice    = ""#df['airfoil']

output_dir        = "./train/"

save_data_pictures          = False
debug_messages              = False

seperate_case_directories   = True  #KEEP TRUE FOR EFFECTIVE MULTIPROCESSING
multi_processing            = True  
delete_case_directory       = False

##############################################################################

seed = random.randint(0, 2**32 - 1)
np.random.seed(seed)
print("Seed: {}".format(seed))

def genMesh(airfoilFile,run_number=0):
    try:
        if(debug_messages): print("Run {}\tRunning: np.loadtext({})".format(run_number,airfoilFile))
        ar = np.loadtxt(airfoilFile, skiprows=1)
    except:
        print("Run {}\tError occured in loadtxt".format(run_number))
        return(-1)
    finally:
        # removing duplicate end point
        if np.max(np.abs(ar[0] - ar[(ar.shape[0]-1)]))<1e-6:
            ar = ar[:-1]

        output = ""
        pointIndex = 1000
        for n in range(ar.shape[0]):
            output += "Point({}) = {{ {}, {}, 0.00000000, 0.005}};\n".format(pointIndex, ar[n][0], ar[n][1])
            pointIndex += 1

        if(debug_messages): print("Run {}\tWriting airfoil.geo".format(run_number))
        with open("airfoil_template.geo", "rt") as inFile:
            with open("airfoil.geo", "wt") as outFile:
                for line in inFile:
                    line = line.replace("POINTS", "{}".format(output))
                    line = line.replace("LAST_POINT_INDEX", "{}".format(pointIndex-1))
                    outFile.write(line)

        if(debug_messages): print("Run {}\tRunning: gmsh airfoil.geo -3 -o airfoil.msh > /dev/null".format(run_number))
        if os.system("gmsh airfoil.geo -3 -o airfoil.msh -format msh2 > /dev/null") != 0:
            print("error during mesh creation!")
            return(-1)

        if(debug_messages): print("Run {}\tRunning: gmshToFoam airfoil.msh > /dev/null".format(run_number))
        if os.system("gmshToFoam airfoil.msh > /dev/null") != 0:
            print("error during conversion to OpenFoam mesh!")
            return(-1)

        if(debug_messages): print("Run {}\tWriting constant/polyMesh/boundary".format(run_number))
        with open("constant/polyMesh/boundary", "rt") as inFile:
            with open("constant/polyMesh/boundaryTemp", "wt") as outFile:
                inBlock = False
                inAerofoil = False
                for line in inFile:
                    if "front" in line or "back" in line:
                        inBlock = True
                    elif "aerofoil" in line:
                        inAerofoil = True
                    if inBlock and "type" in line:
                        line = line.replace("patch", "empty")
                        inBlock = False
                    if inAerofoil and "type" in line:
                        line = line.replace("patch", "wall")
                        inAerofoil = False
                    outFile.write(line)
        os.rename("constant/polyMesh/boundaryTemp","constant/polyMesh/boundary")

        return(0)

def runSim(freestreamX, freestreamY, run_number=0):
    if(debug_messages): print("Run {}\tWriting 0/U".format(run_number))
    with open("U_template", "rt") as inFile:
        with open("0/U", "wt") as outFile:
            for line in inFile:
                line = line.replace("VEL_X", "{}".format(freestreamX))
                line = line.replace("VEL_Y", "{}".format(freestreamY))
                outFile.write(line)
    if(debug_messages):
        print("Showing outputs of simpleFoam -help")
        os.system("simpleFoam -help")
    if(debug_messages): print("Run {}\tRunning: ./Allclean && simpleFoam > foam.log".format(run_number))
    os.system("./Allclean && simpleFoam > foam.log")

def outputProcessing(basename, freestreamX, freestreamY, dataDir=output_dir, pfile='OpenFOAM/postProcessing/internalCloud/500/cloud_p.xy', ufile='OpenFOAM/postProcessing/internalCloud/500/cloud_U.xy', res=128, imageIndex=0, run_number=0): 
    # output layout channels:
    # [0] freestream field X + boundary
    # [1] freestream field Y + boundary
    # [2] binary mask for boundary
    # [3] pressure output
    # [4] velocity X output
    # [5] velocity Y output
    npOutput = np.zeros((6, res, res))

    if(debug_messages): print("Run {}\tLoading Pressure output".format(run_number))
    ar = np.loadtxt(pfile)
    curIndex = 0

    for y in range(res):
        for x in range(res):
            xf = (x / res - 0.5) * 2 + 0.5
            yf = (y / res - 0.5) * 2
            if abs(ar[curIndex][0] - xf)<1e-4 and abs(ar[curIndex][1] - yf)<1e-4:
                npOutput[3][x][y] = ar[curIndex][3]
                curIndex += 1
                # fill input as well
                npOutput[0][x][y] = freestreamX
                npOutput[1][x][y] = freestreamY
            else:
                npOutput[3][x][y] = 0
                # fill mask
                npOutput[2][x][y] = 1.0

    if(debug_messages): print("Run {}\tLoading Velocity output".format(run_number))
    ar = np.loadtxt(ufile)
    curIndex = 0

    for y in range(res):
        for x in range(res):
            xf = (x / res - 0.5) * 2 + 0.5
            yf = (y / res - 0.5) * 2
            if abs(ar[curIndex][0] - xf)<1e-4 and abs(ar[curIndex][1] - yf)<1e-4:
                npOutput[4][x][y] = ar[curIndex][3]
                npOutput[5][x][y] = ar[curIndex][4]
                curIndex += 1
            else:
                npOutput[4][x][y] = 0
                npOutput[5][x][y] = 0

    #fileName = str(uuid.uuid4()) # randomized name
    fileName = "%s_%d_%d" % (basename, int(freestreamX*100), int(freestreamY*100) )

    if save_data_pictures:
        if(debug_messages): print("Run {}\tSaving data pictures".format(run_number))
        utils.saveAsImage('data_pictures/'+fileName+'_pressure.png', npOutput[3])
        utils.saveAsImage('data_pictures/'+fileName+'_velX.png', npOutput[4])
        utils.saveAsImage('data_pictures/'+fileName+'_velY.png', npOutput[5])
        utils.saveAsImage('data_pictures/'+fileName+'_inputX.png', npOutput[0])
        utils.saveAsImage('data_pictures/'+fileName+'_inputY.png', npOutput[1])
        utils.saveAsImage('data_pictures/'+fileName+'_mask.png', npOutput[2])

    print("Run {}\tsaving in ".format(run_number) + dataDir + fileName + ".npz")
    np.savez_compressed(dataDir + fileName, a=npOutput)


files = os.listdir(airfoil_database)
files.sort()
if len(files)==0:
	print("error - no airfoils found in %s" % airfoil_database)
	exit(1)

with multiprocessing.Manager() as manager:
    lock = manager.Lock()
    samples_left = manager.Value(int,samples)
    time_taken = manager.Value(float,0)
    average_time_taken = manager.Value(float,0)

    def dataGen(n):
        sample_start_time = time.time()

        print("Run {}/{}:".format(n,samples))

        if(len(airfoil_choice) == 0):
            fileNumber = np.random.RandomState().randint(0, len(files))   
        else:
            try:
                fileNumber = files.index(airfoil_choice[n])
            except:
                print("Run {} \t{} Airfoil not found".format(n,airfoil_choice[n]))
                return (-1)
        
        basename = os.path.splitext( os.path.basename(files[fileNumber]) )[0]
        print("Run {}\tusing {}".format(n,files[fileNumber]))

        if(len(list_xvel)==0 or len(list_yvel)==0):
            length = freestream_length * np.random.RandomState().uniform(1.,freestream_length_factor) 
            angle  = np.random.RandomState().uniform(-freestream_angle, freestream_angle) 
            if(exclude_low_angle):
                while((angle>(-1)*low_angle and angle<0) or (angle<low_angle and angle>0)):
                    angle  = np.random.RandomState().uniform(-freestream_angle, freestream_angle) 
            print("Run {}\tUsing len {} angle {}".format(n,length,angle))
            fsX =  math.cos(angle) * length
            fsY = -math.sin(angle) * length 
        else:
            fsX = list_xvel[n]
            fsY = list_yvel[n]

        print("Run {}\tResulting freestream vel x,y: {},{}".format(n,fsX,fsY))

        if(seperate_case_directories):
            case_dir = "./OpenFOAM_" + str(n) + "/"
            if(os.path.isdir(case_dir)):
                shutil.rmtree(case_dir)

        else:
            case_dir = "./OpenFOAM/"

        if(seperate_case_directories):
            if(debug_messages): print("Run {}\tCopying ./OpenFOAM to {}".format(n,case_dir))
            shutil.copytree("./OpenFOAM_ref/",case_dir)

        os.chdir(case_dir)
        if genMesh("../" + airfoil_database + files[fileNumber],run_number= n) != 0:
            print("Run {}\tmesh generation failed, aborting".format(n))
            os.chdir("..")
            return (-1)

        runSim(fsX, fsY,run_number= n)
        os.chdir("..")

        outputProcessing(basename, fsX, fsY, imageIndex=n,run_number=n,pfile=case_dir+'postProcessing/internalCloud/500/cloud_p.xy', ufile=case_dir+'postProcessing/internalCloud/500/cloud_U.xy')
        print("Run {}/{}\tdone".format(n,samples))

        if(seperate_case_directories):
            if(delete_case_directory):
                if(debug_messages): print("Run {}\tremoving {}".format(n,case_dir))
                shutil.rmtree(case_dir)

        sample_end_time = time.time()
        sample_time_taken = (sample_end_time - sample_start_time)/60     #minutes
        print("Run {}\tTime taken for generating sample, {} minutes".format(n,sample_time_taken))

        current_time = time.time()
        with lock:
            samples_left.value -= 1
            time_taken.value = (current_time - start_time)
            average_time_taken.value += time_taken.value
            if(debug_messages): print("Run {}\t{} samples left, {} minutes passed till now".format(n,samples_left.value,time_taken.value))
            estimated_time_to_complete = (samples_left.value * time_taken.value) / (samples - samples_left.value)
            # print("\nRun {}\tEstimated time left: {} seconds ({} minutes)\n".format(n,estimated_time_to_complete,estimated_time_to_complete/60))
            print("\nRun {}\tETC: {}\n".format(n, datetime.fromtimestamp( current_time + estimated_time_to_complete )))


    if(multi_processing == True and seperate_case_directories == False):
        print("seperate_case_driectories should be True if multiprocessing is True")
        exit()
    if(len(list_xvel)!=0 or len(list_yvel)!=0):
        if(len(list_xvel)!=samples or len(list_yvel)!=samples):
            print("number of elements in list_xvel and list_yvel should be equal to number of samples")
            exit()
        else:
            pass
    utils.makeDirs( ["./data_pictures", output_dir, "./OpenFOAM/constant/polyMesh/sets", "./OpenFOAM/constant/polyMesh"] )

    start_time = time.time()

    if(multi_processing):
        with concurrent.futures.ProcessPoolExecutor(max_workers=number_cores) as executor:
            executor.map(dataGen,range(samples))
    else:
        for n in range(samples):
            dataGen(n)

    average_time_taken.value /= samples
    end_time = time.time()
    total_time_taken = (end_time - start_time)/60     #minutes
    print("Time taken for generating {} sample/s, {} minutes".format(samples,total_time_taken))
    print("Average time taken per sample: {} seconds".format(average_time_taken.value))

