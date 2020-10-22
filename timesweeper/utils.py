import argparse
import os
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='A set of functions that run slurm \
                                                  jobs to create and parse SLiM \
                                                  simulations for sweep detection.')

    parser.add_argument('-f', '--function', 
                        metavar='SCRIPT_FUNCTION',
                        help='Use one of the available \
                            functions by specifying its name.',
                        required=True, 
                        dest='run_func', 
                        type=str,
                        choices=['launch_sims',
                                 'clean_sims',
                                 'create_feat_vecs',
                                 'train_nets'])

    parser.add_argument('-s', '--slim-paramfile',
                        metavar='SLIM_SIMULATION_FILE',
                        help='Filename of slimfile in /slimfiles/ dir.\
                              New directory will be created with this as prefix \
                              and will contain all the relevant files for this \
                              set of parameters.',
                        dest='slim_name',
                        type=str,
                        required=False,
                        default='test.slim')

    args = parser.parse_args()

    return args

def run_batch_job(cmd, jobName, launchFile, wallTime, qName, mbMem, logFile):
    with open(launchFile,"w") as f:
        f.write("#!/bin/bash\n")
        f.write("#SBATCH --job-name=%s\n" %(jobName))
        f.write("#SBATCH --time=%s\n" %(wallTime))
        f.write("#SBATCH --partition=%s\n" %(qName))
        f.write("#SBATCH --output=%s\n" %(logFile))
        f.write("#SBATCH --mem=%s\n" %(mbMem))
        f.write("#SBATCH --requeue\n")
        f.write("#SBATCH --export=ALL\n")
        f.write("\n%s\n" %(cmd))
    os.system("sbatch %s" %(launchFile))
    
def clean_msOut(msFile):
    """Reads in MS-style output from Slim, removes all extraneous information \
        so that the MS output is the only thing left. Writes to "cleaned_" slimfile.

    Args:
        msFile (str): Filepath of slim output file.
    """
    filepath = os.path.split(msFile)[0]
    filename = os.path.split(msFile)[1]

    with open(msFile, 'r') as rawfile:
        rawMS = [i.strip() for i in rawfile.readlines()]

    #Filter out lines that have integers after them
    cleanMS = []
    listMS = [i for i in rawMS if i]
    for i in range(len(listMS)):
        #print(listMS[i].split())
        #Filter out lines where integer line immediately follows
        if ((listMS[i] == '// Initial random seed:') 
            or (listMS[i] == '// Starting run at generation <start>:')
            or (listMS[i-1] == '// Initial random seed:') 
            or (listMS[i-1] == '// Starting run at generation <start>:')):
            continue

        #Filter out commented lines that aren't ms related
        #Get rid of lines like '// RunInitializeCallbacks():'
        elif ((listMS[i].split()[0] == '//') and (len(listMS[i].split()) > 1)):        
            continue

        #Capture SHIC-required header, make first line
        elif listMS[i].split()[0] == 'SLiM/build/slim':
            shic_header = listMS[i]
        else:
            cleanMS.append(listMS[i])

    #Filter out everything else that isn't ms related
    cleanMS = [i for i in cleanMS if ';' not in i]
    cleanMS = [i for i in cleanMS if '#' not in i]
    
    try:    
        foo = cleanMS.pop(0) #Remove slimfile name
    except:
        pass

    try:
        cleanMS.insert(0, shic_header)
    except:
        print("why does this throw an error but still work?")
    
    if not os.path.exists(os.path.join(filepath, 'cleaned')):
        os.mkdir(os.path.join(filepath, 'cleaned'))
        
    with open(os.path.join(filepath, 'cleaned_' + filename), 'w') as outFile:
        outFile.write('\n'.join(cleanMS))
        
