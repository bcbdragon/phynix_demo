#!/usr/bin/env python3
'''
./watch_file.py -p python3 minibayes.py -d /Users/bennettbullock/minibayes
'''
import mmh3
import os
import sys
import argparse
import subprocess
import time
from copy import deepcopy

def dir_hash(dirName):

    fileContents=[]

    for subdir, dirs, files in os.walk(dirName):

        for file in files:

            if file.endswith('.py') or file.endswith('.txt'):

                fileName=os.path.join(subdir,file)

                with open(fileName,'r') as f:

                    fileContents.append(f.read())

    return mmh3.hash(' '.join(fileContents))


def dir_hashes(dirs):

    return {dr:dir_hash(dr) for dr in dirs}



if __name__=='__main__':

    parser=argparse.ArgumentParser()

    parser.add_argument('-p1',
                        dest='process1',
                        nargs='+')
    parser.add_argument('-p2',
                        dest='process2',
                        nargs='+')
    parser.add_argument('-ps1',
                        dest='process_string_1',
                        type=str)
    parser.add_argument('-ps2',
                        dest='process_string_2',
                        type=str)
    parser.add_argument('-d',
                        dest='dirs',
                        nargs='+')

    args=parser.parse_args()

    dirs=args.dirs

    if args.process1:

        process1=args.process1

    elif args.process_string_1:

        process1=args.process_string_1.split(' ')

    else:

        raise Exception('Process 1 not provided')

    process2=None

    if args.process2:

        process2=args.process2

    elif args.process_string_2:

        process2=args.process_string_2.split(' ')
    
    lastDirHashes={}
    dirHashes=dir_hashes(dirs)
    
    while True:

        time.sleep(1)

        if lastDirHashes != dirHashes:

            print('*'*30)
            print('file changed, rerunning subprocess')

            subprocess.run(process1)

            if process2 is not None:

                print(process2)

                subprocess.run(process2)

        lastDirHashes=dirHashes        
        dirHashes=deepcopy(dir_hashes(dirs))
