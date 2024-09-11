#!/usr/bin/env python
# Copied description and defaults from codabench/iris/bundle/ingestion_program/ingestion.py

# Usage: python ingestion.py input_dir output_dir ingestion_program_dir submission_program_dir

# AS A PARTICIPANT, DO NOT MODIFY THIS CODE.
#
# This is the "ingestion program" written by the organizers.
# This program also runs on the challenge platform to test your code.
#
# ALL INFORMATION, SOFTWARE, DOCUMENTATION, AND DATA ARE PROVIDED "AS-IS".
# ISABELLE GUYON, CHALEARN, AND/OR OTHER ORGANIZERS OR CODE AUTHORS DISCLAIM
# ANY EXPRESSED OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR ANY PARTICULAR PURPOSE, AND THE
# WARRANTY OF NON-INFRIGEMENT OF ANY THIRD PARTY'S INTELLECTUAL PROPERTY RIGHTS.
# IN NO EVENT SHALL ISABELLE GUYON AND/OR OTHER ORGANIZERS BE LIABLE FOR ANY SPECIAL,
# INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER ARISING OUT OF OR IN
# CONNECTION WITH THE USE OR PERFORMANCE OF SOFTWARE, DOCUMENTS, MATERIALS,
# PUBLICATIONS, OR INFORMATION MADE AVAILABLE FOR THE CHALLENGE.
#
# Main contributors: Isabelle Guyon and Arthur Pesah, March-October 2014
# Lukasz Romaszko April 2015
# Originally inspired by code code: Ben Hamner, Kaggle, March 2013
# Modified by Ivan Judson and Christophe Poulain, Microsoft, December 2013
# Last modifications Isabelle Guyon, October 2017

# ===== Begin Imageomics modifications =====
import os
import re
from sys import argv, path, executable, exit
import subprocess
import time
from datetime import datetime, timezone
from packaging.version import Version, InvalidVersion


def install_from_whitelist(req_file, program_dir):
    whitelist = open(os.path.join(program_dir,"whitelist.txt"), 'r').readlines()
    whitelist = [i.rstrip('\n') for i in whitelist]
    # print(whitelist)

    for package in open(req_file, 'r').readlines():
        package = package.rstrip('\n')
        package_version = package.split("==")
        if len(package_version) > 2:
            # invalid format, don't use
            print(f"requested package {package} has invalid format, will install latest version (of {package_version[0]}) if allowed")
            package = package_version[0]
        elif len(package_version) == 2:
            version_str = package_version[1]
            try:
                 Version(version_str)
            except InvalidVersion:
                 exit(f"requested package {package} has invalid version, please check that {version_str} is the correct version of {package_version[0]}.")
            #     package = package_version[0]

                
        #print("accepted package name: ", package)
        #print("package name ", package_version[0])
        if package_version[0] in whitelist:
            # package must be in whitelist, so format check unnecessary
            subprocess.check_call([executable, "-m", "pip", "install", package])
            print(f"{package_version[0]} installed")
        else:
            exit(f"{package_version[0]} is not an allowed package. Please contact the organizers on GitHub to request acceptance of the package.")


if __name__ == "__main__":
    
    print("We're running ingestion")

    # Get the current UTC time
    current_time_utc = datetime.now(timezone.utc)
    # Print the timestamp in UTC
    print("Current UTC Time:", current_time_utc.strftime('%Y-%m-%d %H:%M:%S'))

    #### INPUT/OUTPUT: Get input and output directory names
    input_dir = os.path.abspath(argv[1])
    output_dir = os.path.abspath(argv[2])
    program_dir = os.path.abspath(argv[3])
    submission_dir = os.path.abspath(argv[4])
    
    print("Using input_dir: " + input_dir)
    print("Using output_dir: " + output_dir)
    print("Using program_dir: " + program_dir)
    print("Using submission_dir: " + submission_dir)

    path.append(program_dir) # In order to access libraries from our own code
    path.append(submission_dir) # In order to access libraries of the user

    start = time.time()
    requirements_file = os.path.join(submission_dir, "requirements.txt")
    if os.path.isfile(requirements_file):
        install_from_whitelist(requirements_file, program_dir)
    end = time.time()

    elapsed = time.strftime("%H:%M:%S", time.gmtime(end - start))

    print(f"pip handling packages takes {elapsed}.")

    # Import remaining packages
    from PIL import Image
    from tqdm import tqdm
    from model import Model
    

    print("model imported")
    submit_model = Model()
    submit_model.load()

    if hasattr(submit_model, "device"):
        print(f"model running on device: {submit_model.device}")

    img_list = os.listdir(input_dir)
    num_of_datapoint = len(img_list)

    with open(os.path.join(output_dir, "predictions.txt"), 'w') as f:
        print("predictions file opened")
        start = time.time()
        for idx, filename in tqdm(enumerate(img_list), total=num_of_datapoint):
            try:
                image_path = os.path.join(input_dir, filename)
                if os.path.isdir(image_path):
                    continue
                datapoint = Image.open(image_path)
            except Exception as e:
                print(f"{image_path}: {e}")
                continue
            
            score = submit_model.predict(datapoint)

            if idx ==  num_of_datapoint - 1:
                f.write(filename + "," + str(round(score, 4)))
            else:
                f.write(filename + "," + str(round(score, 4)) + '\n')
        end = time.time()
        elapsed = time.strftime("%H:%M:%S", time.gmtime(end - start))

        print(f"model inference takes {elapsed}.")

        print(f"we looped {idx} times")       
