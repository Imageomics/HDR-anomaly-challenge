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
from sys import argv, path, executable
import subprocess
import time


if __name__ == "__main__":
    #### INPUT/OUTPUT: Get input and output directory names
    print("We're running ingestion")

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
    if os.path.isfile(os.path.join(submission_dir, "requirements.txt")):
        subprocess.check_call([executable, "-m", "pip", "install", "-r", os.path.join(submission_dir, "requirements.txt")])
    end = time.time()

    elapsed = time.strftime("%H:%M:%S", time.gmtime(end - start))

    print(f"pip handling packages takes {elapsed}.")

    from PIL import Image
    from tqdm import tqdm
    from model import Model
    

    print("model imported")
    submit_model = Model()
    submit_model.load()



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
                f.write(filename + " " + str(round(score, 4)))
            else:
                f.write(filename + " " + str(round(score, 4)) + '\n')
        end = time.time()
        elapsed = time.strftime("%H:%M:%S", time.gmtime(end - start))

        print(f"model inference takes {elapsed}.")

        print(f"we looped {idx} times")
        
        
