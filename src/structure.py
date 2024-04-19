#!/usr/bin/env python3

#    MIT License

#    COPYRIGHT (C) 2024 MERCK SHARP & DOHME CORP. A SUBSIDIARY OF MERCK & CO., 
#    INC., RAHWAY, NJ, USA. ALL RIGHTS RESERVED

#    Permission is hereby granted, free of charge, to any person obtaining a copy
#    of this software and associated documentation files (the "Software"), to deal
#    in the Software without restriction, including without limitation the rights
#    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#    copies of the Software, and to permit persons to whom the Software is
#    furnished to do so, subject to the following conditions:

#    The above copyright notice and this permission notice shall be included in all
#    copies or substantial portions of the Software.

#    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#    SOFTWARE.

"""preprocessing script to generate structures using ImmuneBuilder"""

import numpy as np
from sys import stdout
import torch


# immune builder structure prediciton (similar to igfold, AF2, etc.)
def immune_builder(sequence, output="mAb.pdb"):
    from ImmuneBuilder import ABodyBuilder2
    predictor = ABodyBuilder2()
    antibody = predictor.predict(sequence)
    antibody.save(output)
    


# if name is main executable
if __name__ == "__main__":
    
    # read arguments
    import argparse
    parser = argparse.ArgumentParser(description='ImmuneBuilder PDBs')
    parser.add_argument('--h', type=str, required=False, help='heavy chain amino acid sequence')
    parser.add_argument('--l', type=str, required=False,help='light chain amino acid sequence')
    parser.add_argument('--output', type=str, default='mAb.pdb', help='output file name')
    args = parser.parse_args()
    
    if args.h and args.l:
        sequence = {
            'H': args.h,
            'L': args.l}
        immune_builder(sequence, args.output)
    elif args.h or args.l:
        print('Error: must provide both heavy and light chain sequences')
        exit()

