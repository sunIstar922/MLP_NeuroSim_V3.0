/*******************************************************************************
* Copyright (c) 2015-2017
* School of Electrical, Computer and Energy Engineering, Arizona State University
* PI: Prof. Shimeng Yu
* All rights reserved.
*   
* This source code is part of NeuroSim - a device-circuit-algorithm framework to benchmark 
* neuro-inspired architectures with synaptic devices(e.g., SRAM and emerging non-volatile memory). 
* Copyright of the model is maintained by the developers, and the model is distributed under 
* the terms of the Creative Commons Attribution-NonCommercial 4.0 International Public License 
* http://creativecommons.org/licenses/by-nc/4.0/legalcode.
* The source code is free and you can redistribute and/or modify it
* by providing that the following conditions are met:
*   
*  1) Redistributions of source code must retain the above copyright notice,
*     this list of conditions and the following disclaimer. 
*   
*  2) Redistributions in binary form must reproduce the above copyright notice,
*     this list of conditions and the following disclaimer in the documentation
*     and/or other materials provided with the distribution.
*   
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
* ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
* WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
* DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
* FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
* DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
* SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
* CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
* OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
* OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
* 
* Developer list: 
*   Pai-Yu Chen     Email: pchen72 at asu dot edu 
*                     
*   Xiaochen Peng   Email: xpeng15 at asu dot edu
********************************************************************************/

#include <cstdio>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <stdlib.h>
#include <random>
#include <vector>
#include "Cell.h"
#include "Array.h"
#include "formula.h"
#include "NeuroSim.h"
#include "Param.h"
#include "IO.h"
#include "Train.h"
#include "Test.h"
#include "Mapping.h"
#include "Definition.h"
#include "omp.h"

using namespace std;

int main() {
    gen.seed(0);
    
    // Load in MNIST data
    ReadTrainingDataFromFile("patch60000_train.txt", "label60000_train.txt");
    ReadTestingDataFromFile("patch10000_test.txt", "label10000_test.txt");

    // Initialization of synaptic array from input to hidden layer
    arrayIH->Initialization<RealDevice>(); 

    // Initialization of synaptic array from hidden to output layer
    arrayHO->Initialization<RealDevice>();

    omp_set_num_threads(16);

    // Initialization of NeuroSim synaptic cores
    param->relaxArrayCellWidth = 0;
    NeuroSimSubArrayInitialize(subArrayIH, arrayIH, inputParameterIH, techIH, cellIH);
    param->relaxArrayCellWidth = 1;
    NeuroSimSubArrayInitialize(subArrayHO, arrayHO, inputParameterHO, techHO, cellHO);
    
    // Calculate synaptic core area and leakage power
    NeuroSimSubArrayArea(subArrayIH);
    NeuroSimSubArrayArea(subArrayHO);
    NeuroSimSubArrayLeakagePower(subArrayIH);
    NeuroSimSubArrayLeakagePower(subArrayHO);

    // Initialize the neuron peripheries
    NeuroSimNeuronInitialize(subArrayIH, inputParameterIH, techIH, cellIH, adderIH, muxIH, muxDecoderIH, dffIH, subtractorIH);
    NeuroSimNeuronInitialize(subArrayHO, inputParameterHO, techHO, cellHO, adderHO, muxHO, muxDecoderHO, dffHO, subtractorHO);

    // Create and open output CSV file
    ofstream mywriteoutfile;
    mywriteoutfile.open("output.csv");
    
    // Write header to the CSV file
    mywriteoutfile << "Epoch, Accuracy (%), Read Latency (s), Write Latency (s), Read Energy (J), Write Energy (J)" << endl;

    // Training loop
    for (int i = 1; i <= param->totalNumEpochs / param->interNumEpochs; i++) {
        Train(param->numTrainImagesPerEpoch, param->interNumEpochs, param->optimization_type);
        if (!param->useHardwareInTraining && param->useHardwareInTestingFF) {
            WeightToConductance();
        }
        Validate();
        
        if (HybridCell* temp = dynamic_cast<HybridCell*>(arrayIH->cell[0][0])) {
            WeightTransfer();
        } else if (_2T1F* temp = dynamic_cast<_2T1F*>(arrayIH->cell[0][0])) {
            WeightTransfer_2T1F();
        }

        // Calculate performance metrics
        double accuracy = (double)correct / param->numMnistTestImages * 100;
        double readLatency = subArrayIH->readLatency + subArrayHO->readLatency;
        double writeLatency = subArrayIH->writeLatency + subArrayHO->writeLatency;
        double readEnergy = arrayIH->readEnergy + subArrayIH->readDynamicEnergy + arrayHO->readEnergy + subArrayHO->readDynamicEnergy;
        double writeEnergy = arrayIH->writeEnergy + subArrayIH->writeDynamicEnergy + arrayHO->writeEnergy + subArrayHO->writeDynamicEnergy;

        // Write metrics to output.csv
        mywriteoutfile << i * param->interNumEpochs << ", " << accuracy << ", " 
                       << readLatency << ", " << writeLatency << ", " 
                       << readEnergy << ", " << writeEnergy << endl;

        // Print metrics to console
        printf("Accuracy at %d epochs is: %.2f%\n", i * param->interNumEpochs, accuracy);
        printf("\tRead latency=%.4e s\n", readLatency);
        printf("\tWrite latency=%.4e s\n", writeLatency);
        printf("\tRead energy=%.4e J\n", readEnergy);
        printf("\tWrite energy=%.4e J\n", writeEnergy);
    }

    // Close the output file
    mywriteoutfile.close();

    return 0;
}


