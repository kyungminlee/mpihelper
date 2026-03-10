#pragma once
#include "MPIHelper.hh"
#include "Data.hh"
#include <unordered_map>
#include <string>


inline void fullExchange(
  RealRegionScalar const & inData,
  RealRegionScalar & outData,
  MPIHelper & helper,
  bool average=true
) {
  Mesh const & globalMesh = outData.getMesh();
  Mesh const & localMesh  = inData.getMesh();
  int globalSize = (int)globalMesh.getRegionSize();

  // Build global region name -> index map
  std::unordered_map<std::string, int> globalIndex;
  for (int i = 0; i < globalSize; ++i) {
    globalIndex[globalMesh.getRegionName(i)] = i;
  }

  // Accumulation buffers over the global mesh, zeroed for regions not in local mesh
  std::vector<double> sendVals(globalSize, 0.0);
  std::vector<int>    sendCounts(globalSize, 0);

  for (int li = 0; li < (int)localMesh.getRegionSize(); ++li) {
    auto it = globalIndex.find(localMesh.getRegionName(li));
    if (it != globalIndex.end()) {
      int gi = it->second;
      sendVals[gi]   = inData[li];
      sendCounts[gi] = 1;
    }
  }

  // Reduce sums across all ranks
  std::vector<double> sumVals(globalSize);
  helper.allreduce(sendVals.data(), sumVals.data(), globalSize, MPI_SUM);

  if (average) {
    std::vector<int> totalCounts(globalSize);
    helper.allreduce(sendCounts.data(), totalCounts.data(), globalSize, MPI_SUM);
    for (int i = 0; i < globalSize; ++i) {
      outData[i] = totalCounts[i] > 0 ? sumVals[i] / totalCounts[i] : 0.0;
    }
  } else {
    for (int i = 0; i < globalSize; ++i) {
      outData[i] = sumVals[i];
    }
  }
}
