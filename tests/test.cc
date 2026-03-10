#include "MPIHelper.hh"
#include <gtest/gtest.h>
#include <string>
#include <vector>
#include "DataExchange.hh"

class MPIEnvironment : public ::testing::Environment {
public:
  void SetUp() override {
    int argc = 0;
    char **argv = nullptr;
    MPI_Init(&argc, &argv);
  }
  void TearDown() override { MPI_Finalize(); }
};

const ::testing::Environment *const mpi_env =
    ::testing::AddGlobalTestEnvironment(new MPIEnvironment);

// Each rank sends its own rank value as a single scalar.
// After allgather, every rank should hold [0, 1, 2, ..., size-1].
TEST(MPIHelper, allgather) {
  MPIHelper helper(MPI_COMM_WORLD);
  int rank = helper.rank();
  int size = helper.size();

  std::vector<int> recvbuf(size);
  helper.allgather(rank, recvbuf.data(), 1);

  // recvbuf[i] must equal i because rank i contributed value i
  for (int i = 0; i < size; ++i) {
    EXPECT_EQ(recvbuf[i], i);
  }
}

// Rank r sends (r+1) copies of the integer r.
// After allgatherv the concatenated buffer is:
//   [0 | 1 1 | 2 2 2 | 3 3 3 3 | ...]
// and displs is [0, 1, 3, 6, 10, ...] (prefix sums of sendcounts).
TEST(MPIHelper, allgatherv) {
  MPIHelper helper(MPI_COMM_WORLD);
  int rank = helper.rank();
  int size = helper.size();

  // rank r contributes (r+1) elements, each equal to r
  int sendcount = rank + 1;
  std::vector<int> sendbuf(sendcount, rank);

  std::vector<int> recvbuf;
  std::vector<int> displs;
  helper.allgatherv(sendbuf.data(), sendcount, recvbuf, displs);

  // displs has (size+1) entries: displs[r] is the start index of rank r's data,
  // displs[size] is the total element count
  ASSERT_EQ((int)displs.size(), size + 1);
  EXPECT_EQ(displs[0], 0);

  int offset = 0;
  for (int r = 0; r < size; ++r) {
    // displs[r] must equal the sum of sendcounts for ranks 0..(r-1)
    EXPECT_EQ(displs[r], offset);
    // The (r+1) elements contributed by rank r must all equal r
    for (int j = 0; j < r + 1; ++j) {
      EXPECT_EQ(recvbuf[offset + j], r);
    }
    offset += r + 1;
  }
  // displs[size] must equal the total number of received elements
  EXPECT_EQ(displs[size], offset);
}

// Rank r sends (r+1) strings. String j from rank r has length (r*3 + j + 1)
// and is filled with the character ('a' + r).
//
// Example with 3 ranks:
//   rank 0: {"a"}                              (1 string, lengths: 1)
//   rank 1: {"bb", "bbb", "bbbb"}              (2 strings... wait, sendcount = rank+1 = 2)
//            actually: {"bb", "bbb"}           (2 strings, lengths: 1*3+0+1=4? No:)
//            len = r*3 + j + 1:
//              r=1,j=0 -> len=4: "bbbb"   (wait, that's rank 1)
//   Let's be precise:
//     rank 0, j=0: len=0*3+0+1=1,  "a"
//     rank 1, j=0: len=1*3+0+1=4,  "bbbb"
//     rank 1, j=1: len=1*3+1+1=5,  "bbbbb"
//     rank 2, j=0: len=2*3+0+1=7,  "ccccccc"
//     rank 2, j=1: len=2*3+1+1=8,  "cccccccc"
//     rank 2, j=2: len=2*3+2+1=9,  "ccccccccc"
//
// After allgatherv the output contains all strings from all ranks in rank order,
// and displs[r] is the index of the first string contributed by rank r.
TEST(MPIHelper, allgatherv_string) {
  MPIHelper helper(MPI_COMM_WORLD);
  int rank = helper.rank();
  int size = helper.size();

  int sendcount = rank + 1;
  std::vector<std::string> sendbuf(sendcount);
  for (int j = 0; j < sendcount; ++j) {
    int len = rank * 3 + j + 1;
    sendbuf[j] = std::string(len, 'a' + rank);
  }

  std::vector<std::string> recvbuf;
  std::vector<int> displs;
  helper.allgatherv(sendbuf.data(), sendcount, recvbuf, displs);

  // displs[r] is the index into recvbuf of rank r's first string
  ASSERT_EQ((int)displs.size(), size + 1);
  EXPECT_EQ(displs[0], 0);

  int offset = 0;
  for (int r = 0; r < size; ++r) {
    EXPECT_EQ(displs[r], offset);
    // Verify each string contributed by rank r
    for (int j = 0; j < r + 1; ++j) {
      int len = r * 3 + j + 1;
      EXPECT_EQ(recvbuf[offset + j], std::string(len, 'a' + r));
    }
    offset += r + 1;
  }
  // displs[size] is the total string count across all ranks
  EXPECT_EQ(displs[size], offset);
}

// Global mesh: regions "A", "B", "C", ... (1 + size regions total).
// Local mesh per rank r: regions "A" and the rank's unique region ('B'+r).
//
// Values contributed:
//   region "A"       <- rank r contributes (r+1),  so values are {1, 2, ..., size}
//   region 'B'+r     <- rank r contributes (r*10),  no other rank has this region
//
// With average=true each output region is the mean over contributing ranks:
//   "A"    = (1+2+...+size) / size = (size+1)/2.0
//   'B'+r  = (r*10) / 1           = r*10.0          (only rank r contributes)
TEST(DataExchange, fullExchange_average) {
  MPIHelper helper(MPI_COMM_WORLD);
  int rank = helper.rank();
  int size = helper.size();

  // Global mesh shared by all ranks as the output target
  Mesh globalMesh;
  globalMesh.addRegion("A");
  for (int r = 0; r < size; ++r) {
    std::string name(1, 'B' + r);
    globalMesh.addRegion(name);
  }
  Data globalData(globalMesh);
  RealRegionScalar outData(globalData);

  // Local mesh: "A" (index 0) and this rank's unique region (index 1)
  Mesh localMesh;
  localMesh.addRegion("A");
  std::string myRegion(1, 'B' + rank);
  localMesh.addRegion(myRegion);
  Data localData(localMesh);
  RealRegionScalar inData(localData);
  inData[0] = rank + 1;     // rank r contributes (r+1) to "A"
  inData[1] = rank * 10.0;  // rank r contributes (r*10) to its unique region

  fullExchange(inData, outData, helper, /*average=*/true);

  // "A": mean of {1, 2, ..., size} = (size+1)/2.0
  double expectedA = (size + 1) / 2.0;
  EXPECT_DOUBLE_EQ(outData[0], expectedA);

  // 'B'+r: only rank r contributes, so the average is just that rank's value
  for (int r = 0; r < size; ++r) {
    EXPECT_DOUBLE_EQ(outData[1 + r], r * 10.0);
  }
}

// Same setup as fullExchange_average but with average=false, so regions are summed.
//
// With average=false:
//   "A"    = 1+2+...+size = size*(size+1)/2
//   'B'+r  = r*10.0       (single contributor, sum equals the value)
TEST(DataExchange, fullExchange_sum) {
  MPIHelper helper(MPI_COMM_WORLD);
  int rank = helper.rank();
  int size = helper.size();

  Mesh globalMesh;
  globalMesh.addRegion("A");
  for (int r = 0; r < size; ++r) {
    std::string name(1, 'B' + r);
    globalMesh.addRegion(name);
  }
  Data globalData(globalMesh);
  RealRegionScalar outData(globalData);

  Mesh localMesh;
  localMesh.addRegion("A");
  std::string myRegion(1, 'B' + rank);
  localMesh.addRegion(myRegion);
  Data localData(localMesh);
  RealRegionScalar inData(localData);
  inData[0] = rank + 1;
  inData[1] = rank * 10.0;

  fullExchange(inData, outData, helper, /*average=*/false);

  // "A": sum of {1, 2, ..., size} = size*(size+1)/2
  double expectedA = size * (size + 1) / 2.0;
  EXPECT_DOUBLE_EQ(outData[0], expectedA);

  // 'B'+r: single contributor, so sum equals that rank's value
  for (int r = 0; r < size; ++r) {
    EXPECT_DOUBLE_EQ(outData[1 + r], r * 10.0);
  }
}
