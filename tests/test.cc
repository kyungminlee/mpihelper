#include "MPIHelper.hh"
#include <gtest/gtest.h>
#include <string>
#include <vector>

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

TEST(MPIHelper, allgather) {
  MPIHelper helper(MPI_COMM_WORLD);
  int rank = helper.rank();
  int size = helper.size();

  std::vector<int> recvbuf(size);
  helper.allgather(rank, recvbuf.data(), 1);

  for (int i = 0; i < size; ++i) {
    EXPECT_EQ(recvbuf[i], i);
  }
}

TEST(MPIHelper, allgatherv) {
  MPIHelper helper(MPI_COMM_WORLD);
  int rank = helper.rank();
  int size = helper.size();

  // rank r sends (r+1) copies of value r
  int sendcount = rank + 1;
  std::vector<int> sendbuf(sendcount, rank);

  std::vector<int> recvbuf;
  std::vector<int> displs;
  helper.allgatherv(sendbuf.data(), sendcount, recvbuf, displs);

  ASSERT_EQ((int)displs.size(), size + 1);
  EXPECT_EQ(displs[0], 0);

  int offset = 0;
  for (int r = 0; r < size; ++r) {
    EXPECT_EQ(displs[r], offset);
    for (int j = 0; j < r + 1; ++j) {
      EXPECT_EQ(recvbuf[offset + j], r);
    }
    offset += r + 1;
  }
  EXPECT_EQ(displs[size], offset);
}

TEST(MPIHelper, allgatherv_string) {
  MPIHelper helper(MPI_COMM_WORLD);
  int rank = helper.rank();
  int size = helper.size();

  // rank r sends (r+1) strings; string j has (r*3 + j + 1) chars of 'a'+r
  // e.g. rank 0: {"a"}, rank 1: {"bb", "bbb", "bbbb"}, rank 2: {"ccc"x4, "ccc"x5, "ccc"x6, "ccc"x7}
  int sendcount = rank + 1;
  std::vector<std::string> sendbuf(sendcount);
  for (int j = 0; j < sendcount; ++j) {
    int len = rank * 3 + j + 1;
    sendbuf[j] = std::string(len, 'a' + rank);
  }

  std::vector<std::string> recvbuf;
  std::vector<int> displs;
  helper.allgatherv(sendbuf.data(), sendcount, recvbuf, displs);

  ASSERT_EQ((int)displs.size(), size + 1);
  EXPECT_EQ(displs[0], 0);

  int offset = 0;
  for (int r = 0; r < size; ++r) {
    EXPECT_EQ(displs[r], offset);
    for (int j = 0; j < r + 1; ++j) {
      int len = r * 3 + j + 1;
      EXPECT_EQ(recvbuf[offset + j], std::string(len, 'a' + r));
    }
    offset += r + 1;
  }
  EXPECT_EQ(displs[size], offset);
}
