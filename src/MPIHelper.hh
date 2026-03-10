#include <mpi.h>
#include <stdexcept>
#include <string>
#include <vector>

class MPIError : public std::runtime_error {
public:
  MPIError(char const * message)
  : std::runtime_error(message)
  {}
};

template <typename T>
class MPIDataType;

#define DEF_DT(T, DT) \
template <> \
class MPIDataType<T> { \
public: \
  static MPI_Datatype type() { return DT; } \
};

DEF_DT(bool, MPI_CXX_BOOL);
DEF_DT(char, MPI_CHAR);
DEF_DT(unsigned char, MPI_UNSIGNED_CHAR);
DEF_DT(short, MPI_SHORT);
DEF_DT(unsigned short, MPI_UNSIGNED_SHORT);
DEF_DT(int, MPI_INT);
DEF_DT(unsigned int, MPI_UNSIGNED);
DEF_DT(long, MPI_LONG);
DEF_DT(unsigned long, MPI_UNSIGNED_LONG);
DEF_DT(long long, MPI_LONG_LONG);
DEF_DT(unsigned long long, MPI_UNSIGNED_LONG_LONG);

class MPIHelper {
public:
  MPIHelper(MPI_Comm comm)
  : _comm(comm)
  {
    if (MPI_Comm_size(_comm, &_size) != MPI_SUCCESS) {
      throw MPIError("Failed to get size");
    }
    if (MPI_Comm_rank(_comm, &_rank) != MPI_SUCCESS) {
      throw MPIError("Failed to get rank");
    }
  }

  template <typename T1, typename T2>
  void allgather(
    T1 sendbuf,
    T2 * recvbuf,
    int count
  ) const {
    static_assert(sizeof(T1) == sizeof(T2));
    if (MPI_Allgather(&sendbuf, 1, MPIDataType<T1>::type(), recvbuf, count, MPIDataType<T2>::type(), _comm) != MPI_SUCCESS) {
      throw MPIError("Failed to allgather");
    }
  }

  template <typename T1, typename T2, typename A2, typename A3>
  void allgatherv(
    T1 const * sendbuf, int sendcount,
    std::vector<T2, A2> & recvbuf,
    std::vector<int, A3> & displs
  ) {
    static_assert(sizeof(T1) == sizeof(T2));
    std::vector<int> recvcounts(_size);
    allgather(sendcount, recvcounts.data(), 1);
    displs.resize(_size + 1);
    displs[0] = 0;
    for (int i = 0; i < _size; ++i) {
      displs[i + 1] = displs[i] + recvcounts[i];
    }
    recvbuf.resize(displs[_size]);
    if (MPI_Allgatherv(
          sendbuf, sendcount, MPIDataType<T1>::type(),
          recvbuf.data(), recvcounts.data(), displs.data(), MPIDataType<T2>::type(),
          _comm) != MPI_SUCCESS) {
      throw MPIError("Failed to allgatherv");
    }
  }

  template <typename A2, typename A3>
  void allgatherv(
    std::string const * sendbuf, int sendcount,
    std::vector<std::string, A2> & recvbuf,
    std::vector<int, A3> & displs
  ) {
    // Step 1: gather string counts per rank -> build string-level displs
    std::vector<int> recvcounts_str(_size);
    allgather(sendcount, recvcounts_str.data(), 1);
    displs.resize(_size + 1);
    displs[0] = 0;
    for (int i = 0; i < _size; ++i) {
      displs[i + 1] = displs[i] + recvcounts_str[i];
    }
    int total_strings = displs[_size];

    // Step 2: allgatherv individual string char lengths
    std::vector<int> local_strsizes(sendcount);
    for (int i = 0; i < sendcount; ++i) {
      local_strsizes[i] = (int)sendbuf[i].size();
    }
    std::vector<int> all_strsizes;
    std::vector<int> strsizes_displs;
    allgatherv(local_strsizes.data(), sendcount, all_strsizes, strsizes_displs);

    // Step 3: flatten local strings into a single char buffer, allgatherv chars
    std::string local_flat;
    for (int i = 0; i < sendcount; ++i) {
      local_flat += sendbuf[i];
    }
    std::vector<char> flat_chars;
    std::vector<int> char_displs;
    allgatherv(local_flat.data(), (int)local_flat.size(), flat_chars, char_displs);

    // Step 4: reconstruct strings from flat char buffer using gathered sizes
    recvbuf.resize(total_strings);
    int char_offset = 0;
    for (int i = 0; i < total_strings; ++i) {
      recvbuf[i].assign(flat_chars.data() + char_offset, all_strsizes[i]);
      char_offset += all_strsizes[i];
    }
  }

  int size() const { return _size; }
  int rank() const { return _rank; }

private:
  MPI_Comm _comm = MPI_COMM_NULL;
  int _size = -1;
  int _rank = -1;
};