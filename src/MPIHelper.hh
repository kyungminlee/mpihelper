#include <mpi.h>
#include <stdexcept>
#include <string>
#include <vector>
#include <span>

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
    int _size = size();
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
    int _size = size();
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

  template <typename T>
  void allreduce(T const * sendbuf, T * recvbuf, int count, MPI_Op op) const {
    if (MPI_Allreduce(sendbuf, recvbuf, count, MPIDataType<T>::type(), op, _comm) != MPI_SUCCESS) {
      throw MPIError("Failed to allreduce");
    }
  }

  template <typename T>
  void scatter(T const * sendbuf, int sendcount, T * recvbuf, int recvcount, int root) const {
    if (MPI_Scatter(sendbuf, sendcount, MPIDataType<T>::type(), recvbuf, recvcount, MPIDataType<T>::type(), root, _comm) != MPI_SUCCESS) {
      throw MPIError("Failed to scatter");
    }
  }

  template <typename T>
  void reduce(T const * sendbuf, T * recvbuf, int count, MPI_Op op, int root) const {
    if (MPI_Reduce(sendbuf, recvbuf, count, MPIDataType<T>::type(), op, root, _comm) != MPI_SUCCESS) {
      throw MPIError("Failed to reduce");
    }
  }

  template <typename T>
  MPI_Request isend(T const * buf, int count, int dest, int tag) const {
    MPI_Request request;
    if (MPI_Isend(buf, count, MPIDataType<T>::type(), dest, tag, _comm, &request) != MPI_SUCCESS) {
      throw MPIError("Failed to isend");
    }
    return request;
  }

  template <typename T>
  MPI_Request irecv(T * buf, int count, int source, int tag) const {
    MPI_Request request;
    if (MPI_Irecv(buf, count, MPIDataType<T>::type(), source, tag, _comm, &request) != MPI_SUCCESS) {
      throw MPIError("Failed to irecv");
    }
    return request;
  }

  int size() const {
    int size = -1;
    if (MPI_Comm_size(_comm, &size) != MPI_SUCCESS) {
      throw MPIError("Failed to get size");
    }
    return size;
  }

  int rank() const {
    int rank = -1;
    if (MPI_Comm_size(_comm, &rank) != MPI_SUCCESS) {
      throw MPIError("Failed to get size");
    }
    return rank;
  }

private:
  MPI_Comm _comm = MPI_COMM_NULL;
};