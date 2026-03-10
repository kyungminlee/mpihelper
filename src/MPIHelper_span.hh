#pragma once

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

  // sendbuf is a single scalar; recvbuf.size() must equal size() * recvcount_per_rank
  template <typename T1, typename T2>
  void allgather(T1 sendbuf, std::span<T2> recvbuf) const {
    static_assert(sizeof(T1) == sizeof(T2));
    int recvcount = (int)recvbuf.size() / size();
    if (MPI_Allgather(&sendbuf, 1, MPIDataType<T1>::type(), recvbuf.data(), recvcount, MPIDataType<T2>::type(), _comm) != MPI_SUCCESS) {
      throw MPIError("Failed to allgather");
    }
  }

  template <typename T1, typename T2, typename A2, typename A3>
  void allgatherv(
    std::span<T1 const> sendbuf,
    std::vector<T2, A2> & recvbuf,
    std::vector<int, A3> & displs
  ) {
    static_assert(sizeof(T1) == sizeof(T2));
    int _size = size();
    std::vector<int> recvcounts(_size);
    allgather((int)sendbuf.size(), std::span(recvcounts));
    displs.resize(_size + 1);
    displs[0] = 0;
    for (int i = 0; i < _size; ++i) {
      displs[i + 1] = displs[i] + recvcounts[i];
    }
    recvbuf.resize(displs[_size]);
    if (MPI_Allgatherv(
          sendbuf.data(), (int)sendbuf.size(), MPIDataType<T1>::type(),
          recvbuf.data(), recvcounts.data(), displs.data(), MPIDataType<T2>::type(),
          _comm) != MPI_SUCCESS) {
      throw MPIError("Failed to allgatherv");
    }
  }

  template <typename A2, typename A3>
  void allgatherv(
    std::span<std::string const> sendbuf,
    std::vector<std::string, A2> & recvbuf,
    std::vector<int, A3> & displs
  ) {
    int _size = size();
    // Step 1: gather string counts per rank -> build string-level displs
    std::vector<int> recvcounts_str(_size);
    allgather((int)sendbuf.size(), std::span(recvcounts_str));
    displs.resize(_size + 1);
    displs[0] = 0;
    for (int i = 0; i < _size; ++i) {
      displs[i + 1] = displs[i] + recvcounts_str[i];
    }
    int total_strings = displs[_size];

    // Step 2: allgatherv individual string char lengths
    std::vector<int> local_strsizes(sendbuf.size());
    for (int i = 0; i < (int)sendbuf.size(); ++i) {
      local_strsizes[i] = (int)sendbuf[i].size();
    }
    std::vector<int> all_strsizes;
    std::vector<int> strsizes_displs;
    allgatherv(std::span<int const>(local_strsizes), all_strsizes, strsizes_displs);

    // Step 3: flatten local strings into a single char buffer, allgatherv chars
    std::string local_flat;
    for (auto const & s : sendbuf) {
      local_flat += s;
    }
    std::vector<char> flat_chars;
    std::vector<int> char_displs;
    allgatherv(std::span<char const>(local_flat.data(), local_flat.size()), flat_chars, char_displs);

    // Step 4: reconstruct strings from flat char buffer using gathered sizes
    recvbuf.resize(total_strings);
    int char_offset = 0;
    for (int i = 0; i < total_strings; ++i) {
      recvbuf[i].assign(flat_chars.data() + char_offset, all_strsizes[i]);
      char_offset += all_strsizes[i];
    }
  }

  template <typename T>
  void allreduce(std::span<T const> sendbuf, std::span<T> recvbuf, MPI_Op op) const {
    if (MPI_Allreduce(sendbuf.data(), recvbuf.data(), (int)sendbuf.size(), MPIDataType<T>::type(), op, _comm) != MPI_SUCCESS) {
      throw MPIError("Failed to allreduce");
    }
  }

  // sendbuf.size() must equal size() * recvbuf.size() (only significant at root)
  template <typename T>
  void scatter(std::span<T const> sendbuf, std::span<T> recvbuf, int root) const {
    int sendcount = (int)sendbuf.size() / size();
    if (MPI_Scatter(sendbuf.data(), sendcount, MPIDataType<T>::type(), recvbuf.data(), (int)recvbuf.size(), MPIDataType<T>::type(), root, _comm) != MPI_SUCCESS) {
      throw MPIError("Failed to scatter");
    }
  }

  template <typename T>
  void reduce(std::span<T const> sendbuf, std::span<T> recvbuf, MPI_Op op, int root) const {
    if (MPI_Reduce(sendbuf.data(), recvbuf.data(), (int)sendbuf.size(), MPIDataType<T>::type(), op, root, _comm) != MPI_SUCCESS) {
      throw MPIError("Failed to reduce");
    }
  }

  template <typename T>
  MPI_Request isend(std::span<T const> buf, int dest, int tag) const {
    MPI_Request request;
    if (MPI_Isend(buf.data(), (int)buf.size(), MPIDataType<T>::type(), dest, tag, _comm, &request) != MPI_SUCCESS) {
      throw MPIError("Failed to isend");
    }
    return request;
  }

  template <typename T>
  MPI_Request irecv(std::span<T> buf, int source, int tag) const {
    MPI_Request request;
    if (MPI_Irecv(buf.data(), (int)buf.size(), MPIDataType<T>::type(), source, tag, _comm, &request) != MPI_SUCCESS) {
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
    if (MPI_Comm_rank(_comm, &rank) != MPI_SUCCESS) {
      throw MPIError("Failed to get rank");
    }
    return rank;
  }

private:
  MPI_Comm _comm = MPI_COMM_NULL;
};
