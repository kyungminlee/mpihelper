#pragma once
#include <string>
#include <vector>

class Mesh {
public:
  std::string getRegionName(int i) const {
    return _regions[i];
  }

  void addRegion(std::string const & name) {
    _regions.push_back(name);
  }

  std::size_t getRegionSize() const {
    return _regions.size();
  }

private:
  std::vector<std::string> _regions;
};

class Data {
public:
  Data(Mesh const & mesh): _mesh(&mesh) {}
  Mesh const & getMesh() const { return *_mesh; }
  Mesh const * _mesh = nullptr;
};

class RealRegionScalar {
public:
  RealRegionScalar(Data & data)
  : _data(&data) {
    _values.resize(_data->getMesh().getRegionSize(), 0.0);
  }

  Mesh const & getMesh() const { return _data->getMesh(); }

  double operator[](std::size_t i) const {
    return _values[i];
  }

  double & operator[](std::size_t i) {
    return _values[i];
  }

  double * data() {
    return _values.data();
  }
  double const * data() const {
    return _values.data();
  }

private:
  Data * _data = nullptr;
  std::vector<double> _values;
};