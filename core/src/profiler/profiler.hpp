#pragma once

#include <chrono>
#include <fstream>

using time_point = std::chrono::steady_clock::time_point;

struct measure_t {
  std::string id;
  double duration;

  friend std::ostream& operator<<(std::ostream& os, const measure_t& measure) {
    os << measure.id << ": " << measure.duration << std::endl;
    return os;
  }
};

class Profiler{
  public:
    Profiler();
    ~Profiler();
  static Profiler& getProfiler();
  void addMeasure(const std::string& id, const time_point& start, const time_point& stop);

  private:
  std::ofstream outputFile;
  std::string fileName;
  bool initialized = false;
};

class ScopeProfiler{
  public:
  ScopeProfiler(const std::string& _id);
  ~ScopeProfiler();
  private:
    Profiler* profiler;
    time_point start;
    std::string id;
};
