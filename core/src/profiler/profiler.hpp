#pragma once

#include <sys/types.h>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <map>
#include <vector>

using time_point = std::chrono::steady_clock::time_point;

// time utilities
inline double getDeltaSecs(const auto& delta_t);
inline std::string formatDate(uint64_t timestamp);
inline std::string formatTime(uint64_t timestamp);
inline uint64_t getTimestampMicroseconds();

// measure structure used for the profiler
struct measure_t {
  std::string id;
  double duration;
  uint64_t FLOPS;
  uint64_t BYTES;

  friend std::ostream& operator<<(std::ostream& os, const measure_t& measure) {
    os << measure.id << ": " << measure.duration << std::endl;
    return os;
  }
};

// unique profiler class
class Profiler{
  public:
    Profiler();
    ~Profiler();
  static Profiler& getProfiler();
  void addMeasure(const std::string& id, const time_point& start, const time_point& stop, uint64_t FLOPS, uint64_t BYTES);

  private:
  void computeCalculations();

  private:
  std::ofstream outputFile;
  std::string fileName;
  bool initialized = false;
  std::map<std::string, std::vector<measure_t>> sessions;
};

// scope profiler class, when destructed automatically adds the measure to the profiler
class ScopeProfiler{
  public:
  ScopeProfiler(const std::string& _id);
  ScopeProfiler(const std::string& _id, uint64_t FLOPS, uint64_t BYTES);
  ~ScopeProfiler();
  private:
    Profiler* profiler;
    time_point start;
    std::string id;
    uint64_t FLOPS;
    uint64_t BYTES;
};
