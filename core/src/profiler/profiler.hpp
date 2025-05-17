#pragma once

#include <chrono>
#include <cstdint>
#include <fstream>
#include <unordered_map>
#include <vector>

using time_point = std::chrono::steady_clock::time_point;

inline double getDeltaSecs(const auto& delta_t);
inline std::string formatDate(uint64_t timestamp);
inline std::string formatTime(uint64_t timestamp);
inline uint64_t getTimestampMicroseconds();

struct measure_t {
  std::string id;
  double duration;
  uint64_t FLOPS;

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
  void addMeasure(const std::string& id, const time_point& start, const time_point& stop, uint64_t FLOPS);

  private:
  void computeCalculations();

  private:
  std::ofstream outputFile;
  std::string fileName;
  bool initialized = false;
  std::unordered_map<std::string, std::vector<measure_t>> sessions;
};

class ScopeProfiler{
  public:
  ScopeProfiler(const std::string& _id);
  ScopeProfiler(const std::string& _id, uint64_t FLOPS);
  ~ScopeProfiler();
  private:
    Profiler* profiler;
    time_point start;
    std::string id;
    uint64_t FLOPS;
};
