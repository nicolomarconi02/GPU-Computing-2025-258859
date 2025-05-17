#include "profiler/profiler.hpp"
#include <chrono>
#include <cstdint>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>
#include <sys/time.h>
#include <sys/types.h>

#include "defines.hpp"

inline double getDeltaSecs(const auto& delta_t) {
  return std::chrono::duration_cast<std::chrono::microseconds>(delta_t)
             .count() /
         1e6;
}

inline std::string formatDate(uint64_t timestamp) {
  char buff[70];
  time_t seconds = timestamp * 1e-6;
  struct tm* timeinfo;
  timeinfo = localtime(&seconds);
  strftime(buff, 70, "%Y_%m_%d", timeinfo);
  return std::string(buff);
}

inline std::string formatTime(uint64_t timestamp) {
  char buff[70];
  time_t seconds = timestamp * 1e-6;
  struct tm* timeinfo;
  timeinfo = localtime(&seconds);
  strftime(buff, 70, "%H_%M_%S", timeinfo);
  return std::string(buff);
}

inline uint64_t getTimestampMicroseconds() {
  struct timeval tv;
  gettimeofday(&tv, nullptr);
  return tv.tv_sec * 1000000 + tv.tv_usec;
}

ScopeProfiler::ScopeProfiler(const std::string& _id, uint64_t _FLOPS,
                             uint64_t _BYTES)
    : id(_id), FLOPS(_FLOPS), BYTES(_BYTES) {
  profiler = &Profiler::getProfiler();
  start = std::chrono::steady_clock::now();
}

ScopeProfiler::ScopeProfiler(const std::string& _id)
    : ScopeProfiler(_id, 0, 0) {}

ScopeProfiler::~ScopeProfiler() {
  profiler->addMeasure(id, start, std::chrono::steady_clock::now(), FLOPS,
                       BYTES);
}

Profiler::Profiler() {
  if (!std::filesystem::exists(PROFILER_OUTPUT_PATH)) {
    std::filesystem::create_directory(PROFILER_OUTPUT_PATH);
  }
  fileName = "session-" + formatDate(getTimestampMicroseconds()) + "-" +
             formatTime(getTimestampMicroseconds());

  outputFile.open(PROFILER_OUTPUT_PATH + fileName, std::ofstream::out);
  if (!outputFile.is_open()) {
    initialized = false;
    return;
  }

  initialized = true;
}

Profiler::~Profiler() {
  if (outputFile.is_open()) {
    computeCalculations();
    outputFile.close();
    initialized = false;
  }
}

Profiler& Profiler::getProfiler() {
  static Profiler profiler;
  return profiler;
}

void Profiler::addMeasure(const std::string& id, const time_point& start,
                          const time_point& stop, uint64_t FLOPS,
                          uint64_t BYTES) {
  if (!initialized) {
    return;
  }

  measure_t record;
  record.id = id;
  record.duration = getDeltaSecs(stop - start);
  record.FLOPS = FLOPS;
  record.BYTES = BYTES;

  auto it = sessions.find(id);
  if (it == sessions.end()) {
    std::vector<measure_t> vec;
    record.id = record.id + "_0";
    vec.push_back(record);
    sessions.emplace(id, vec);
  } else {
    record.id = record.id + "_" + std::to_string(it->second.size() - 1);
    it->second.push_back(record);
  }

  outputFile << record;
  std::cout << record;
}

void Profiler::computeCalculations() {
  for (const auto& [key, vec] : sessions) {
    uint64_t totalFLOPS = 0;
    double totalDuration = 0.0;
    uint64_t totalBYTES = 0;
    outputFile << "________________________________" << std::endl
               << key << std::endl;
    for (const auto& measure : vec) {
      totalFLOPS += measure.FLOPS;
      totalDuration += measure.duration;
      totalBYTES += measure.BYTES;

      outputFile << measure.id
                 << " GFLOPS/s: " << measure.FLOPS / (measure.duration * 1e9)
                 << " bandwidth: " << measure.BYTES / (measure.duration * 1e9)
                 << " GB/s" << std::endl;
    }

    double estimatedFLOPS = totalFLOPS / (totalDuration * 1e9);
    double estimatedBandwidth = totalBYTES / (totalDuration * 1e9);
    outputFile << "Estimated GFLOPS/s: " << estimatedFLOPS << " GFLOPS/s"
               << std::endl
               << "Peak GFLOPS/s: " << GPU_FP64_THROUGHPUT * 1e3 << " GFLOPS/s"
               << std::endl
               << "Efficiency FLOPS: "
               << (estimatedFLOPS / (GPU_FP64_THROUGHPUT * 1e3)) * 100.0 << " %"
               << std::endl
               << "Estimated bandwidth: " << estimatedBandwidth << " GB/s"
               << std::endl
               << "Peak bandwidth: " << GPU_MEMORY_BANDWIDTH << " GB/s"
               << std::endl
               << "Efficiency bandwidth: "
               << (estimatedBandwidth / GPU_MEMORY_BANDWIDTH) * 100.0 << " %"
               << std::endl;
  }
}
