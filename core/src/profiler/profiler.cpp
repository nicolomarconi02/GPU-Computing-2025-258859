#include "profiler/profiler.hpp"
#include <chrono>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sys/time.h>

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

ScopeProfiler::ScopeProfiler(const std::string& _id) {
  profiler = &Profiler::getProfiler();
  start = std::chrono::steady_clock::now();
  id = _id;
}

ScopeProfiler::~ScopeProfiler() {
  profiler->addMeasure(id, start, std::chrono::steady_clock::now());
}

Profiler::Profiler() {
  if (!std::filesystem::exists(PROFILER_OUTPUT_PATH)) {
    std::filesystem::create_directory(PROFILER_OUTPUT_PATH);
  }
  fileName = "session-" + ::formatDate(::getTimestampMicroseconds()) + "-" +
             ::formatTime(::getTimestampMicroseconds());

  outputFile.open(PROFILER_OUTPUT_PATH + fileName, std::ofstream::out);
  if (!outputFile.is_open()) {
    initialized = false;
    return;
  }

  initialized = true;
}

Profiler::~Profiler() {
  if (outputFile.is_open()) {
    outputFile.close();
    initialized = false;
  }
}

Profiler& Profiler::getProfiler() {
  static Profiler profiler;
  return profiler;
}

void Profiler::addMeasure(const std::string& id, const time_point& start,
                          const time_point& stop) {
  if (!initialized) {
    return;
  }

  measure_t record;
  record.id = id;
  record.duration = ::getDeltaSecs(stop - start);

  outputFile << record;
  std::cout << record;
}
