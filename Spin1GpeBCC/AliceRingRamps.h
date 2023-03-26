#pragma once

#include <array>
#include <iostream>

#define Z_QUANTIZED 0
#define Y_QUANTIZED 1
#define X_QUANTIZED 2

#define BASIS Z_QUANTIZED

#include "utils.h"

struct Signal
{
	double Bq = 0;
	double3 Bb = { 0, 0, 0 };
};

enum class RampType
{
	CONSTANT = 0,
	LINEAR,
	FAST_EXTRACTION
};

#define ENABLE_DECAY_BIAS 1

//// Quadrupole ////
std::array<double, 4> Bqs = { 4.3, 4.3, 0.0, 0.0 };
std::array<double, 4> BqDurations = { 10.0, 192.03, 0.2, 2000.0 };
std::array<RampType, 4> BqTypes = { RampType::LINEAR, RampType::CONSTANT, RampType::LINEAR, RampType::CONSTANT };

//// Bias ////
#if ENABLE_DECAY_BIAS
// Implement also the other basises, this is now only for z-quantized
std::array<double3, 6> Bbs = { make_double3(0, 0, 1.0), make_double3(0, 0, 0.045), make_double3(0, 0, 0.045), make_double3(0, 0, 0), make_double3(0, 0, 1.2), make_double3(0, 0, 1.2) };
std::array<double, 6> BbDurations = { 10, 10, 2.02, 180, 0.5, 2000.0 };
std::array<RampType, 6> BbTypes = { RampType::CONSTANT, RampType::LINEAR, RampType::CONSTANT, RampType::LINEAR, RampType::FAST_EXTRACTION, RampType::CONSTANT };
#else
std::array<double, 6> Bbs = { 1.0, 0.045, 0.045, 0, 0 };
std::array<double, 6> BbDurations = { 10, 10, 2.02, 180, 2000.0 };
std::array<RampType, 6> BbTypes = { RampType::CONSTANT, RampType::LINEAR, RampType::CONSTANT, RampType::LINEAR, RampType::CONSTANT };
#endif

////////// Ideal projection ramp ////////////
// std::array<double, 3> Bqs = { 4.3, 4.3, 0.0 };
// std::array<double, 3> BqDurations = { 10.0, 192.03, 2000.0 };
// std::array<RampType, 3> BqTypes = { RampType::LINEAR, RampType::CONSTANT, RampType::CONSTANT };

// #if BASIS == Z_QUANTIZED
// std::array<double3, 5> Bbs = { make_double3(0, 0, 1.0), make_double3(0, 0, 0.045), make_double3(0, 0, 0.045), make_double3(0, 0, 0), make_double3(0, 0, 1.2) };
// #elif BASIS == Y_QUANTIZED
// std::array<double3, 5> Bbs = { make_double3(0, 0, 1.0), make_double3(0, 0, 0.045), make_double3(0, 0, 0.045), make_double3(0, 0, 0), make_double3(0, 1.2, 0) };
// #elif BASIS == X_QUANTIZED
// std::array<double3, 5> Bbs = { make_double3(0, 0, 1.0), make_double3(0, 0, 0.045), make_double3(0, 0, 0.045), make_double3(0, 0, 0), make_double3(1.2, 0, 0) };
// #endif
// std::array<double, 5> BbDurations = { 10, 10, 2.02, 180, 2000.0 };
// std::array<RampType, 5> BbTypes = { RampType::CONSTANT, RampType::LINEAR, RampType::CONSTANT, RampType::LINEAR, RampType::CONSTANT };
///////////////////////////////////////////////

void printBasis()
{
	std::cout << "Using Alice ring creation process" << std::endl;
#if BASIS == Z_QUANTIZED
	std::cout << "Using z-quantized basis!" << std::endl;
#elif BASIS == Y_QUANTIZED
	std::cout << "Using y-quantized basis!" << std::endl;
#elif BASIS == X_QUANTIZED
	std::cout << "Using x-quantized basis!" << std::endl;
#endif
}

Signal getSignal(double t)
{
	Signal signal;

	double tOrig = t;

	/// Bq
	uint32_t BqRampIdx = 0;
	for (; BqRampIdx < Bqs.size(); ++BqRampIdx)
	{
		double tInRamp = t - BqDurations[BqRampIdx];
		if (tInRamp < 0)
		{
			break;
		}
		t = tInRamp;
	}
	double prevBq = (BqRampIdx > 0) ? Bqs[BqRampIdx - 1] : 0.0;
	switch (BqTypes[BqRampIdx])
	{
	case RampType::CONSTANT:
		signal.Bq = Bqs[BqRampIdx];
		break;
	case RampType::LINEAR:
		signal.Bq = prevBq + t * (Bqs[BqRampIdx] - prevBq) / BqDurations[BqRampIdx];
		break;
	case RampType::FAST_EXTRACTION:
		signal.Bq = prevBq + (Bqs[BqRampIdx] - prevBq) * sqrt(t / BqDurations[BqRampIdx]);
		break;
	default:
		std::cout << "Invalid magnetic ramp type: " << static_cast<int>(BqTypes[BqRampIdx]) << std::endl;
		exit(1);
		break;
	}

	t = tOrig;

	// Bz
	uint32_t BbRampIdx = 0;
	for (; BbRampIdx < Bbs.size(); ++BbRampIdx)
	{
		double tInRamp = t - BbDurations[BbRampIdx];
		if (tInRamp < 0)
		{
			break;
		}
		t = tInRamp;
	}
	double3 prevBb = (BbRampIdx > 0) ? Bbs[BbRampIdx - 1] : make_double3(0, 0, 0);
	switch (BbTypes[BbRampIdx])
	{
	case RampType::CONSTANT:
		signal.Bb = Bbs[BbRampIdx];
		break;
	case RampType::LINEAR:
		signal.Bb = prevBb + t * (Bbs[BbRampIdx] - prevBb) / BbDurations[BbRampIdx];
		break;
	case RampType::FAST_EXTRACTION:
		signal.Bb = prevBb + (Bbs[BbRampIdx] - prevBb) * sqrt(t / BbDurations[BbRampIdx]);
		break;
	default:
		std::cout << "Invalid magnetic ramp type: " << static_cast<int>(BbTypes[BbRampIdx]) << std::endl;
		exit(1);
		break;
	}

	return signal;
}