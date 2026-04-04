#pragma once

#include <array>
#include <iostream>

#include "utils.h"

#define DISABLE 0

struct Signal
{
	myFloat Bq = 0;
	myFloat3 Bb = { 0, 0, 0 };
};

enum class RampType
{
	CONSTANT = 0,
	LINEAR,
	FAST_EXTRACTION
};

// Experimentally realistic ramps
//// Quadrupole ////
std::array<myFloat, 1> Bqs = { 4.3 };
std::array<myFloat, 1> BqDurations = { 100.0 };
std::array<RampType, 1> BqTypes = { RampType::CONSTANT };

//// Bias ////
std::array<myFloat3, 1> Bbs = { {0, 0, 0} };
std::array<myFloat, 1> BbDurations = { 100 };
std::array<RampType, 1> BbTypes = { RampType::CONSTANT };

// Start with the magnetic field zero being at the center of the condensate
//// Quadrupole ////
//std::array<myFloat, 1> Bqs = { 4.3 };
//std::array<myFloat, 1> BqDurations = { 100.0 };
//std::array<RampType, 1> BqTypes = { RampType::CONSTANT };
//
////// Bias ////
//// Implement also the other basises, this is now only for z-quantized
//std::array<myFloat3, 1> Bbs = { {0, 0, 0} };
//std::array<myFloat, 1> BbDurations = { 100 };
//std::array<RampType, 1> BbTypes = { RampType::CONSTANT };

Signal getSignal(myFloat t)
{
#if DISABLE
	return {};
#else
	Signal signal;

	myFloat tOrig = t;

	/// Bq
	uint32_t BqRampIdx = 0;
	for (; BqRampIdx < Bqs.size(); ++BqRampIdx)
	{
		myFloat tInRamp = t - BqDurations[BqRampIdx];
		if (tInRamp < 0)
		{
			break;
		}
		t = tInRamp;
	}
	myFloat prevBq = (BqRampIdx > 0) ? Bqs[BqRampIdx - 1] : 0.0;
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
		myFloat tInRamp = t - BbDurations[BbRampIdx];
		if (tInRamp < 0)
		{
			break;
		}
		t = tInRamp;
	}
	myFloat3 prevBb = (BbRampIdx > 0) ? Bbs[BbRampIdx - 1] : myFloat3{0, 0, 0};
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
#endif
}