#pragma once

#include <array>
#include <iostream>

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
const std::vector<myFloat> Bqs = { 4.3, 0.0, 0.0 };
const std::vector<myFloat> BqDurations = { OPT_TRAP_OFF + GRADIENT_OFF_DELAY, GRADIENT_OFF_DUARATION, 100 };
const std::vector<RampType> BqTypes = { RampType::CONSTANT, RampType::LINEAR, RampType::CONSTANT };

//// Bias ////
const std::vector<myFloat3> Bbs = { {0, 0, 0.219}, {0, 0, 0}, {0, 0, 0}, {0, 0, 3.0} };
const std::vector<myFloat> BbDurations = { STATE_PREP_DURATION, CREATION_RAMP_DURATION, TOTAL_HOLD_TIME, 100 };
const std::vector<RampType> BbTypes = { RampType::CONSTANT, RampType::LINEAR, RampType::CONSTANT, RampType::FAST_EXTRACTION };

// Start with the magnetic field zero being at the center of the condensate
//// Quadrupole ////
//std::array<myFloat, 1> Bqs = { 4.3 };
//std::array<myFloat, 1> BqDurations = { 100.0 };
//std::array<RampType, 1> BqTypes = { RampType::CONSTANT };
//
////// Bias ////
//// Implement also the other basises, this is now only for z-quantized
//std::array<myFloat3, 1> Bbs = { make_myFloat3(0, 0, 0) };
//std::array<myFloat, 1> BbDurations = { 100 };
//std::array<RampType, 1> BbTypes = { RampType::CONSTANT };

void printRamp()
{
	std::cout << "Using knot/skyrmion creation ramp" << std::endl;
}

Signal getSignal(myFloat t)
{
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
		signal.Bq = prevBq + (Bqs[BqRampIdx] - prevBq) * (1.0 - exp(-t / PROJECTION_RAMP_DURATION));
		break;
	default:
		std::cout << "Invalid magnetic ramp type: " << static_cast<int>(BqTypes[BqRampIdx]) << std::endl;
		exit(1);
		break;
	}

	t = tOrig;

	// Bb
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
		signal.Bb = prevBb + (Bbs[BbRampIdx] - prevBb) * (1.0 - exp(-t / PROJECTION_RAMP_DURATION));
		break;
	default:
		std::cout << "Invalid magnetic ramp type: " << static_cast<int>(BbTypes[BbRampIdx]) << std::endl;
		exit(1);
		break;
	}

	return signal;
}