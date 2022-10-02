#pragma once

#include <array>
#include <iostream>

struct Signal
{
	double Bq = 0;
	double Bz = 0;
};

enum class RampType
{
	CONSTANT = 0,
	LINEAR,
	FAST_EXTRACTION
};

#define ENABLE_DECAY_BIAS 1

std::array<double, 4> Bqs = { 4.3, 4.3, 0.0, 0.0 };
std::array<double, 4> BqDurations = { 10.0, 192.03, 0.2, 2000.0 };
//std::array<double, 4> BqDurations = { 10.0, 184.03, 0.2, 2000.0 };
std::array<RampType, 4> BqTypes = { RampType::LINEAR, RampType::CONSTANT, RampType::LINEAR, RampType::CONSTANT };

#if ENABLE_DECAY_BIAS
std::array<double, 6> Bzs = { 1.0, 0.045, 0.045, 0, 1.2, 1.2 };
//std::array<double, 6> Bzs = { 1.0, 0.045, 0.045, 0.002, 1.2, 1.2 };
std::array<double, 6> BzDurations = { 10, 10, 2.02, 180, 0.5, 2000.0 };
//std::array<double, 6> BzDurations = { 10, 10, 2.02, 172, 0.5, 2000.0 };
std::array<RampType, 6> BzTypes = { RampType::CONSTANT, RampType::LINEAR, RampType::CONSTANT, RampType::LINEAR, RampType::FAST_EXTRACTION, RampType::CONSTANT };
#else
std::array<double, 6> Bzs = { 1.0, 0.045, 0.045, 0, 0 };
std::array<double, 6> BzDurations = { 10, 10, 2.02, 180, 2000.0 };
std::array<RampType, 6> BzTypes = { RampType::CONSTANT, RampType::LINEAR, RampType::CONSTANT, RampType::LINEAR, RampType::CONSTANT };
#endif

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
	uint32_t BzRampIdx = 0;
	for (; BzRampIdx < Bzs.size(); ++BzRampIdx)
	{
		double tInRamp = t - BzDurations[BzRampIdx];
		if (tInRamp < 0)
		{
			break;
		}
		t = tInRamp;
	}
	double prevBz = (BzRampIdx > 0) ? Bzs[BzRampIdx - 1] : 0.0;
	switch (BzTypes[BzRampIdx])
	{
	case RampType::CONSTANT:
		signal.Bz = Bzs[BzRampIdx];
		break;
	case RampType::LINEAR:
		signal.Bz = prevBz + t * (Bzs[BzRampIdx] - prevBz) / BzDurations[BzRampIdx];
		break;
	case RampType::FAST_EXTRACTION:
		signal.Bz = prevBz + (Bzs[BzRampIdx] - prevBz) * sqrt(t / BzDurations[BzRampIdx]);
		break;
	default:
		std::cout << "Invalid magnetic ramp type: " << static_cast<int>(BzTypes[BzRampIdx]) << std::endl;
		exit(1);
		break;
	}

	return signal;
}