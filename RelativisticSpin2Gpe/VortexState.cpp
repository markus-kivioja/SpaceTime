#include "VortexState.hpp"
#include <fstream>
#include <iostream>

Spin1GroundState::Spin1GroundState()
{
}

bool Spin1GroundState::load(const std::string &path)
{
	std::ifstream fs(path.c_str(), std::ios::binary | std::ios::in);
	if(fs.fail() != 0) return false;

	fs.close();
	return true;
}

bool Spin1GroundState::save(const std::string &path) const
{
	std::ofstream fs(path.c_str(), std::ios::binary | std::ios_base::trunc);
	if(fs.fail() != 0) return false;


	fs.close();
	return true;
}


