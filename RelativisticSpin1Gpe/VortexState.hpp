#ifndef _VORTEXSTATE_HPP_INCLUDED_
#define _VORTEXSTATE_HPP_INCLUDED_

/*
	- To create, save, and load vortex states
	- Vortex state of formula \Psi(r, \phi, z, t) = f(r,z) e^{i \kappa \phi - i \mu t}
	- To satisfy Gross-Pitaevskii equation: i \partial_t \Psi = ( -0.5 \nabla^2 + V + g |\Psi|^2 ) \Psi, where V is given potential.
	- Normalized by \int |\Psi|^2 = 1.
	- Covers both two- and three-dimensional vortex solutions
*/

#include <Types/Buffer.hpp>
#include <Types/Complex.hpp>
#include <string>

class Spin1GroundState
{
public:
	Spin1GroundState();
	virtual ~Spin1GroundState() { }

	// load and save binary file
	bool load(const std::string &path);
	bool save(const std::string &path) const;
protected:
};

#endif //_VORTEXSTATE_HPP_INCLUDED_
