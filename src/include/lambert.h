// complex Lambert.cpp

#include <iostream>
#include <complex>

std::complex<double> zexpz(std::complex<double> z);

std::complex<double> zexpz_d(std::complex<double> z);

std::complex<double> zexpz_dd(std::complex<double> z);

std::complex<double> InitPoint(std::complex<double> z, int k);

std::complex<double> LambertW(std::complex<double> z, int k = 0);

