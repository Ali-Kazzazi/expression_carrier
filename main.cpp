#include <iostream>
#include <iomanip>
#include "exier_powDiv.hpp"
#include "neuron.hpp"

int main()
{
	std::cout << std::fixed << std::setprecision(2);

	// auto x1 = std::make_shared<EXIER>(2.0f);
	// auto x2 = std::make_shared<EXIER>(0.0f);

	// auto w1 = std::make_shared<EXIER>(-3.0f);
	// auto w2 = std::make_shared<EXIER>(1.0f);

	// auto b = std::make_shared<EXIER>(6.8813735870195432f);

	// auto x1w1 = x1 * w1;
	// auto x2w2 = x2 * w2;

	// auto x1w1x2w2 = x1w1 + x2w2;

	// auto n = x1w1x2w2 + b;
	// auto o = n->tanh();

	// auto e = (2 * n)->exp();

	// auto o = (e - 1) / (e + 1); // tanh(n) = (e^(2n) - 1) / (e^(2n) + 1)

	// auto x1 = std::make_shared<EXIER>(2.0f);
	// auto o = 3 * x1 - 1.0f;

	// o->grad = 1.0f;
	// o->backward();

	// o->repr();
	// std::cout << std::endl;
	// std::cout << o->toJSON() << std::endl;

	return 0;
}