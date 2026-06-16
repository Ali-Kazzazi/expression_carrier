#include <iostream>
#include <iomanip>
#include <vector>
#include <ctime>
#include "exier_powDiv.hpp"
#include "mlp.hpp"

void print(std::shared_ptr<EXIER> input)
{
	input->repr();
	std::cout << std::endl;
	std::cout << input->toJSON() << std::endl;
}

int main()
{
	srand(static_cast<unsigned>(time(nullptr)));
	std::cout << std::fixed << std::setprecision(2);

	auto model = MLP(3, {4, 4, 1}, {act_relu(), act_relu(), act_sigmoid()});

	auto model2 = MLP(3, {4, 4, 1});

	std::vector<std::shared_ptr<EXIER>> x = {
		std::make_shared<EXIER>(2.0f),
		std::make_shared<EXIER>(3.0f),
		std::make_shared<EXIER>(-1.0f)};
	auto out = model(x);
	print(out[0]);

	return 0;
}