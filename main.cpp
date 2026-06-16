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

	std::vector<std::vector<std::shared_ptr<EXIER>>> xs = {
		{std::make_shared<EXIER>(2.0f), std::make_shared<EXIER>(3.0f), std::make_shared<EXIER>(-1.0f)},
		{std::make_shared<EXIER>(3.0f), std::make_shared<EXIER>(-1.0f), std::make_shared<EXIER>(0.5f)},
		{std::make_shared<EXIER>(0.5f), std::make_shared<EXIER>(1.0f), std::make_shared<EXIER>(1.0f)},
		{std::make_shared<EXIER>(1.0f), std::make_shared<EXIER>(1.0f), std::make_shared<EXIER>(-1.0f)},
	};

	std::vector<std::shared_ptr<EXIER>> ys = {
		std::make_shared<EXIER>(1.0f),
		std::make_shared<EXIER>(-1.0f),
		std::make_shared<EXIER>(-1.0f),
		std::make_shared<EXIER>(1.0f),
	};

	for (int epoch = 1; epoch <= 100; epoch++)
	{
		// Zero gradients
		for (auto &p : model2.parameters())
		{
			p->grad = 0.0f;
		}

		auto loss = std::make_shared<EXIER>(0.0f);

		// Forward pass
		for (size_t i = 0; i < xs.size(); i++)
		{
			auto pred = model2(xs[i]);

			auto diff = ys[i] - pred[0];
			loss = loss + diff->pow(2);
		}

		// Backward pass
		loss->backward();

		// std::cout << "grad[0] = "
        //   << model2.parameters()[0]->grad
        //   << std::endl;

		// Gradient descent step
		for (auto &p : model2.parameters())
		{
			p->data -= 0.005f * p->grad;
		}

		std::cout << "epoch " << epoch
				  << " loss = " << loss->data
				  << std::endl;
	}

	return 0;
}