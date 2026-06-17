#include <iostream>
#include <iomanip>
#include <vector>
#include <ctime>
#include "exier_powDiv.hpp"
#include "mlp.hpp"
#include "optimizer.hpp"

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

	// auto model = MLP(3, {4, 4, 1});

	GD optimizer(model.parameters().size(), 0.005f);

	std::vector<std::vector<std::shared_ptr<EXIER>>> xs = {
		{std::make_shared<EXIER>(2.0f), std::make_shared<EXIER>(3.0f), std::make_shared<EXIER>(-1.0f)},
		{std::make_shared<EXIER>(3.0f), std::make_shared<EXIER>(-1.0f), std::make_shared<EXIER>(0.5f)},
		{std::make_shared<EXIER>(0.5f), std::make_shared<EXIER>(1.0f), std::make_shared<EXIER>(1.0f)},
		{std::make_shared<EXIER>(1.0f), std::make_shared<EXIER>(1.0f), std::make_shared<EXIER>(-1.0f)},
	};

	std::vector<std::shared_ptr<EXIER>> ys = {
		std::make_shared<EXIER>(1.0f),
		std::make_shared<EXIER>(0.0f),
		std::make_shared<EXIER>(0.0f),
		std::make_shared<EXIER>(1.0f),
	};

	for (int epoch = 1; epoch <= 100; epoch++)
	{
		
		for (auto &p : model.parameters())
		{
			p->grad = 0.0f;
		}

		auto loss = std::make_shared<EXIER>(0.0f);

		
		for (size_t i = 0; i < xs.size(); i++)
		{
			auto pred = model(xs[i]);

			auto diff = ys[i] - pred[0];
			loss = loss + diff->pow(2);
		}

		
		loss->backward();

		optimizer.step(model.parameters());


		std::cout << "epoch " << epoch
				  << " loss = " << loss->data
				  << std::endl;
	}

	return 0;
}