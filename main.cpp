#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <ctime>
#include "include/exier.hpp"
#include "include/mlp.hpp"
#include "include/optimizer.hpp"
#include "include/loss.hpp"
#include "include/mnist.hpp"

void print(std::shared_ptr<EXIER> input)
{
	input->repr();
	std::cout << std::endl;
	std::cout << input->toJSON() << std::endl;
}

int main()
{
	srand(static_cast<unsigned>(time(nullptr)));

	int num_train = 200;
	int batch_size = 10;

	read_mnist train("mnist_train.csv");

	auto model = MLP(784, {128, 64, 10}, {act_relu(), act_relu(), act_linear()});

	auto params = model.parameters();
	Adam optimizer(params.size(), 0.001f);

	for (int epoch = 1; epoch <= 200; epoch++)
	{
		train.reset();

		float epoch_loss = 0.0f;
		int correct = 0;
		int sample_count = 0;
		int num_batches = 0;

		read_mnist::exier_vector E_sample;

		while (sample_count < num_train)
		{
			for (auto &p : params)
				p->grad = 0.0f;

			auto batch_loss = std::make_shared<EXIER>(0.0f);
			int batch_count = 0;

			while (batch_count < batch_size && sample_count < num_train && train.next(E_sample))
			{
				int correct_class = (int)E_sample.label->data;

				auto pred = model(E_sample.pixels);
				auto sample_loss = cross_entropy_loss(pred, correct_class); // ✅ from loss.hpp

				batch_loss = batch_loss + sample_loss;

				auto probs = softmax(pred);
				int pred_class = 0;
				for (int j = 1; j < 10; j++)
					if (probs[j]->data > probs[pred_class]->data)
						pred_class = j;
				if (pred_class == correct_class)
					correct++;

				batch_count++;
				sample_count++;
			}

			batch_loss = batch_loss / (float)batch_count;
			batch_loss->backward();
			optimizer.step(params);

			epoch_loss += batch_loss->data;
			num_batches++;
		}

		std::cout << "epoch " << epoch
				  << " | loss = " << epoch_loss / num_batches
				  << " | acc = " << correct << "/" << num_train
				  << "\n";
	}

	return 0;
}