#pragma once

#include <iostream>
#include <string>
#include <vector>
#include <set>
#include <assert.h>
#include <array>
#include <ctime>

#include "exier.hpp"

using Activation = std::function<std::shared_ptr<EXIER>(std::shared_ptr<EXIER>)>;

inline Activation act_tanh()
{
     return [](std::shared_ptr<EXIER> x)
     { return x->tanh(); };
}
inline Activation act_linear()
{
     return [](std::shared_ptr<EXIER> x)
     { return x; };
}

inline Activation act_relu()
{
     return [](std::shared_ptr<EXIER> x)
     {
          float val = x->data > 0 ? x->data : 0.0f;
          auto out = std::make_shared<EXIER>(val, std::vector<std::shared_ptr<EXIER>>{x}, "relu");
          out->_backward = [weak_out = std::weak_ptr<EXIER>(out), x]()
          {
               auto out = weak_out.lock();
               if (!out)
                    return;
               x->grad += (x->data > 0 ? 1.0f : 0.0f) * out->grad;
          };
          return out;
     };
}

inline Activation act_sigmoid()
{
     return [](std::shared_ptr<EXIER> x)
     {
          float val = 1.0f / (1.0f + std::expf(-x->data));
          auto out = std::make_shared<EXIER>(val, std::vector<std::shared_ptr<EXIER>>{x}, "sigmoid");
          out->_backward = [weak_out = std::weak_ptr<EXIER>(out), x]()
          {
               auto out = weak_out.lock();
               if (!out)
                    return;
               x->grad += out->data * (1.0f - out->data) * out->grad; // sigmoid' = sigmoid * (1 - sigmoid)
          };
          return out;
     };
}

class Neuron
{
public:
     int num_input;
     Activation activation;
     std::vector<std::shared_ptr<EXIER>> weights;
     std::shared_ptr<EXIER> bias;

     Neuron(int num_inputs, Activation act = act_tanh()) : num_input(num_inputs), activation(act)
     {
          for (int i = 0; i < num_inputs; ++i)
               weights.push_back(std::make_shared<EXIER>(static_cast<float>(rand()) / RAND_MAX * 2 - 1));
          bias = std::make_shared<EXIER>(0.0f);
     }

     std::vector<std::shared_ptr<EXIER>> parameters()
     {
          std::vector<std::shared_ptr<EXIER>> params = weights;
          params.push_back(bias);
          return params;
     }

     std::shared_ptr<EXIER> operator()(std::vector<std::shared_ptr<EXIER>> &x)
     {
          assert(weights.size() == x.size());

          auto out = std::make_shared<EXIER>(0.0f);
          for (size_t i = 0; i < num_input; i++)
               out = out + (weights[i] * x[i]);

          out = out + bias;
          out = activation(out);
          return out;
     }
};

class Layer
{
public:
     int num_input;
     int num_output;
     std::vector<std::shared_ptr<Neuron>> neurons;

     Layer(int num_inputs, int num_outputs, Activation act = act_tanh())
         : num_input(num_inputs), num_output(num_outputs)
     {
          for (int i = 0; i < num_output; ++i)
               neurons.push_back(std::make_shared<Neuron>(num_input, act));
     }

     std::vector<std::shared_ptr<EXIER>> parameters()
     {
          std::vector<std::shared_ptr<EXIER>> params;
          for (auto &n : neurons)
               for (auto &p : n->parameters())
                    params.push_back(p);
          return params;
     }

     std::vector<std::shared_ptr<EXIER>> operator()(std::vector<std::shared_ptr<EXIER>> &x)
     {
          assert(x.size() == num_input);

          std::vector<std::shared_ptr<EXIER>> out;
          for (size_t i = 0; i < num_output; i++)
               out.push_back(neurons[i]->operator()(x));
          return out;
     }
};

class MLP
{
public:
     int num_input;
     std::vector<int> output_num;
     std::vector<std::shared_ptr<Layer>> layers;

     MLP(int num_inputs, std::vector<int> output_nums, std::vector<Activation> activations = {})
         : num_input(num_inputs), output_num(output_nums)
     {
          if (activations.empty())
               activations.assign(output_nums.size(), act_tanh());

          assert(activations.size() == output_nums.size());

          std::vector<int> all_layers = {num_inputs};
          all_layers.insert(all_layers.end(), output_nums.begin(), output_nums.end());

          for (size_t i = 0; i < all_layers.size() - 1; i++)
               layers.push_back(std::make_shared<Layer>(all_layers[i], all_layers[i + 1], activations[i]));
     }

     std::vector<std::shared_ptr<EXIER>> parameters()
     {
          std::vector<std::shared_ptr<EXIER>> params;
          for (auto &l : layers)
               for (auto &p : l->parameters())
                    params.push_back(p);
          return params;
     }

     std::vector<std::shared_ptr<EXIER>> operator()(std::vector<std::shared_ptr<EXIER>> &x)
     {
          assert(x.size() == num_input);

          std::vector<std::shared_ptr<EXIER>> out = x;
          for (size_t i = 0; i < layers.size(); i++)
               out = layers[i]->operator()(out);
          return out;
     }
};