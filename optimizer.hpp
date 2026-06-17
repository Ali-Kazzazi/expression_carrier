#pragma once

#include <iostream>
#include <vector>
#include "exier_powDiv.hpp"

class GD
{
public:
    float lr;
    int t = 0;

    std::vector<float> m; // first moment
    std::vector<float> v; // second moment

    GD(int num_params, float lr = 0.001f) : lr(lr) {}

    void step(std::vector<std::shared_ptr<EXIER>> &params)
    {
        t++;
        for (size_t i = 0; i < params.size(); i++)
        {
            float g = params[i]->grad;
            params[i]->data -= lr * g;
        }
    }
};

class Adam
{
public:
    float lr;
    float beta1;
    float beta2;
    float epsilon;
    int t = 0;

    std::vector<float> m; // first moment
    std::vector<float> v; // second moment

    Adam(int num_params, float lr = 0.001f, float beta1 = 0.9f, float beta2 = 0.999f, float epsilon = 1e-8f)
        : lr(lr), beta1(beta1), beta2(beta2), epsilon(epsilon),
          m(num_params, 0.0f), v(num_params, 0.0f) {}

    void step(std::vector<std::shared_ptr<EXIER>> &params)
    {
        t++;
        for (size_t i = 0; i < params.size(); i++)
        {
            float g = params[i]->grad;
            m[i] = beta1 * m[i] + (1 - beta1) * g;
            v[i] = beta2 * v[i] + (1 - beta2) * g * g;

            float m_hat = m[i] / (1 - std::powf(beta1, t)); // bias correction
            float v_hat = v[i] / (1 - std::powf(beta2, t));

            params[i]->data -= lr * m_hat / (std::sqrtf(v_hat) + epsilon);
        }
    }
};