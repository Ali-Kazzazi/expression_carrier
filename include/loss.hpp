#pragma once

#include <vector>
#include <memory>
#include "exier.hpp"

// Mean Squared Error — for regression or one-hot MSE classification
inline std::shared_ptr<EXIER> mse_loss(std::vector<std::shared_ptr<EXIER>> &pred,
                                       std::vector<std::shared_ptr<EXIER>> &target)
{
    auto loss = std::make_shared<EXIER>(0.0f);
    for (size_t i = 0; i < pred.size(); i++)
    {
        auto diff = target[i] - pred[i];
        loss = loss + diff->pow(2.0f);
    }
    return loss / (float)pred.size();
}

// Mean Absolute Error
inline std::shared_ptr<EXIER> mae_loss(std::vector<std::shared_ptr<EXIER>> &pred,
                                       std::vector<std::shared_ptr<EXIER>> &target)
{
    auto loss = std::make_shared<EXIER>(0.0f);
    for (size_t i = 0; i < pred.size(); i++)
    {
        auto diff = target[i] - pred[i];
        // abs via sqrt(x^2) since no abs() defined on EXIER
        loss = loss + diff->pow(2.0f)->pow(0.5f);
    }
    return loss / (float)pred.size();
}

// Cross-entropy — expects raw logits, applies softmax internally
inline std::shared_ptr<EXIER> cross_entropy_loss(std::vector<std::shared_ptr<EXIER>> &logits, int correct_class)
{
    auto probs = softmax(logits);
    auto loss = (probs[correct_class] + 1e-7f)->log();
    return std::make_shared<EXIER>(0.0f) - loss;
}

// Binary cross-entropy — expects a single sigmoid output and a 0/1 target
inline std::shared_ptr<EXIER> bce_loss(std::shared_ptr<EXIER> pred, float target)
{
    auto p = pred + 1e-7f;
    auto one_minus_p = std::make_shared<EXIER>(1.0f) - pred + 1e-7f;

    if (target > 0.5f)
        return std::make_shared<EXIER>(0.0f) - p->log();
    else
        return std::make_shared<EXIER>(0.0f) - one_minus_p->log();
}