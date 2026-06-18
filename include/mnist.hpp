#pragma once

#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <stdexcept>
#include "exier.hpp"

class read_mnist
{
public:
    struct sample_vector
    {
        int label;
        std::vector<float> pixels;
    };

    struct exier_vector
    {
        std::shared_ptr<EXIER> label;
        std::vector<std::shared_ptr<EXIER>> pixels;
    };

    explicit read_mnist(const std::string &filename)
    {
        file.open(filename);

        if (!file.is_open())
            throw std::runtime_error("Failed to open file: " + filename);
    }

    void reset()
    {
        file.clear();
        file.seekg(0);
        std::string header;
        std::getline(file, header);
    }

    bool next(exier_vector &s)
    {
        std::string line;

        if (!std::getline(file, line))
            return false;

        std::stringstream ss(line);
        std::string cell;

        // Read label
        if (!std::getline(ss, cell, ','))
            return false;

        s.label = std::make_shared<EXIER>(std::stof(cell));

        // Read pixels
        s.pixels.clear();
        s.pixels.reserve(784);

        while (std::getline(ss, cell, ','))
        {
            s.pixels.push_back(std::make_shared<EXIER>(std::stof(cell) / 255.0f));
        }

        return true;
    }

    bool next(sample_vector &s)
    {
        std::string line;

        if (!std::getline(file, line))
            return false;

        std::stringstream ss(line);
        std::string cell;

        // Read label
        if (!std::getline(ss, cell, ','))
            return false;

        s.label = std::stoi(cell);

        // Read pixels
        s.pixels.clear();
        s.pixels.reserve(784);

        while (std::getline(ss, cell, ','))
        {
            s.pixels.push_back(std::stof(cell) / 255.0f);
        }

        return true;
    }

private:
    std::ifstream file;
};