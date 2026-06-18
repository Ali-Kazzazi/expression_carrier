#pragma once

#include <iostream>
#include <cmath>
#include <string>
#include <memory>
#include <vector>
#include <functional>
#include <set>

class EXIER : public std::enable_shared_from_this<EXIER>
{
public:
	float data;
	std::vector<std::shared_ptr<EXIER>> prev;
	std::string op;
	float grad = 0;
	std::function<void()> _backward = []() {};

	EXIER(float data, std::vector<std::shared_ptr<EXIER>> children = {}, std::string _op = "")
		: data(data)
	{
		this->prev = std::move(children);
		this->op = _op;
	}

	void backward()
	{
		std::vector<std::shared_ptr<EXIER>> topo;
		std::set<EXIER *> visited;

		std::function<void(std::shared_ptr<EXIER>)> buildTopo = [&](std::shared_ptr<EXIER> v)
		{
			if (visited.count(v.get()) == 0)
			{
				visited.insert(v.get());
				for (auto &child : v->prev)
					buildTopo(child);
				topo.push_back(v);
			}
		};

		buildTopo(shared_from_this());
		grad = 1.0f;
		std::reverse(topo.begin(), topo.end());

		for (auto &node : topo)
			node->_backward();

		topo.clear();
	}

	// std::string print()
	// {
	//     return std::string("EXIER: ") + "[data: " + std::to_string(data) + std::string("  grad: ") + std::to_string(grad) + std::string("  op: ") + op + " ]";
	// }

	std::string print()
	{
		char buf[64];
		snprintf(buf, sizeof(buf), "[ data: %.3f | grad: %.3f | op: %s ]", data, grad, op.c_str());
		return std::string(buf);
	}

	void repr(int depth = -1, int indent = 0, bool last = true, std::string prefix = "")
	{
		if (depth == 0)
			return;

		std::string connector = last ? "`-- " : "|-- ";
		std::string childPad = last ? "    " : "|   ";

		std::cout << prefix << connector << print() << "\n";

		for (size_t i = 0; i < prev.size(); i++)
		{
			bool isLast = (i == prev.size() - 1);
			prev[i]->repr(
				depth == -1 ? -1 : depth - 1,
				indent + 1,
				isLast,
				prefix + childPad);
		}
	}

	std::string toJSON()
	{
		std::string json = "{\"id\":\"" + std::to_string((size_t)this) + "\"," +
						   "\"data\":" + std::to_string(data) + "," +
						   "\"grad\":" + std::to_string(grad) + "," +
						   "\"op\":\"" + op + "\"," +
						   "\"prev\":[";

		for (size_t i = 0; i < prev.size(); i++)
		{
			if (i > 0)
				json += ",";
			json += prev[i]->toJSON();
		}
		json += "]}";
		return json;
	}

	std::shared_ptr<EXIER> operator+(std::shared_ptr<EXIER> other)
	{
		auto prev = std::vector<std::shared_ptr<EXIER>>{shared_from_this(), other};
		auto out = std::make_shared<EXIER>(data + other->data, prev, "+");

		out->_backward = [weak_out = std::weak_ptr<EXIER>(out), self = shared_from_this(), other]()
		{
			auto out = weak_out.lock();
			if (!out)
				return;
			self->grad += out->grad;
			other->grad += out->grad;
		};

		return out;
	}

	std::shared_ptr<EXIER> operator-(std::shared_ptr<EXIER> other)
	{
		auto prev = std::vector<std::shared_ptr<EXIER>>{shared_from_this(), other};
		auto out = std::make_shared<EXIER>(data - other->data, prev, "-");

		out->_backward = [weak_out = std::weak_ptr<EXIER>(out), self = shared_from_this(), other]()
		{
			auto out = weak_out.lock();
			if (!out)
				return;
			self->grad += out->grad;
			other->grad -= out->grad;
		};

		return out;
	}

	std::shared_ptr<EXIER> operator*(std::shared_ptr<EXIER> other)
	{
		auto prev = std::vector<std::shared_ptr<EXIER>>{shared_from_this(), other};
		auto out = std::make_shared<EXIER>(data * other->data, prev, "*");

		out->_backward = [weak_out = std::weak_ptr<EXIER>(out), self = shared_from_this(), other]()
		{
			auto out = weak_out.lock();
			if (!out)
				return;
			self->grad += other->data * out->grad;
			other->grad += self->data * out->grad;
		};

		return out;
	}

	std::shared_ptr<EXIER> operator/(std::shared_ptr<EXIER> other)
	{
		auto out = this->operator*(other->pow(-1.0f));

		return out;
	}

	std::shared_ptr<EXIER> pow(std::shared_ptr<EXIER> other)
	{
		auto prev = std::vector<std::shared_ptr<EXIER>>{shared_from_this(), other};
		auto out = std::make_shared<EXIER>(std::powf(data, other->data), prev, "**" + std::to_string(other->data));

		out->_backward = [weak_out = std::weak_ptr<EXIER>(out), self = shared_from_this(), other]()
		{
			auto out = weak_out.lock();
			if (!out)
				return;
			self->grad += other->data * std::powf(self->data, other->data - 1) * out->grad;
			other->grad += std::logf(self->data) * std::powf(self->data, other->data) * out->grad;
		};

		return out;
	}

	std::shared_ptr<EXIER> pow(float other)
	{
		auto prev = std::vector<std::shared_ptr<EXIER>>{shared_from_this()};
		auto out = std::make_shared<EXIER>(std::powf(data, other), prev, "**" + std::to_string(other));

		out->_backward = [weak_out = std::weak_ptr<EXIER>(out), self = shared_from_this(), other]()
		{
			auto out = weak_out.lock();
			if (!out)
				return;
			self->grad += other * std::powf(self->data, other - 1) * out->grad;
			// other is a constant, so it doesn't receive gradient
		};

		return out;
	}

	std::shared_ptr<EXIER> tanh()
	{
		float x = data;
		float t = (std::exp(2 * x) - 1) / (std::exp(2 * x) + 1); // tanh(x) = (e^(2x) - 1) / (e^(2x) + 1)
		auto prev = std::vector<std::shared_ptr<EXIER>>{shared_from_this()};
		auto out = std::make_shared<EXIER>(t, prev, "tanh");
		out->_backward = [weak_out = std::weak_ptr<EXIER>(out), self = shared_from_this()]()
		{
			auto out = weak_out.lock();
			if (!out)
				return;
			float t = out->data;
			self->grad += (1 - std::powf(t, 2)) * out->grad;
		};
		return out;
	}

	std::shared_ptr<EXIER> exp()
	{
		float x = data;
		float e = std::expf(x);
		auto prev = std::vector<std::shared_ptr<EXIER>>{shared_from_this()};
		auto out = std::make_shared<EXIER>(e, prev, "exp");
		out->_backward = [weak_out = std::weak_ptr<EXIER>(out), self = shared_from_this()]()
		{
			auto out = weak_out.lock();
			if (!out)
				return;
			self->grad += out->data * out->grad; // since out->data is e^x
		};
		return out;
	}

	std::shared_ptr<EXIER> log()
	{
		auto prev = std::vector<std::shared_ptr<EXIER>>{shared_from_this()};
		auto out = std::make_shared<EXIER>(std::logf(data), prev, "log");
		out->_backward = [weak_out = std::weak_ptr<EXIER>(out), self = shared_from_this()]()
		{
			auto out = weak_out.lock();
			if (!out)
				return;
			self->grad += (1.0f / self->data) * out->grad;
		};
		return out;
	}
};

// Free operators — correctly track prev
inline std::shared_ptr<EXIER> operator+(const std::shared_ptr<EXIER> &lhs, const std::shared_ptr<EXIER> &rhs)
{
	auto out = std::make_shared<EXIER>(lhs->data + rhs->data,
									   std::vector<std::shared_ptr<EXIER>>{lhs, rhs}, "+");
	out->_backward = [weak_out = std::weak_ptr<EXIER>(out), lhs, rhs]()
	{
		auto out = weak_out.lock();
		if (!out)
			return;
		lhs->grad += out->grad;
		rhs->grad += out->grad;
	};

	return out;
}

inline std::shared_ptr<EXIER> operator-(const std::shared_ptr<EXIER> &lhs, const std::shared_ptr<EXIER> &rhs)
{
	auto out = std::make_shared<EXIER>(lhs->data - rhs->data,
									   std::vector<std::shared_ptr<EXIER>>{lhs, rhs}, "-");
	out->_backward = [weak_out = std::weak_ptr<EXIER>(out), lhs, rhs]()
	{
		auto out = weak_out.lock();
		if (!out)
			return;
		lhs->grad += out->grad;
		rhs->grad -= out->grad;
	};
	return out;
}

inline std::shared_ptr<EXIER> operator*(const std::shared_ptr<EXIER> &lhs, const std::shared_ptr<EXIER> &rhs)
{
	float lhs_data = lhs->data, rhs_data = rhs->data;
	auto out = std::make_shared<EXIER>(lhs_data * rhs_data,
									   std::vector<std::shared_ptr<EXIER>>{lhs, rhs}, "*");
	out->_backward = [weak_out = std::weak_ptr<EXIER>(out), lhs, rhs, lhs_data, rhs_data]()
	{
		auto out = weak_out.lock();
		if (!out)
			return;
		lhs->grad += rhs_data * out->grad;
		rhs->grad += lhs_data * out->grad;
	};
	return out;
}

inline std::shared_ptr<EXIER> operator/(const std::shared_ptr<EXIER> &lhs, const std::shared_ptr<EXIER> &rhs)
{
	return lhs * rhs->pow(-1.0f); // ✅ operator* now visible
}

inline std::shared_ptr<EXIER> operator/(const std::shared_ptr<EXIER> &lhs, float rhs)
{
	return lhs * std::make_shared<EXIER>(std::powf(rhs, -1.0f));
}

inline std::shared_ptr<EXIER> operator/(float lhs, const std::shared_ptr<EXIER> &rhs)
{
	return std::make_shared<EXIER>(lhs) * rhs->pow(-1.0f);
}

inline std::shared_ptr<EXIER> pow(const std::shared_ptr<EXIER> &base, const std::shared_ptr<EXIER> &exponent)
{
	auto out = std::make_shared<EXIER>(std::powf(base->data, exponent->data),
									   std::vector<std::shared_ptr<EXIER>>{base, exponent}, "**");
	out->_backward = [weak_out = std::weak_ptr<EXIER>(out), base, exponent]()
	{
		auto out = weak_out.lock();
		if (!out)
			return;
		base->grad += exponent->data * std::powf(base->data, exponent->data - 1) * out->grad;
		exponent->grad += std::logf(base->data) * std::powf(base->data, exponent->data) * out->grad;
	};
	return out;
}

inline std::shared_ptr<EXIER> tanh(const std::shared_ptr<EXIER> &x)
{
	auto out = std::make_shared<EXIER>(std::tanhf(x->data), std::vector<std::shared_ptr<EXIER>>{x}, "tanh");
	out->_backward = [weak_out = std::weak_ptr<EXIER>(out), x]()
	{
		auto out = weak_out.lock();
		if (!out)
			return;
		float t = out->data;
		x->grad += (1 - std::powf(t, 2)) * out->grad;
	};
	return out;
}

inline std::shared_ptr<EXIER> exp(const std::shared_ptr<EXIER> &x)
{
	auto out = std::make_shared<EXIER>(std::expf(x->data), std::vector<std::shared_ptr<EXIER>>{x}, "exp");
	out->_backward = [weak_out = std::weak_ptr<EXIER>(out), x]()
	{
		auto out = weak_out.lock();
		if (!out)
			return;
		x->grad += out->data * out->grad; // since out->data is e^x
	};
	return out;
}

inline std::vector<std::shared_ptr<EXIER>> softmax(std::vector<std::shared_ptr<EXIER>> &x)
{
	// find max for numerical stability
	float max_val = x[0]->data;
	for (auto &xi : x)
		if (xi->data > max_val)
			max_val = xi->data;

	// exp(x - max)
	std::vector<std::shared_ptr<EXIER>> exps;
	for (auto &xi : x)
		exps.push_back((xi - std::make_shared<EXIER>(max_val))->exp());

	// sum
	auto sum = std::make_shared<EXIER>(0.0f);
	for (auto &e : exps)
		sum = sum + e;

	// divide
	std::vector<std::shared_ptr<EXIER>> out;
	for (auto &e : exps)
		out.push_back(e / sum);

	return out;
}

// float on the right
inline std::shared_ptr<EXIER> operator+(const std::shared_ptr<EXIER> &lhs, float rhs) { return lhs + std::make_shared<EXIER>(rhs); }
inline std::shared_ptr<EXIER> operator-(const std::shared_ptr<EXIER> &lhs, float rhs) { return lhs - std::make_shared<EXIER>(rhs); }
inline std::shared_ptr<EXIER> operator*(const std::shared_ptr<EXIER> &lhs, float rhs) { return lhs * std::make_shared<EXIER>(rhs); }

// float on the left
inline std::shared_ptr<EXIER> operator+(float lhs, const std::shared_ptr<EXIER> &rhs) { return std::make_shared<EXIER>(lhs) + rhs; }
inline std::shared_ptr<EXIER> operator-(float lhs, const std::shared_ptr<EXIER> &rhs) { return std::make_shared<EXIER>(lhs) - rhs; }
inline std::shared_ptr<EXIER> operator*(float lhs, const std::shared_ptr<EXIER> &rhs) { return std::make_shared<EXIER>(lhs) * rhs; }

inline std::shared_ptr<EXIER> pow(const std::shared_ptr<EXIER> &base, float exponent) { return pow(base, std::make_shared<EXIER>(exponent)); }
inline std::shared_ptr<EXIER> pow(float base, const std::shared_ptr<EXIER> &exponent) { return pow(std::make_shared<EXIER>(base), exponent); }
