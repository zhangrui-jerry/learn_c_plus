#pragma once
#include <fstream>
#include <iostream>
#include <unordered_map>
#include <map>
#include <string>

#define LOG_OUT mylog::log.log
#define LOG_FILE mylog::log.logf
#define LOG_ADD mylog::log.add_value
#define LOG_ADDS mylog::log.add_values
#define LOG_ALL_MEAN mylog::log.log_all_mean
#define LOG_MEAN mylog::log.log_mean


namespace mylog
{
	using namespace std;
	class Log
	{
	public:
		Log()
			: add_value_count_(0)
		{
			file.open("../log/log.txt");
		}
		template <typename ... T>
		void log(T ... args)
		{
			((cout << args << " "), ...);
			cout << endl;
		}

		template <typename ... T>
		void logf(T ... args)
		{
			((file << args << " "), ...);
			file << endl;
		}

		template <typename ... T, typename ValName>
		void add_values(ValName& name, T ... args)
		{
			(add_value(name + std::to_string(add_value_count_++), args), ...);
			add_value_count_ = 0;
		}

		void add_value(const string& name, const double &val)
		{
			auto& temp = mean_val[name];
			if (temp.count_ > 1000)
				return;
			temp.value_ += val;
			temp.values_.push_back(val);
			temp.count_++;
			//log(name, "val is ", val);
		}

		void log_mean(const string& name)
		{
			log(name, "mean value", mean_val[name].get_mean(), "stdev", mean_val[name].get_stdev());
		}

		void log_all_mean()
		{
			for (const auto& iter : mean_val)
				log_mean(iter.first);
		}
	private:
		struct Val
		{
			Val(const double& value = 0) :value_(value), count_(0) {  }
			double value_;
			std::vector<double> values_;
			int count_;
			double get_mean()
			{
				return value_ / (double)count_;
			}
			double get_stdev()
			{
				double err = 0.0;
				double mean = get_mean();
				for (const auto& val : values_)
				{
					err += (val - mean) * (val - mean);
				}
				return std::sqrt(err / (double)count_);
			}
		};

		int add_value_count_;
		  map<string, Val> mean_val;
		ofstream file;
	};

	static Log log;
}