#include "TimeTest.hpp"
#include <string>
#include <vector>
#include <windows.h>
#include <iostream>
#include <unordered_map>
#include <chrono>

namespace time_test
{
	using namespace std;
	class TimeInfo
	{
	public:
		TimeInfo() : time_mean(0.0), count(0) {}
		double time_start;
		double time_end;
		double time_cost;
		double time_mean;
		double get_mean_time()
		{
			return time_mean / (double)count;
		}
		int count;
	};

	class TimeTest
	{
		unordered_map<string, TimeInfo> times;
		_LARGE_INTEGER freq;
		_LARGE_INTEGER time_init;
		double time_begin;

	public:
		double get_time_now()
		{
			std::chrono::duration<double, std::milli> current_time =
				std::chrono::high_resolution_clock::now().time_since_epoch();
			return current_time.count();
		}

		void init_time_()
		{
			QueryPerformanceFrequency(&freq);
			QueryPerformanceCounter(&time_init);
			time_begin = get_time_now();
		}

		void print_time_spend_()
		{
			cout << get_time_now() - time_begin << std::endl;
		}

		void start_time_(std::string name)
		{
			_LARGE_INTEGER time_start;
			QueryPerformanceCounter(&time_start);
			double start_time = 1000.0 * (time_start.QuadPart - time_init.QuadPart) / (double)freq.QuadPart;
			times[name].time_start = start_time;
		}

		void end_time_(std::string name)
		{
			_LARGE_INTEGER time_end;
			QueryPerformanceCounter(&time_end);
			double end_time = 1000.0 * (time_end.QuadPart - time_init.QuadPart) / (double)freq.QuadPart;
			auto iter = times.find(name);
			if (iter != times.end())
			{
				auto &t = iter->second;
				t.time_end = end_time;
				t.time_cost = t.time_end - t.time_start;
				t.time_mean += t.time_cost;
				t.count++;
				// cout << name << " cost time " << t.time_cost << "ms" << endl;
			}
			else
			{
				cout << "error time end" << endl;
			}
		}

		void print_time_()
		{
			for (auto iter : times)
			{
				cout << iter.first << " cost time " << iter.second.time_cost << " mean time " << iter.second.get_mean_time() << endl;
			}
		}
	};

	static TimeTest tt;

	void init_time()
	{
		tt.init_time_();
	}
	void start_time(std::string name)
	{
	    tt.start_time_(name);
	}
	void end_time(std::string name)
	{
	    tt.end_time_(name);
	}
	void print_time()
	{
		tt.print_time_();
	}
}