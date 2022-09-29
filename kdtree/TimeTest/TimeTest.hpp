#pragma once
#include <string>
namespace time_test
{
	void init_time();
	void start_time(std::string name);
	void end_time(std::string name);
	void print_time();
}
#if 1
#define INIT_TIME time_test::init_time
#define START_TIME time_test::start_time
#define END_TIME time_test::end_time
#define PRINTF_TIME time_test::print_time
#else
#define INIT_TIME()
#define START_TIME()
#define END_TIME()
#define PRINTF_TIME()
#endif