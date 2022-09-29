#pragma once
#include "pplx/pplxtasks.h"
namespace thread{
	constexpr int thread_num = 32;
    template<typename T>
    inline void compute_tile(T tile, const int range)
    {
		const int size = range;
		const int task_num = thread_num - 1;
		const int tile_size = size / task_num;
		std::array<pplx::task<void>, task_num + 1> tasks;

		for (int i = 0; i < task_num; i++)
		{
			tasks[i] = pplx::create_task([tile, size, tile_size, i] { tile(tile_size * i, tile_size * (i + 1)); });
		}
		tasks[task_num] = pplx::create_task([tile, size, tile_size, task_num] { tile(tile_size * task_num, size); });


		auto joinTask = pplx::when_all(std::begin(tasks), std::end(tasks));
		auto result = joinTask.wait();
    }

	template<typename T>
	inline void compute_tile4(T tile, const int range)
	{
		std::array<pplx::task<void>, 4> tasks;

		for (int i = 0; i < 4; i++)
		{
			tasks[i] = pplx::create_task([tile, i] { tile(i); });
		}
		
		auto joinTask = pplx::when_all(std::begin(tasks), std::end(tasks));
		auto result = joinTask.wait();
	}
}