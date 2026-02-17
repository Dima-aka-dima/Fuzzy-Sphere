#include <cstdint>
#include <complex>
#include <cmath>
// #include <numbers> // c++20
#include <vector>
#include <iostream>
#include <fstream>
#include <iomanip>

using i8  = int8_t;
using i16 = int16_t;
using i32 = int32_t;
using i64 = int64_t;

using u8  = uint8_t;
using u16 = uint16_t;
using u32 = uint32_t;
using u64 = uint64_t;
using u128 = __uint128_t;

using sz  = size_t;

using f32 = float;
using f64 = double;

const std::complex<f64> I(0, 1);

// const f64 pi = std::numbers::pi;
const f64 pi = M_PI;

template<class T> T sign(T x) {return (x > 0) - (x < 0); };


template<typename T>
std::ostream& operator<<(std::ostream& stream, const std::vector<T>& v) 
{
	for(auto a: v) stream << a << " ";
	return stream;
}

constexpr u64 binomial(u64 n, u64 k) 
{
	if (k > n) return 0;
	if (k == 0 || k == n) return 1;
	if (k > n - k) k = n - k;
	
	u64 result = 1;
	for (u64 i = 0; i < k; ++i) {
		result *= (n - i);
		result /= (i + 1);
	}
	return result;
}


template<typename... Vectors>
void writeCSV(const std::string& filename, const std::tuple<Vectors...>& data) 
{
    std::ofstream file(filename);
	sz size = std::get<0>(data).size();	
	for (sz row = 0; row < size; ++row) 
    {
		sz col = 0;
        std::apply([&](const auto&... vecs) { 
                ((file << vecs[row] << (++col < sizeof...(Vectors) ? "," : "")), ...); 
        }, data);

		file << '\n';
	}
}

#include <sys/mman.h>
#include <cstdlib>
#include <new>
#include <limits>
#include <memory>

template <typename T, sz MB = 2>
struct HugePageAllocator {
	using value_type = T;
	
	static_assert(MB != 0 and (MB & (MB - 1)) == 0, "MB must be a power of 2");
	
	template <typename U>
	struct rebind { using other = HugePageAllocator<U, MB>; };

	HugePageAllocator() noexcept {}
	template <typename U> HugePageAllocator(const HugePageAllocator<U, MB>&) noexcept {}

	T* allocate(sz n) {
		if (n > std::numeric_limits<sz>::max() / sizeof(T))
			throw std::bad_alloc();

		size_t size = n * sizeof(T);
		void* ptr = nullptr;
		size_t alignment = MB * 1024 * 1024;

		if (posix_memalign(&ptr, alignment, size) != 0) {
			throw std::bad_alloc();
		}

		madvise(ptr, size, MADV_HUGEPAGE);

		return static_cast<T*>(ptr);
	}

	void deallocate(T* p, sz) noexcept { free(p); }
};

#ifdef __linux__
	template <typename T, size_t MB = 2>
	using PlatformAllocator = HugePageAllocator<T, MB>;
#else
	template <typename T, size_t MB = 0>
	using PlatformAllocator = std::allocator<T>;
#endif

