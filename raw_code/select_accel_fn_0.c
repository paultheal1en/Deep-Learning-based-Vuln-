static bool select_accel_fn(const void *buf, size_t len)

{

    uintptr_t ibuf = (uintptr_t)buf;

#ifdef CONFIG_AVX2_OPT

    if (len % 128 == 0 && ibuf % 32 == 0 && (cpuid_cache & CACHE_AVX2)) {

        return buffer_zero_avx2(buf, len);

    }

    if (len % 64 == 0 && ibuf % 16 == 0 && (cpuid_cache & CACHE_SSE4)) {

        return buffer_zero_sse4(buf, len);

    }

#endif

    if (len % 64 == 0 && ibuf % 16 == 0 && (cpuid_cache & CACHE_SSE2)) {

        return buffer_zero_sse2(buf, len);

    }

    return buffer_zero_int(buf, len);

}
