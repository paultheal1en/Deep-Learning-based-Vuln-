static always_inline target_ulong MASK (uint32_t start, uint32_t end)

{

    target_ulong ret;



#if defined(TARGET_PPC64)

    if (likely(start == 0)) {

        ret = (uint64_t)(-1ULL) << (63 - end);

    } else if (likely(end == 63)) {

        ret = (uint64_t)(-1ULL) >> start;

    }

#else

    if (likely(start == 0)) {

        ret = (uint32_t)(-1ULL) << (31  - end);

    } else if (likely(end == 31)) {

        ret = (uint32_t)(-1ULL) >> start;

    }

#endif

    else {

        ret = (((target_ulong)(-1ULL)) >> (start)) ^

            (((target_ulong)(-1ULL) >> (end)) >> 1);

        if (unlikely(start > end))

            return ~ret;

    }



    return ret;

}
