static inline uint64_t hpet_calculate_diff(HPETTimer *t, uint64_t current)

{



    if (t->config & HPET_TN_32BIT) {

        uint32_t diff, cmp;



        cmp = (uint32_t)t->cmp;

        diff = cmp - (uint32_t)current;

        diff = (int32_t)diff > 0 ? diff : (uint32_t)0;

        return (uint64_t)diff;

    } else {

        uint64_t diff, cmp;



        cmp = t->cmp;

        diff = cmp - current;

        diff = (int64_t)diff > 0 ? diff : (uint64_t)0;

        return diff;

    }

}
