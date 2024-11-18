uint32_t HELPER(neon_narrow_sat_s32)(CPUState *env, uint64_t x)

{

    if ((int64_t)x != (int32_t)x) {

        SET_QC();

        return (x >> 63) ^ 0x7fffffff;

    }

    return x;

}
