static inline uint32_t efsctui(uint32_t val)

{

    CPU_FloatU u;



    u.l = val;

    /* NaN are not treated the same way IEEE 754 does */

    if (unlikely(float32_is_nan(u.f)))

        return 0;



    return float32_to_uint32(u.f, &env->vec_status);

}
