static inline void gen_neon_narrow_satu(int size, TCGv dest, TCGv src)

{

    switch (size) {

    case 0: gen_helper_neon_narrow_sat_u8(dest, cpu_env, src); break;

    case 1: gen_helper_neon_narrow_sat_u16(dest, cpu_env, src); break;

    case 2: gen_helper_neon_narrow_sat_u32(dest, cpu_env, src); break;

    default: abort();

    }

}
