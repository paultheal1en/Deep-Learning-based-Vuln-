static inline void gen_neon_mull(TCGv_i64 dest, TCGv a, TCGv b, int size, int u)

{

    TCGv_i64 tmp;



    switch ((size << 1) | u) {

    case 0: gen_helper_neon_mull_s8(dest, a, b); break;

    case 1: gen_helper_neon_mull_u8(dest, a, b); break;

    case 2: gen_helper_neon_mull_s16(dest, a, b); break;

    case 3: gen_helper_neon_mull_u16(dest, a, b); break;

    case 4:

        tmp = gen_muls_i64_i32(a, b);

        tcg_gen_mov_i64(dest, tmp);


        break;

    case 5:

        tmp = gen_mulu_i64_i32(a, b);

        tcg_gen_mov_i64(dest, tmp);


        break;

    default: abort();

    }



    /* gen_helper_neon_mull_[su]{8|16} do not free their parameters.

       Don't forget to clean them now.  */

    if (size < 2) {

        tcg_temp_free_i32(a);

        tcg_temp_free_i32(b);

    }

}