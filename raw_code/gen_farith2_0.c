static always_inline void gen_farith2 (void *helper,

                                       int rb, int rc)

{

    if (unlikely(rc == 31))

      return;



    if (rb != 31)

        tcg_gen_helper_1_1(helper, cpu_fir[rc], cpu_fir[rb]);

    else {

        TCGv tmp = tcg_const_i64(0);

        tcg_gen_helper_1_1(helper, cpu_fir[rc], tmp);

        tcg_temp_free(tmp);

    }

}
