static always_inline void gen_fcmov (void *func,

                                     int ra, int rb, int rc)

{

    int l1;

    TCGv tmp;



    if (unlikely(rc == 31))

        return;



    l1 = gen_new_label();

    tmp = tcg_temp_new(TCG_TYPE_I64);

    if (ra != 31) {

        tmp = tcg_temp_new(TCG_TYPE_I64);

        tcg_gen_helper_1_1(func, tmp, cpu_fir[ra]);

    } else  {

        tmp = tcg_const_i64(0);

        tcg_gen_helper_1_1(func, tmp, tmp);

    }

    tcg_gen_brcondi_i64(TCG_COND_EQ, tmp, 0, l1);

    if (rb != 31)

        tcg_gen_mov_i64(cpu_fir[rc], cpu_fir[ra]);

    else

        tcg_gen_movi_i64(cpu_fir[rc], 0);

    gen_set_label(l1);

}
