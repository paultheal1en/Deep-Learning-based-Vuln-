static always_inline void gen_cmov (TCGCond inv_cond,

                                    int ra, int rb, int rc,

                                    int islit, uint8_t lit, int mask)

{

    int l1;



    if (unlikely(rc == 31))

        return;



    l1 = gen_new_label();



    if (ra != 31) {

        if (mask) {

            TCGv tmp = tcg_temp_new(TCG_TYPE_I64);

            tcg_gen_andi_i64(tmp, cpu_ir[ra], 1);

            tcg_gen_brcondi_i64(inv_cond, tmp, 0, l1);

            tcg_temp_free(tmp);

        } else

            tcg_gen_brcondi_i64(inv_cond, cpu_ir[ra], 0, l1);

    } else {

        /* Very uncommon case - Do not bother to optimize.  */

        TCGv tmp = tcg_const_i64(0);

        tcg_gen_brcondi_i64(inv_cond, tmp, 0, l1);

        tcg_temp_free(tmp);

    }



    if (islit)

        tcg_gen_movi_i64(cpu_ir[rc], lit);

    else

        tcg_gen_mov_i64(cpu_ir[rc], cpu_ir[rb]);

    gen_set_label(l1);

}
