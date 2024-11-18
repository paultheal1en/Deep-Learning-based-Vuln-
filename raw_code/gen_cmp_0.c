static always_inline void gen_cmp(TCGCond cond,

                                  int ra, int rb, int rc,

                                  int islit, uint8_t lit)

{

    int l1, l2;

    TCGv tmp;



    if (unlikely(rc == 31))

    return;



    l1 = gen_new_label();

    l2 = gen_new_label();



    if (ra != 31) {

        tmp = tcg_temp_new(TCG_TYPE_I64);

        tcg_gen_mov_i64(tmp, cpu_ir[ra]);

    } else

        tmp = tcg_const_i64(0);

    if (islit)

        tcg_gen_brcondi_i64(cond, tmp, lit, l1);

    else

        tcg_gen_brcond_i64(cond, tmp, cpu_ir[rb], l1);



    tcg_gen_movi_i64(cpu_ir[rc], 0);

    tcg_gen_br(l2);

    gen_set_label(l1);

    tcg_gen_movi_i64(cpu_ir[rc], 1);

    gen_set_label(l2);

}
