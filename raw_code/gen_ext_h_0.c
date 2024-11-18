static always_inline void gen_ext_h(void (*tcg_gen_ext_i64)(TCGv t0, TCGv t1),

                                    int ra, int rb, int rc,

                                    int islit, uint8_t lit)

{

    if (unlikely(rc == 31))

        return;



    if (ra != 31) {

        if (islit) {

            if (lit != 0)

                tcg_gen_shli_i64(cpu_ir[rc], cpu_ir[ra], 64 - ((lit & 7) * 8));

            else

                tcg_gen_mov_i64(cpu_ir[rc], cpu_ir[ra]);

        } else {

            TCGv tmp1, tmp2;

            tmp1 = tcg_temp_new(TCG_TYPE_I64);

            tcg_gen_andi_i64(tmp1, cpu_ir[rb], 7);

            tcg_gen_shli_i64(tmp1, tmp1, 3);

            tmp2 = tcg_const_i64(64);

            tcg_gen_sub_i64(tmp1, tmp2, tmp1);

            tcg_temp_free(tmp2);

            tcg_gen_shl_i64(cpu_ir[rc], cpu_ir[ra], tmp1);

            tcg_temp_free(tmp1);

        }

        if (tcg_gen_ext_i64)

            tcg_gen_ext_i64(cpu_ir[rc], cpu_ir[rc]);

    } else

        tcg_gen_movi_i64(cpu_ir[rc], 0);

}
