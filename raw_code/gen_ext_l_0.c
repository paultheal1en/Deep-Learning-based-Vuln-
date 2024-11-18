static always_inline void gen_ext_l(void (*tcg_gen_ext_i64)(TCGv t0, TCGv t1),

                                    int ra, int rb, int rc,

                                    int islit, uint8_t lit)

{

    if (unlikely(rc == 31))

        return;



    if (ra != 31) {

        if (islit) {

                tcg_gen_shri_i64(cpu_ir[rc], cpu_ir[ra], (lit & 7) * 8);

        } else {

            TCGv tmp = tcg_temp_new(TCG_TYPE_I64);

            tcg_gen_andi_i64(tmp, cpu_ir[rb], 7);

            tcg_gen_shli_i64(tmp, tmp, 3);

            tcg_gen_shr_i64(cpu_ir[rc], cpu_ir[ra], tmp);

            tcg_temp_free(tmp);

        }

        if (tcg_gen_ext_i64)

            tcg_gen_ext_i64(cpu_ir[rc], cpu_ir[rc]);

    } else

        tcg_gen_movi_i64(cpu_ir[rc], 0);

}
