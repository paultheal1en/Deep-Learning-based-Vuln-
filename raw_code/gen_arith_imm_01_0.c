static void gen_arith_imm(DisasContext *ctx, uint32_t opc,

                          int rt, int rs, int16_t imm)

{

    target_ulong uimm = (target_long)imm; /* Sign extend to 32/64 bits */

    const char *opn = "imm arith";



    if (rt == 0 && opc != OPC_ADDI && opc != OPC_DADDI) {

        /* If no destination, treat it as a NOP.

           For addi, we must generate the overflow exception when needed. */

        MIPS_DEBUG("NOP");

        return;

    }

    switch (opc) {

    case OPC_ADDI:

        {

            TCGv t0 = tcg_temp_local_new();

            TCGv t1 = tcg_temp_new();

            TCGv t2 = tcg_temp_new();

            int l1 = gen_new_label();



            gen_load_gpr(t1, rs);

            tcg_gen_addi_tl(t0, t1, uimm);

            tcg_gen_ext32s_tl(t0, t0);



            tcg_gen_xori_tl(t1, t1, ~uimm);

            tcg_gen_xori_tl(t2, t0, uimm);

            tcg_gen_and_tl(t1, t1, t2);

            tcg_temp_free(t2);

            tcg_gen_brcondi_tl(TCG_COND_GE, t1, 0, l1);

            tcg_temp_free(t1);

            /* operands of same sign, result different sign */

            generate_exception(ctx, EXCP_OVERFLOW);

            gen_set_label(l1);

            tcg_gen_ext32s_tl(t0, t0);

            gen_store_gpr(t0, rt);

            tcg_temp_free(t0);

        }

        opn = "addi";

        break;

    case OPC_ADDIU:

        if (rs != 0) {

            tcg_gen_addi_tl(cpu_gpr[rt], cpu_gpr[rs], uimm);

            tcg_gen_ext32s_tl(cpu_gpr[rt], cpu_gpr[rt]);

        } else {

            tcg_gen_movi_tl(cpu_gpr[rt], uimm);

        }

        opn = "addiu";

        break;

#if defined(TARGET_MIPS64)

    case OPC_DADDI:

        {

            TCGv t0 = tcg_temp_local_new();

            TCGv t1 = tcg_temp_new();

            TCGv t2 = tcg_temp_new();

            int l1 = gen_new_label();



            gen_load_gpr(t1, rs);

            tcg_gen_addi_tl(t0, t1, uimm);



            tcg_gen_xori_tl(t1, t1, ~uimm);

            tcg_gen_xori_tl(t2, t0, uimm);

            tcg_gen_and_tl(t1, t1, t2);

            tcg_temp_free(t2);

            tcg_gen_brcondi_tl(TCG_COND_GE, t1, 0, l1);

            tcg_temp_free(t1);

            /* operands of same sign, result different sign */

            generate_exception(ctx, EXCP_OVERFLOW);

            gen_set_label(l1);

            gen_store_gpr(t0, rt);

            tcg_temp_free(t0);

        }

        opn = "daddi";

        break;

    case OPC_DADDIU:

        if (rs != 0) {

            tcg_gen_addi_tl(cpu_gpr[rt], cpu_gpr[rs], uimm);

        } else {

            tcg_gen_movi_tl(cpu_gpr[rt], uimm);

        }

        opn = "daddiu";

        break;

#endif

    }

    (void)opn; /* avoid a compiler warning */

    MIPS_DEBUG("%s %s, %s, " TARGET_FMT_lx, opn, regnames[rt], regnames[rs], uimm);

}
