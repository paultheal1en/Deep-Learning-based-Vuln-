static void gen_arith(DisasContext *ctx, uint32_t opc,

                      int rd, int rs, int rt)

{

    const char *opn = "arith";



    if (rd == 0 && opc != OPC_ADD && opc != OPC_SUB

       && opc != OPC_DADD && opc != OPC_DSUB) {

        /* If no destination, treat it as a NOP.

           For add & sub, we must generate the overflow exception when needed. */

        MIPS_DEBUG("NOP");

        return;

    }



    switch (opc) {

    case OPC_ADD:

        {

            TCGv t0 = tcg_temp_local_new();

            TCGv t1 = tcg_temp_new();

            TCGv t2 = tcg_temp_new();

            int l1 = gen_new_label();



            gen_load_gpr(t1, rs);

            gen_load_gpr(t2, rt);

            tcg_gen_add_tl(t0, t1, t2);

            tcg_gen_ext32s_tl(t0, t0);

            tcg_gen_xor_tl(t1, t1, t2);

            tcg_gen_xor_tl(t2, t0, t2);

            tcg_gen_andc_tl(t1, t2, t1);

            tcg_temp_free(t2);

            tcg_gen_brcondi_tl(TCG_COND_GE, t1, 0, l1);

            tcg_temp_free(t1);

            /* operands of same sign, result different sign */

            generate_exception(ctx, EXCP_OVERFLOW);

            gen_set_label(l1);

            gen_store_gpr(t0, rd);

            tcg_temp_free(t0);

        }

        opn = "add";

        break;

    case OPC_ADDU:

        if (rs != 0 && rt != 0) {

            tcg_gen_add_tl(cpu_gpr[rd], cpu_gpr[rs], cpu_gpr[rt]);

            tcg_gen_ext32s_tl(cpu_gpr[rd], cpu_gpr[rd]);

        } else if (rs == 0 && rt != 0) {

            tcg_gen_mov_tl(cpu_gpr[rd], cpu_gpr[rt]);

        } else if (rs != 0 && rt == 0) {

            tcg_gen_mov_tl(cpu_gpr[rd], cpu_gpr[rs]);

        } else {

            tcg_gen_movi_tl(cpu_gpr[rd], 0);

        }

        opn = "addu";

        break;

    case OPC_SUB:

        {

            TCGv t0 = tcg_temp_local_new();

            TCGv t1 = tcg_temp_new();

            TCGv t2 = tcg_temp_new();

            int l1 = gen_new_label();



            gen_load_gpr(t1, rs);

            gen_load_gpr(t2, rt);

            tcg_gen_sub_tl(t0, t1, t2);

            tcg_gen_ext32s_tl(t0, t0);

            tcg_gen_xor_tl(t2, t1, t2);

            tcg_gen_xor_tl(t1, t0, t1);

            tcg_gen_and_tl(t1, t1, t2);

            tcg_temp_free(t2);

            tcg_gen_brcondi_tl(TCG_COND_GE, t1, 0, l1);

            tcg_temp_free(t1);

            /* operands of different sign, first operand and result different sign */

            generate_exception(ctx, EXCP_OVERFLOW);

            gen_set_label(l1);

            gen_store_gpr(t0, rd);

            tcg_temp_free(t0);

        }

        opn = "sub";

        break;

    case OPC_SUBU:

        if (rs != 0 && rt != 0) {

            tcg_gen_sub_tl(cpu_gpr[rd], cpu_gpr[rs], cpu_gpr[rt]);

            tcg_gen_ext32s_tl(cpu_gpr[rd], cpu_gpr[rd]);

        } else if (rs == 0 && rt != 0) {

            tcg_gen_neg_tl(cpu_gpr[rd], cpu_gpr[rt]);

            tcg_gen_ext32s_tl(cpu_gpr[rd], cpu_gpr[rd]);

        } else if (rs != 0 && rt == 0) {

            tcg_gen_mov_tl(cpu_gpr[rd], cpu_gpr[rs]);

        } else {

            tcg_gen_movi_tl(cpu_gpr[rd], 0);

        }

        opn = "subu";

        break;

#if defined(TARGET_MIPS64)

    case OPC_DADD:

        {

            TCGv t0 = tcg_temp_local_new();

            TCGv t1 = tcg_temp_new();

            TCGv t2 = tcg_temp_new();

            int l1 = gen_new_label();



            gen_load_gpr(t1, rs);

            gen_load_gpr(t2, rt);

            tcg_gen_add_tl(t0, t1, t2);

            tcg_gen_xor_tl(t1, t1, t2);

            tcg_gen_xor_tl(t2, t0, t2);

            tcg_gen_andc_tl(t1, t2, t1);

            tcg_temp_free(t2);

            tcg_gen_brcondi_tl(TCG_COND_GE, t1, 0, l1);

            tcg_temp_free(t1);

            /* operands of same sign, result different sign */

            generate_exception(ctx, EXCP_OVERFLOW);

            gen_set_label(l1);

            gen_store_gpr(t0, rd);

            tcg_temp_free(t0);

        }

        opn = "dadd";

        break;

    case OPC_DADDU:

        if (rs != 0 && rt != 0) {

            tcg_gen_add_tl(cpu_gpr[rd], cpu_gpr[rs], cpu_gpr[rt]);

        } else if (rs == 0 && rt != 0) {

            tcg_gen_mov_tl(cpu_gpr[rd], cpu_gpr[rt]);

        } else if (rs != 0 && rt == 0) {

            tcg_gen_mov_tl(cpu_gpr[rd], cpu_gpr[rs]);

        } else {

            tcg_gen_movi_tl(cpu_gpr[rd], 0);

        }

        opn = "daddu";

        break;

    case OPC_DSUB:

        {

            TCGv t0 = tcg_temp_local_new();

            TCGv t1 = tcg_temp_new();

            TCGv t2 = tcg_temp_new();

            int l1 = gen_new_label();



            gen_load_gpr(t1, rs);

            gen_load_gpr(t2, rt);

            tcg_gen_sub_tl(t0, t1, t2);

            tcg_gen_xor_tl(t2, t1, t2);

            tcg_gen_xor_tl(t1, t0, t1);

            tcg_gen_and_tl(t1, t1, t2);

            tcg_temp_free(t2);

            tcg_gen_brcondi_tl(TCG_COND_GE, t1, 0, l1);

            tcg_temp_free(t1);

            /* operands of different sign, first operand and result different sign */

            generate_exception(ctx, EXCP_OVERFLOW);

            gen_set_label(l1);

            gen_store_gpr(t0, rd);

            tcg_temp_free(t0);

        }

        opn = "dsub";

        break;

    case OPC_DSUBU:

        if (rs != 0 && rt != 0) {

            tcg_gen_sub_tl(cpu_gpr[rd], cpu_gpr[rs], cpu_gpr[rt]);

        } else if (rs == 0 && rt != 0) {

            tcg_gen_neg_tl(cpu_gpr[rd], cpu_gpr[rt]);

        } else if (rs != 0 && rt == 0) {

            tcg_gen_mov_tl(cpu_gpr[rd], cpu_gpr[rs]);

        } else {

            tcg_gen_movi_tl(cpu_gpr[rd], 0);

        }

        opn = "dsubu";

        break;

#endif

    case OPC_MUL:

        if (likely(rs != 0 && rt != 0)) {

            tcg_gen_mul_tl(cpu_gpr[rd], cpu_gpr[rs], cpu_gpr[rt]);

            tcg_gen_ext32s_tl(cpu_gpr[rd], cpu_gpr[rd]);

        } else {

            tcg_gen_movi_tl(cpu_gpr[rd], 0);

        }

        opn = "mul";

        break;

    }

    (void)opn; /* avoid a compiler warning */

    MIPS_DEBUG("%s %s, %s, %s", opn, regnames[rd], regnames[rs], regnames[rt]);

}
