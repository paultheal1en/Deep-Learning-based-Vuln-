static ExitStatus translate_one(DisasContext *ctx, uint32_t insn)

{

    int32_t disp21, disp16, disp12 __attribute__((unused));

    uint16_t fn11;

    uint8_t opc, ra, rb, rc, fpfn, fn7, lit;

    bool islit;

    TCGv va, vb, vc, tmp;

    TCGv_i32 t32;

    ExitStatus ret;



    /* Decode all instruction fields */

    opc = extract32(insn, 26, 6);

    ra = extract32(insn, 21, 5);

    rb = extract32(insn, 16, 5);

    rc = extract32(insn, 0, 5);

    islit = extract32(insn, 12, 1);

    lit = extract32(insn, 13, 8);



    disp21 = sextract32(insn, 0, 21);

    disp16 = sextract32(insn, 0, 16);

    disp12 = sextract32(insn, 0, 12);



    fn11 = extract32(insn, 5, 11);

    fpfn = extract32(insn, 5, 6);

    fn7 = extract32(insn, 5, 7);



    if (rb == 31 && !islit) {

        islit = true;

        lit = 0;

    }



    ret = NO_EXIT;

    switch (opc) {

    case 0x00:

        /* CALL_PAL */

        ret = gen_call_pal(ctx, insn & 0x03ffffff);

        break;

    case 0x01:

        /* OPC01 */

        goto invalid_opc;

    case 0x02:

        /* OPC02 */

        goto invalid_opc;

    case 0x03:

        /* OPC03 */

        goto invalid_opc;

    case 0x04:

        /* OPC04 */

        goto invalid_opc;

    case 0x05:

        /* OPC05 */

        goto invalid_opc;

    case 0x06:

        /* OPC06 */

        goto invalid_opc;

    case 0x07:

        /* OPC07 */

        goto invalid_opc;



    case 0x09:

        /* LDAH */

        disp16 = (uint32_t)disp16 << 16;

        /* fall through */

    case 0x08:

        /* LDA */

        va = dest_gpr(ctx, ra);

        /* It's worth special-casing immediate loads.  */

        if (rb == 31) {

            tcg_gen_movi_i64(va, disp16);

        } else {

            tcg_gen_addi_i64(va, load_gpr(ctx, rb), disp16);

        }

        break;



    case 0x0A:

        /* LDBU */

        REQUIRE_TB_FLAG(TB_FLAGS_AMASK_BWX);

        gen_load_mem(ctx, &tcg_gen_qemu_ld8u, ra, rb, disp16, 0, 0);

        break;

    case 0x0B:

        /* LDQ_U */

        gen_load_mem(ctx, &tcg_gen_qemu_ld64, ra, rb, disp16, 0, 1);

        break;

    case 0x0C:

        /* LDWU */

        REQUIRE_TB_FLAG(TB_FLAGS_AMASK_BWX);

        gen_load_mem(ctx, &tcg_gen_qemu_ld16u, ra, rb, disp16, 0, 0);

        break;

    case 0x0D:

        /* STW */

        REQUIRE_TB_FLAG(TB_FLAGS_AMASK_BWX);

        gen_store_mem(ctx, &tcg_gen_qemu_st16, ra, rb, disp16, 0, 0);

        break;

    case 0x0E:

        /* STB */

        REQUIRE_TB_FLAG(TB_FLAGS_AMASK_BWX);

        gen_store_mem(ctx, &tcg_gen_qemu_st8, ra, rb, disp16, 0, 0);

        break;

    case 0x0F:

        /* STQ_U */

        gen_store_mem(ctx, &tcg_gen_qemu_st64, ra, rb, disp16, 0, 1);

        break;



    case 0x10:

        vc = dest_gpr(ctx, rc);

        vb = load_gpr_lit(ctx, rb, lit, islit);



        if (ra == 31) {

            if (fn7 == 0x00) {

                /* Special case ADDL as SEXTL.  */

                tcg_gen_ext32s_i64(vc, vb);

                break;

            }

            if (fn7 == 0x29) {

                /* Special case SUBQ as NEGQ.  */

                tcg_gen_neg_i64(vc, vb);

                break;

            }

        }



        va = load_gpr(ctx, ra);

        switch (fn7) {

        case 0x00:

            /* ADDL */

            tcg_gen_add_i64(vc, va, vb);

            tcg_gen_ext32s_i64(vc, vc);

            break;

        case 0x02:

            /* S4ADDL */

            tmp = tcg_temp_new();

            tcg_gen_shli_i64(tmp, va, 2);

            tcg_gen_add_i64(tmp, tmp, vb);

            tcg_gen_ext32s_i64(vc, tmp);

            tcg_temp_free(tmp);

            break;

        case 0x09:

            /* SUBL */

            tcg_gen_sub_i64(vc, va, vb);

            tcg_gen_ext32s_i64(vc, vc);

            break;

        case 0x0B:

            /* S4SUBL */

            tmp = tcg_temp_new();

            tcg_gen_shli_i64(tmp, va, 2);

            tcg_gen_sub_i64(tmp, tmp, vb);

            tcg_gen_ext32s_i64(vc, tmp);

            tcg_temp_free(tmp);

            break;

        case 0x0F:

            /* CMPBGE */

            gen_helper_cmpbge(vc, va, vb);

            break;

        case 0x12:

            /* S8ADDL */

            tmp = tcg_temp_new();

            tcg_gen_shli_i64(tmp, va, 3);

            tcg_gen_add_i64(tmp, tmp, vb);

            tcg_gen_ext32s_i64(vc, tmp);

            tcg_temp_free(tmp);

            break;

        case 0x1B:

            /* S8SUBL */

            tmp = tcg_temp_new();

            tcg_gen_shli_i64(tmp, va, 3);

            tcg_gen_sub_i64(tmp, tmp, vb);

            tcg_gen_ext32s_i64(vc, tmp);

            tcg_temp_free(tmp);

            break;

        case 0x1D:

            /* CMPULT */

            tcg_gen_setcond_i64(TCG_COND_LTU, vc, va, vb);

            break;

        case 0x20:

            /* ADDQ */

            tcg_gen_add_i64(vc, va, vb);

            break;

        case 0x22:

            /* S4ADDQ */

            tmp = tcg_temp_new();

            tcg_gen_shli_i64(tmp, va, 2);

            tcg_gen_add_i64(vc, tmp, vb);

            tcg_temp_free(tmp);

            break;

        case 0x29:

            /* SUBQ */

            tcg_gen_sub_i64(vc, va, vb);

            break;

        case 0x2B:

            /* S4SUBQ */

            tmp = tcg_temp_new();

            tcg_gen_shli_i64(tmp, va, 2);

            tcg_gen_sub_i64(vc, tmp, vb);

            tcg_temp_free(tmp);

            break;

        case 0x2D:

            /* CMPEQ */

            tcg_gen_setcond_i64(TCG_COND_EQ, vc, va, vb);

            break;

        case 0x32:

            /* S8ADDQ */

            tmp = tcg_temp_new();

            tcg_gen_shli_i64(tmp, va, 3);

            tcg_gen_add_i64(vc, tmp, vb);

            tcg_temp_free(tmp);

            break;

        case 0x3B:

            /* S8SUBQ */

            tmp = tcg_temp_new();

            tcg_gen_shli_i64(tmp, va, 3);

            tcg_gen_sub_i64(vc, tmp, vb);

            tcg_temp_free(tmp);

            break;

        case 0x3D:

            /* CMPULE */

            tcg_gen_setcond_i64(TCG_COND_LEU, vc, va, vb);

            break;

        case 0x40:

            /* ADDL/V */

            gen_helper_addlv(vc, cpu_env, va, vb);

            break;

        case 0x49:

            /* SUBL/V */

            gen_helper_sublv(vc, cpu_env, va, vb);

            break;

        case 0x4D:

            /* CMPLT */

            tcg_gen_setcond_i64(TCG_COND_LT, vc, va, vb);

            break;

        case 0x60:

            /* ADDQ/V */

            gen_helper_addqv(vc, cpu_env, va, vb);

            break;

        case 0x69:

            /* SUBQ/V */

            gen_helper_subqv(vc, cpu_env, va, vb);

            break;

        case 0x6D:

            /* CMPLE */

            tcg_gen_setcond_i64(TCG_COND_LE, vc, va, vb);

            break;

        default:

            goto invalid_opc;

        }

        break;



    case 0x11:

        if (fn7 == 0x20) {

            if (rc == 31) {

                /* Special case BIS as NOP.  */

                break;

            }

            if (ra == 31) {

                /* Special case BIS as MOV.  */

                vc = dest_gpr(ctx, rc);

                if (islit) {

                    tcg_gen_movi_i64(vc, lit);

                } else {

                    tcg_gen_mov_i64(vc, load_gpr(ctx, rb));

                }

                break;

            }

        }



        vc = dest_gpr(ctx, rc);

        vb = load_gpr_lit(ctx, rb, lit, islit);



        if (fn7 == 0x28 && ra == 31) {

            /* Special case ORNOT as NOT.  */

            tcg_gen_not_i64(vc, vb);

            break;

        }



        va = load_gpr(ctx, ra);

        switch (fn7) {

        case 0x00:

            /* AND */

            tcg_gen_and_i64(vc, va, vb);

            break;

        case 0x08:

            /* BIC */

            tcg_gen_andc_i64(vc, va, vb);

            break;

        case 0x14:

            /* CMOVLBS */

            tmp = tcg_temp_new();

            tcg_gen_andi_i64(tmp, va, 1);

            tcg_gen_movcond_i64(TCG_COND_NE, vc, tmp, load_zero(ctx),

                                vb, load_gpr(ctx, rc));

            tcg_temp_free(tmp);

            break;

        case 0x16:

            /* CMOVLBC */

            tmp = tcg_temp_new();

            tcg_gen_andi_i64(tmp, va, 1);

            tcg_gen_movcond_i64(TCG_COND_EQ, vc, tmp, load_zero(ctx),

                                vb, load_gpr(ctx, rc));

            tcg_temp_free(tmp);

            break;

        case 0x20:

            /* BIS */

            tcg_gen_or_i64(vc, va, vb);

            break;

        case 0x24:

            /* CMOVEQ */

            tcg_gen_movcond_i64(TCG_COND_EQ, vc, va, load_zero(ctx),

                                vb, load_gpr(ctx, rc));

            break;

        case 0x26:

            /* CMOVNE */

            tcg_gen_movcond_i64(TCG_COND_NE, vc, va, load_zero(ctx),

                                vb, load_gpr(ctx, rc));

            break;

        case 0x28:

            /* ORNOT */

            tcg_gen_orc_i64(vc, va, vb);

            break;

        case 0x40:

            /* XOR */

            tcg_gen_xor_i64(vc, va, vb);

            break;

        case 0x44:

            /* CMOVLT */

            tcg_gen_movcond_i64(TCG_COND_LT, vc, va, load_zero(ctx),

                                vb, load_gpr(ctx, rc));

            break;

        case 0x46:

            /* CMOVGE */

            tcg_gen_movcond_i64(TCG_COND_GE, vc, va, load_zero(ctx),

                                vb, load_gpr(ctx, rc));

            break;

        case 0x48:

            /* EQV */

            tcg_gen_eqv_i64(vc, va, vb);

            break;

        case 0x61:

            /* AMASK */

            REQUIRE_REG_31(ra);

            {

                uint64_t amask = ctx->tb->flags >> TB_FLAGS_AMASK_SHIFT;

                tcg_gen_andi_i64(vc, vb, ~amask);

            }

            break;

        case 0x64:

            /* CMOVLE */

            tcg_gen_movcond_i64(TCG_COND_LE, vc, va, load_zero(ctx),

                                vb, load_gpr(ctx, rc));

            break;

        case 0x66:

            /* CMOVGT */

            tcg_gen_movcond_i64(TCG_COND_GT, vc, va, load_zero(ctx),

                                vb, load_gpr(ctx, rc));

            break;

        case 0x6C:

            /* IMPLVER */

            REQUIRE_REG_31(ra);

            tcg_gen_movi_i64(vc, ctx->implver);

            break;

        default:

            goto invalid_opc;

        }

        break;



    case 0x12:

        vc = dest_gpr(ctx, rc);

        va = load_gpr(ctx, ra);

        switch (fn7) {

        case 0x02:

            /* MSKBL */

            gen_msk_l(ctx, vc, va, rb, islit, lit, 0x01);

            break;

        case 0x06:

            /* EXTBL */

            gen_ext_l(ctx, vc, va, rb, islit, lit, 0x01);

            break;

        case 0x0B:

            /* INSBL */

            gen_ins_l(ctx, vc, va, rb, islit, lit, 0x01);

            break;

        case 0x12:

            /* MSKWL */

            gen_msk_l(ctx, vc, va, rb, islit, lit, 0x03);

            break;

        case 0x16:

            /* EXTWL */

            gen_ext_l(ctx, vc, va, rb, islit, lit, 0x03);

            break;

        case 0x1B:

            /* INSWL */

            gen_ins_l(ctx, vc, va, rb, islit, lit, 0x03);

            break;

        case 0x22:

            /* MSKLL */

            gen_msk_l(ctx, vc, va, rb, islit, lit, 0x0f);

            break;

        case 0x26:

            /* EXTLL */

            gen_ext_l(ctx, vc, va, rb, islit, lit, 0x0f);

            break;

        case 0x2B:

            /* INSLL */

            gen_ins_l(ctx, vc, va, rb, islit, lit, 0x0f);

            break;

        case 0x30:

            /* ZAP */

            if (islit) {

                gen_zapnoti(vc, va, ~lit);

            } else {

                gen_helper_zap(vc, va, load_gpr(ctx, rb));

            }

            break;

        case 0x31:

            /* ZAPNOT */

            if (islit) {

                gen_zapnoti(vc, va, lit);

            } else {

                gen_helper_zapnot(vc, va, load_gpr(ctx, rb));

            }

            break;

        case 0x32:

            /* MSKQL */

            gen_msk_l(ctx, vc, va, rb, islit, lit, 0xff);

            break;

        case 0x34:

            /* SRL */

            if (islit) {

                tcg_gen_shri_i64(vc, va, lit & 0x3f);

            } else {

                tmp = tcg_temp_new();

                vb = load_gpr(ctx, rb);

                tcg_gen_andi_i64(tmp, vb, 0x3f);

                tcg_gen_shr_i64(vc, va, tmp);

                tcg_temp_free(tmp);

            }

            break;

        case 0x36:

            /* EXTQL */

            gen_ext_l(ctx, vc, va, rb, islit, lit, 0xff);

            break;

        case 0x39:

            /* SLL */

            if (islit) {

                tcg_gen_shli_i64(vc, va, lit & 0x3f);

            } else {

                tmp = tcg_temp_new();

                vb = load_gpr(ctx, rb);

                tcg_gen_andi_i64(tmp, vb, 0x3f);

                tcg_gen_shl_i64(vc, va, tmp);

                tcg_temp_free(tmp);

            }

            break;

        case 0x3B:

            /* INSQL */

            gen_ins_l(ctx, vc, va, rb, islit, lit, 0xff);

            break;

        case 0x3C:

            /* SRA */

            if (islit) {

                tcg_gen_sari_i64(vc, va, lit & 0x3f);

            } else {

                tmp = tcg_temp_new();

                vb = load_gpr(ctx, rb);

                tcg_gen_andi_i64(tmp, vb, 0x3f);

                tcg_gen_sar_i64(vc, va, tmp);

                tcg_temp_free(tmp);

            }

            break;

        case 0x52:

            /* MSKWH */

            gen_msk_h(ctx, vc, va, rb, islit, lit, 0x03);

            break;

        case 0x57:

            /* INSWH */

            gen_ins_h(ctx, vc, va, rb, islit, lit, 0x03);

            break;

        case 0x5A:

            /* EXTWH */

            gen_ext_h(ctx, vc, va, rb, islit, lit, 0x03);

            break;

        case 0x62:

            /* MSKLH */

            gen_msk_h(ctx, vc, va, rb, islit, lit, 0x0f);

            break;

        case 0x67:

            /* INSLH */

            gen_ins_h(ctx, vc, va, rb, islit, lit, 0x0f);

            break;

        case 0x6A:

            /* EXTLH */

            gen_ext_h(ctx, vc, va, rb, islit, lit, 0x0f);

            break;

        case 0x72:

            /* MSKQH */

            gen_msk_h(ctx, vc, va, rb, islit, lit, 0xff);

            break;

        case 0x77:

            /* INSQH */

            gen_ins_h(ctx, vc, va, rb, islit, lit, 0xff);

            break;

        case 0x7A:

            /* EXTQH */

            gen_ext_h(ctx, vc, va, rb, islit, lit, 0xff);

            break;

        default:

            goto invalid_opc;

        }

        break;



    case 0x13:

        vc = dest_gpr(ctx, rc);

        vb = load_gpr_lit(ctx, rb, lit, islit);

        va = load_gpr(ctx, ra);

        switch (fn7) {

        case 0x00:

            /* MULL */

            tcg_gen_mul_i64(vc, va, vb);

            tcg_gen_ext32s_i64(vc, vc);

            break;

        case 0x20:

            /* MULQ */

            tcg_gen_mul_i64(vc, va, vb);

            break;

        case 0x30:

            /* UMULH */

            tmp = tcg_temp_new();

            tcg_gen_mulu2_i64(tmp, vc, va, vb);

            tcg_temp_free(tmp);

            break;

        case 0x40:

            /* MULL/V */

            gen_helper_mullv(vc, cpu_env, va, vb);

            break;

        case 0x60:

            /* MULQ/V */

            gen_helper_mulqv(vc, cpu_env, va, vb);

            break;

        default:

            goto invalid_opc;

        }

        break;



    case 0x14:

        REQUIRE_TB_FLAG(TB_FLAGS_AMASK_FIX);

        vc = dest_fpr(ctx, rc);

        switch (fpfn) { /* fn11 & 0x3F */

        case 0x04:

            /* ITOFS */

            REQUIRE_REG_31(rb);

            t32 = tcg_temp_new_i32();

            va = load_gpr(ctx, ra);

            tcg_gen_trunc_i64_i32(t32, va);

            gen_helper_memory_to_s(vc, t32);

            tcg_temp_free_i32(t32);

            break;

        case 0x0A:

            /* SQRTF */

            REQUIRE_REG_31(ra);

            vb = load_fpr(ctx, rb);

            gen_helper_sqrtf(vc, cpu_env, vb);

            break;

        case 0x0B:

            /* SQRTS */

            REQUIRE_REG_31(ra);

            gen_sqrts(ctx, rb, rc, fn11);

            break;

        case 0x14:

            /* ITOFF */

            REQUIRE_REG_31(rb);

            t32 = tcg_temp_new_i32();

            va = load_gpr(ctx, ra);

            tcg_gen_trunc_i64_i32(t32, va);

            gen_helper_memory_to_f(vc, t32);

            tcg_temp_free_i32(t32);

            break;

        case 0x24:

            /* ITOFT */

            REQUIRE_REG_31(rb);

            va = load_gpr(ctx, ra);

            tcg_gen_mov_i64(vc, va);

            break;

        case 0x2A:

            /* SQRTG */

            REQUIRE_REG_31(ra);

            vb = load_fpr(ctx, rb);

            gen_helper_sqrtg(vc, cpu_env, vb);

            break;

        case 0x02B:

            /* SQRTT */

            REQUIRE_REG_31(ra);

            gen_sqrtt(ctx, rb, rc, fn11);

            break;

        default:

            goto invalid_opc;

        }

        break;



    case 0x15:

        /* VAX floating point */

        /* XXX: rounding mode and trap are ignored (!) */

        vc = dest_fpr(ctx, rc);

        vb = load_fpr(ctx, rb);

        va = load_fpr(ctx, ra);

        switch (fpfn) { /* fn11 & 0x3F */

        case 0x00:

            /* ADDF */

            gen_helper_addf(vc, cpu_env, va, vb);

            break;

        case 0x01:

            /* SUBF */

            gen_helper_subf(vc, cpu_env, va, vb);

            break;

        case 0x02:

            /* MULF */

            gen_helper_mulf(vc, cpu_env, va, vb);

            break;

        case 0x03:

            /* DIVF */

            gen_helper_divf(vc, cpu_env, va, vb);

            break;

        case 0x1E:

            /* CVTDG -- TODO */

            REQUIRE_REG_31(ra);

            goto invalid_opc;

        case 0x20:

            /* ADDG */

            gen_helper_addg(vc, cpu_env, va, vb);

            break;

        case 0x21:

            /* SUBG */

            gen_helper_subg(vc, cpu_env, va, vb);

            break;

        case 0x22:

            /* MULG */

            gen_helper_mulg(vc, cpu_env, va, vb);

            break;

        case 0x23:

            /* DIVG */

            gen_helper_divg(vc, cpu_env, va, vb);

            break;

        case 0x25:

            /* CMPGEQ */

            gen_helper_cmpgeq(vc, cpu_env, va, vb);

            break;

        case 0x26:

            /* CMPGLT */

            gen_helper_cmpglt(vc, cpu_env, va, vb);

            break;

        case 0x27:

            /* CMPGLE */

            gen_helper_cmpgle(vc, cpu_env, va, vb);

            break;

        case 0x2C:

            /* CVTGF */

            REQUIRE_REG_31(ra);

            gen_helper_cvtgf(vc, cpu_env, vb);

            break;

        case 0x2D:

            /* CVTGD -- TODO */

            REQUIRE_REG_31(ra);

            goto invalid_opc;

        case 0x2F:

            /* CVTGQ */

            REQUIRE_REG_31(ra);

            gen_helper_cvtgq(vc, cpu_env, vb);

            break;

        case 0x3C:

            /* CVTQF */

            REQUIRE_REG_31(ra);

            gen_helper_cvtqf(vc, cpu_env, vb);

            break;

        case 0x3E:

            /* CVTQG */

            REQUIRE_REG_31(ra);

            gen_helper_cvtqg(vc, cpu_env, vb);

            break;

        default:

            goto invalid_opc;

        }

        break;



    case 0x16:

        /* IEEE floating-point */

        switch (fpfn) { /* fn11 & 0x3F */

        case 0x00:

            /* ADDS */

            gen_adds(ctx, ra, rb, rc, fn11);

            break;

        case 0x01:

            /* SUBS */

            gen_subs(ctx, ra, rb, rc, fn11);

            break;

        case 0x02:

            /* MULS */

            gen_muls(ctx, ra, rb, rc, fn11);

            break;

        case 0x03:

            /* DIVS */

            gen_divs(ctx, ra, rb, rc, fn11);

            break;

        case 0x20:

            /* ADDT */

            gen_addt(ctx, ra, rb, rc, fn11);

            break;

        case 0x21:

            /* SUBT */

            gen_subt(ctx, ra, rb, rc, fn11);

            break;

        case 0x22:

            /* MULT */

            gen_mult(ctx, ra, rb, rc, fn11);

            break;

        case 0x23:

            /* DIVT */

            gen_divt(ctx, ra, rb, rc, fn11);

            break;

        case 0x24:

            /* CMPTUN */

            gen_cmptun(ctx, ra, rb, rc, fn11);

            break;

        case 0x25:

            /* CMPTEQ */

            gen_cmpteq(ctx, ra, rb, rc, fn11);

            break;

        case 0x26:

            /* CMPTLT */

            gen_cmptlt(ctx, ra, rb, rc, fn11);

            break;

        case 0x27:

            /* CMPTLE */

            gen_cmptle(ctx, ra, rb, rc, fn11);

            break;

        case 0x2C:

            REQUIRE_REG_31(ra);

            if (fn11 == 0x2AC || fn11 == 0x6AC) {

                /* CVTST */

                gen_cvtst(ctx, rb, rc, fn11);

            } else {

                /* CVTTS */

                gen_cvtts(ctx, rb, rc, fn11);

            }

            break;

        case 0x2F:

            /* CVTTQ */

            REQUIRE_REG_31(ra);

            gen_cvttq(ctx, rb, rc, fn11);

            break;

        case 0x3C:

            /* CVTQS */

            REQUIRE_REG_31(ra);

            gen_cvtqs(ctx, rb, rc, fn11);

            break;

        case 0x3E:

            /* CVTQT */

            REQUIRE_REG_31(ra);

            gen_cvtqt(ctx, rb, rc, fn11);

            break;

        default:

            goto invalid_opc;

        }

        break;



    case 0x17:

        switch (fn11) {

        case 0x010:

            /* CVTLQ */

            REQUIRE_REG_31(ra);

            vc = dest_fpr(ctx, rc);

            vb = load_fpr(ctx, rb);

            gen_cvtlq(vc, vb);

            break;

        case 0x020:

            /* CPYS */

            if (rc == 31) {

                /* Special case CPYS as FNOP.  */

            } else {

                vc = dest_fpr(ctx, rc);

                va = load_fpr(ctx, ra);

                if (ra == rb) {

                    /* Special case CPYS as FMOV.  */

                    tcg_gen_mov_i64(vc, va);

                } else {

                    vb = load_fpr(ctx, rb);

                    gen_cpy_mask(vc, va, vb, 0, 0x8000000000000000ULL);

                }

            }

            break;

        case 0x021:

            /* CPYSN */

            vc = dest_fpr(ctx, rc);

            vb = load_fpr(ctx, rb);

            va = load_fpr(ctx, ra);

            gen_cpy_mask(vc, va, vb, 1, 0x8000000000000000ULL);

            break;

        case 0x022:

            /* CPYSE */

            vc = dest_fpr(ctx, rc);

            vb = load_fpr(ctx, rb);

            va = load_fpr(ctx, ra);

            gen_cpy_mask(vc, va, vb, 0, 0xFFF0000000000000ULL);

            break;

        case 0x024:

            /* MT_FPCR */

            va = load_fpr(ctx, ra);

            gen_helper_store_fpcr(cpu_env, va);

            if (ctx->tb_rm == QUAL_RM_D) {

                /* Re-do the copy of the rounding mode to fp_status

                   the next time we use dynamic rounding.  */

                ctx->tb_rm = -1;

            }

            break;

        case 0x025:

            /* MF_FPCR */

            va = dest_fpr(ctx, ra);

            gen_helper_load_fpcr(va, cpu_env);

            break;

        case 0x02A:

            /* FCMOVEQ */

            gen_fcmov(ctx, TCG_COND_EQ, ra, rb, rc);

            break;

        case 0x02B:

            /* FCMOVNE */

            gen_fcmov(ctx, TCG_COND_NE, ra, rb, rc);

            break;

        case 0x02C:

            /* FCMOVLT */

            gen_fcmov(ctx, TCG_COND_LT, ra, rb, rc);

            break;

        case 0x02D:

            /* FCMOVGE */

            gen_fcmov(ctx, TCG_COND_GE, ra, rb, rc);

            break;

        case 0x02E:

            /* FCMOVLE */

            gen_fcmov(ctx, TCG_COND_LE, ra, rb, rc);

            break;

        case 0x02F:

            /* FCMOVGT */

            gen_fcmov(ctx, TCG_COND_GT, ra, rb, rc);

            break;

        case 0x030:

            /* CVTQL */

            REQUIRE_REG_31(ra);

            vc = dest_fpr(ctx, rc);

            vb = load_fpr(ctx, rb);

            gen_cvtql(vc, vb);

            break;

        case 0x130:

            /* CVTQL/V */

        case 0x530:

            /* CVTQL/SV */

            REQUIRE_REG_31(ra);

            /* ??? I'm pretty sure there's nothing that /sv needs to do that

               /v doesn't do.  The only thing I can think is that /sv is a

               valid instruction merely for completeness in the ISA.  */

            vc = dest_fpr(ctx, rc);

            vb = load_fpr(ctx, rb);

            gen_helper_cvtql_v_input(cpu_env, vb);

            gen_cvtql(vc, vb);

            break;

        default:

            goto invalid_opc;

        }

        break;



    case 0x18:

        switch ((uint16_t)disp16) {

        case 0x0000:

            /* TRAPB */

            /* No-op.  */

            break;

        case 0x0400:

            /* EXCB */

            /* No-op.  */

            break;

        case 0x4000:

            /* MB */

            /* No-op */

            break;

        case 0x4400:

            /* WMB */

            /* No-op */

            break;

        case 0x8000:

            /* FETCH */

            /* No-op */

            break;

        case 0xA000:

            /* FETCH_M */

            /* No-op */

            break;

        case 0xC000:

            /* RPCC */

            va = dest_gpr(ctx, ra);

            if (ctx->tb->cflags & CF_USE_ICOUNT) {

                gen_io_start();

                gen_helper_load_pcc(va, cpu_env);

                gen_io_end();

                ret = EXIT_PC_STALE;

            } else {

                gen_helper_load_pcc(va, cpu_env);

            }

            break;

        case 0xE000:

            /* RC */

            gen_rx(ra, 0);

            break;

        case 0xE800:

            /* ECB */

            break;

        case 0xF000:

            /* RS */

            gen_rx(ra, 1);

            break;

        case 0xF800:

            /* WH64 */

            /* No-op */

            break;

        default:

            goto invalid_opc;

        }

        break;



    case 0x19:

        /* HW_MFPR (PALcode) */

#ifndef CONFIG_USER_ONLY

        REQUIRE_TB_FLAG(TB_FLAGS_PAL_MODE);

        va = dest_gpr(ctx, ra);

        ret = gen_mfpr(ctx, va, insn & 0xffff);

        break;

#else

        goto invalid_opc;

#endif



    case 0x1A:

        /* JMP, JSR, RET, JSR_COROUTINE.  These only differ by the branch

           prediction stack action, which of course we don't implement.  */

        vb = load_gpr(ctx, rb);

        tcg_gen_andi_i64(cpu_pc, vb, ~3);

        if (ra != 31) {

            tcg_gen_movi_i64(cpu_ir[ra], ctx->pc);

        }

        ret = EXIT_PC_UPDATED;

        break;



    case 0x1B:

        /* HW_LD (PALcode) */

#ifndef CONFIG_USER_ONLY

        REQUIRE_TB_FLAG(TB_FLAGS_PAL_MODE);

        {

            TCGv addr = tcg_temp_new();

            vb = load_gpr(ctx, rb);

            va = dest_gpr(ctx, ra);



            tcg_gen_addi_i64(addr, vb, disp12);

            switch ((insn >> 12) & 0xF) {

            case 0x0:

                /* Longword physical access (hw_ldl/p) */

                gen_helper_ldl_phys(va, cpu_env, addr);

                break;

            case 0x1:

                /* Quadword physical access (hw_ldq/p) */

                gen_helper_ldq_phys(va, cpu_env, addr);

                break;

            case 0x2:

                /* Longword physical access with lock (hw_ldl_l/p) */

                gen_helper_ldl_l_phys(va, cpu_env, addr);

                break;

            case 0x3:

                /* Quadword physical access with lock (hw_ldq_l/p) */

                gen_helper_ldq_l_phys(va, cpu_env, addr);

                break;

            case 0x4:

                /* Longword virtual PTE fetch (hw_ldl/v) */

                goto invalid_opc;

            case 0x5:

                /* Quadword virtual PTE fetch (hw_ldq/v) */

                goto invalid_opc;

                break;

            case 0x6:

                /* Incpu_ir[ra]id */

                goto invalid_opc;

            case 0x7:

                /* Incpu_ir[ra]id */

                goto invalid_opc;

            case 0x8:

                /* Longword virtual access (hw_ldl) */

                goto invalid_opc;

            case 0x9:

                /* Quadword virtual access (hw_ldq) */

                goto invalid_opc;

            case 0xA:

                /* Longword virtual access with protection check (hw_ldl/w) */

                tcg_gen_qemu_ld_i64(va, addr, MMU_KERNEL_IDX, MO_LESL);

                break;

            case 0xB:

                /* Quadword virtual access with protection check (hw_ldq/w) */

                tcg_gen_qemu_ld_i64(va, addr, MMU_KERNEL_IDX, MO_LEQ);

                break;

            case 0xC:

                /* Longword virtual access with alt access mode (hw_ldl/a)*/

                goto invalid_opc;

            case 0xD:

                /* Quadword virtual access with alt access mode (hw_ldq/a) */

                goto invalid_opc;

            case 0xE:

                /* Longword virtual access with alternate access mode and

                   protection checks (hw_ldl/wa) */

                tcg_gen_qemu_ld_i64(va, addr, MMU_USER_IDX, MO_LESL);

                break;

            case 0xF:

                /* Quadword virtual access with alternate access mode and

                   protection checks (hw_ldq/wa) */

                tcg_gen_qemu_ld_i64(va, addr, MMU_USER_IDX, MO_LEQ);

                break;

            }

            tcg_temp_free(addr);

            break;

        }

#else

        goto invalid_opc;

#endif



    case 0x1C:

        vc = dest_gpr(ctx, rc);

        if (fn7 == 0x70) {

            /* FTOIT */

            REQUIRE_TB_FLAG(TB_FLAGS_AMASK_FIX);

            REQUIRE_REG_31(rb);

            va = load_fpr(ctx, ra);

            tcg_gen_mov_i64(vc, va);

            break;

        } else if (fn7 == 0x78) {

            /* FTOIS */

            REQUIRE_TB_FLAG(TB_FLAGS_AMASK_FIX);

            REQUIRE_REG_31(rb);

            t32 = tcg_temp_new_i32();

            va = load_fpr(ctx, ra);

            gen_helper_s_to_memory(t32, va);

            tcg_gen_ext_i32_i64(vc, t32);

            tcg_temp_free_i32(t32);

            break;

        }



        vb = load_gpr_lit(ctx, rb, lit, islit);

        switch (fn7) {

        case 0x00:

            /* SEXTB */

            REQUIRE_TB_FLAG(TB_FLAGS_AMASK_BWX);

            REQUIRE_REG_31(ra);

            tcg_gen_ext8s_i64(vc, vb);

            break;

        case 0x01:

            /* SEXTW */

            REQUIRE_TB_FLAG(TB_FLAGS_AMASK_BWX);

            REQUIRE_REG_31(ra);

            tcg_gen_ext16s_i64(vc, vb);

            break;

        case 0x30:

            /* CTPOP */

            REQUIRE_TB_FLAG(TB_FLAGS_AMASK_CIX);

            REQUIRE_REG_31(ra);

            gen_helper_ctpop(vc, vb);

            break;

        case 0x31:

            /* PERR */

            REQUIRE_TB_FLAG(TB_FLAGS_AMASK_MVI);

            va = load_gpr(ctx, ra);

            gen_helper_perr(vc, va, vb);

            break;

        case 0x32:

            /* CTLZ */

            REQUIRE_TB_FLAG(TB_FLAGS_AMASK_CIX);

            REQUIRE_REG_31(ra);

            gen_helper_ctlz(vc, vb);

            break;

        case 0x33:

            /* CTTZ */

            REQUIRE_TB_FLAG(TB_FLAGS_AMASK_CIX);

            REQUIRE_REG_31(ra);

            gen_helper_cttz(vc, vb);

            break;

        case 0x34:

            /* UNPKBW */

            REQUIRE_TB_FLAG(TB_FLAGS_AMASK_MVI);

            REQUIRE_REG_31(ra);

            gen_helper_unpkbw(vc, vb);

            break;

        case 0x35:

            /* UNPKBL */

            REQUIRE_TB_FLAG(TB_FLAGS_AMASK_MVI);

            REQUIRE_REG_31(ra);

            gen_helper_unpkbl(vc, vb);

            break;

        case 0x36:

            /* PKWB */

            REQUIRE_TB_FLAG(TB_FLAGS_AMASK_MVI);

            REQUIRE_REG_31(ra);

            gen_helper_pkwb(vc, vb);

            break;

        case 0x37:

            /* PKLB */

            REQUIRE_TB_FLAG(TB_FLAGS_AMASK_MVI);

            REQUIRE_REG_31(ra);

            gen_helper_pklb(vc, vb);

            break;

        case 0x38:

            /* MINSB8 */

            REQUIRE_TB_FLAG(TB_FLAGS_AMASK_MVI);

            va = load_gpr(ctx, ra);

            gen_helper_minsb8(vc, va, vb);

            break;

        case 0x39:

            /* MINSW4 */

            REQUIRE_TB_FLAG(TB_FLAGS_AMASK_MVI);

            va = load_gpr(ctx, ra);

            gen_helper_minsw4(vc, va, vb);

            break;

        case 0x3A:

            /* MINUB8 */

            REQUIRE_TB_FLAG(TB_FLAGS_AMASK_MVI);

            va = load_gpr(ctx, ra);

            gen_helper_minub8(vc, va, vb);

            break;

        case 0x3B:

            /* MINUW4 */

            REQUIRE_TB_FLAG(TB_FLAGS_AMASK_MVI);

            va = load_gpr(ctx, ra);

            gen_helper_minuw4(vc, va, vb);

            break;

        case 0x3C:

            /* MAXUB8 */

            REQUIRE_TB_FLAG(TB_FLAGS_AMASK_MVI);

            va = load_gpr(ctx, ra);

            gen_helper_maxub8(vc, va, vb);

            break;

        case 0x3D:

            /* MAXUW4 */

            REQUIRE_TB_FLAG(TB_FLAGS_AMASK_MVI);

            va = load_gpr(ctx, ra);

            gen_helper_maxuw4(vc, va, vb);

            break;

        case 0x3E:

            /* MAXSB8 */

            REQUIRE_TB_FLAG(TB_FLAGS_AMASK_MVI);

            va = load_gpr(ctx, ra);

            gen_helper_maxsb8(vc, va, vb);

            break;

        case 0x3F:

            /* MAXSW4 */

            REQUIRE_TB_FLAG(TB_FLAGS_AMASK_MVI);

            va = load_gpr(ctx, ra);

            gen_helper_maxsw4(vc, va, vb);

            break;

        default:

            goto invalid_opc;

        }

        break;



    case 0x1D:

        /* HW_MTPR (PALcode) */

#ifndef CONFIG_USER_ONLY

        REQUIRE_TB_FLAG(TB_FLAGS_PAL_MODE);

        vb = load_gpr(ctx, rb);

        ret = gen_mtpr(ctx, vb, insn & 0xffff);

        break;

#else

        goto invalid_opc;

#endif



    case 0x1E:

        /* HW_RET (PALcode) */

#ifndef CONFIG_USER_ONLY

        REQUIRE_TB_FLAG(TB_FLAGS_PAL_MODE);

        if (rb == 31) {

            /* Pre-EV6 CPUs interpreted this as HW_REI, loading the return

               address from EXC_ADDR.  This turns out to be useful for our

               emulation PALcode, so continue to accept it.  */

            tmp = tcg_temp_new();

            tcg_gen_ld_i64(tmp, cpu_env, offsetof(CPUAlphaState, exc_addr));

            gen_helper_hw_ret(cpu_env, tmp);

            tcg_temp_free(tmp);

        } else {

            gen_helper_hw_ret(cpu_env, load_gpr(ctx, rb));

        }

        ret = EXIT_PC_UPDATED;

        break;

#else

        goto invalid_opc;

#endif



    case 0x1F:

        /* HW_ST (PALcode) */

#ifndef CONFIG_USER_ONLY

        REQUIRE_TB_FLAG(TB_FLAGS_PAL_MODE);

        {

            TCGv addr = tcg_temp_new();

            va = load_gpr(ctx, ra);

            vb = load_gpr(ctx, rb);



            tcg_gen_addi_i64(addr, vb, disp12);

            switch ((insn >> 12) & 0xF) {

            case 0x0:

                /* Longword physical access */

                gen_helper_stl_phys(cpu_env, addr, va);

                break;

            case 0x1:

                /* Quadword physical access */

                gen_helper_stq_phys(cpu_env, addr, va);

                break;

            case 0x2:

                /* Longword physical access with lock */

                gen_helper_stl_c_phys(dest_gpr(ctx, ra), cpu_env, addr, va);

                break;

            case 0x3:

                /* Quadword physical access with lock */

                gen_helper_stq_c_phys(dest_gpr(ctx, ra), cpu_env, addr, va);

                break;

            case 0x4:

                /* Longword virtual access */

                goto invalid_opc;

            case 0x5:

                /* Quadword virtual access */

                goto invalid_opc;

            case 0x6:

                /* Invalid */

                goto invalid_opc;

            case 0x7:

                /* Invalid */

                goto invalid_opc;

            case 0x8:

                /* Invalid */

                goto invalid_opc;

            case 0x9:

                /* Invalid */

                goto invalid_opc;

            case 0xA:

                /* Invalid */

                goto invalid_opc;

            case 0xB:

                /* Invalid */

                goto invalid_opc;

            case 0xC:

                /* Longword virtual access with alternate access mode */

                goto invalid_opc;

            case 0xD:

                /* Quadword virtual access with alternate access mode */

                goto invalid_opc;

            case 0xE:

                /* Invalid */

                goto invalid_opc;

            case 0xF:

                /* Invalid */

                goto invalid_opc;

            }

            tcg_temp_free(addr);

            break;

        }

#else

        goto invalid_opc;

#endif

    case 0x20:

        /* LDF */

        gen_load_mem(ctx, &gen_qemu_ldf, ra, rb, disp16, 1, 0);

        break;

    case 0x21:

        /* LDG */

        gen_load_mem(ctx, &gen_qemu_ldg, ra, rb, disp16, 1, 0);

        break;

    case 0x22:

        /* LDS */

        gen_load_mem(ctx, &gen_qemu_lds, ra, rb, disp16, 1, 0);

        break;

    case 0x23:

        /* LDT */

        gen_load_mem(ctx, &tcg_gen_qemu_ld64, ra, rb, disp16, 1, 0);

        break;

    case 0x24:

        /* STF */

        gen_store_mem(ctx, &gen_qemu_stf, ra, rb, disp16, 1, 0);

        break;

    case 0x25:

        /* STG */

        gen_store_mem(ctx, &gen_qemu_stg, ra, rb, disp16, 1, 0);

        break;

    case 0x26:

        /* STS */

        gen_store_mem(ctx, &gen_qemu_sts, ra, rb, disp16, 1, 0);

        break;

    case 0x27:

        /* STT */

        gen_store_mem(ctx, &tcg_gen_qemu_st64, ra, rb, disp16, 1, 0);

        break;

    case 0x28:

        /* LDL */

        gen_load_mem(ctx, &tcg_gen_qemu_ld32s, ra, rb, disp16, 0, 0);

        break;

    case 0x29:

        /* LDQ */

        gen_load_mem(ctx, &tcg_gen_qemu_ld64, ra, rb, disp16, 0, 0);

        break;

    case 0x2A:

        /* LDL_L */

        gen_load_mem(ctx, &gen_qemu_ldl_l, ra, rb, disp16, 0, 0);

        break;

    case 0x2B:

        /* LDQ_L */

        gen_load_mem(ctx, &gen_qemu_ldq_l, ra, rb, disp16, 0, 0);

        break;

    case 0x2C:

        /* STL */

        gen_store_mem(ctx, &tcg_gen_qemu_st32, ra, rb, disp16, 0, 0);

        break;

    case 0x2D:

        /* STQ */

        gen_store_mem(ctx, &tcg_gen_qemu_st64, ra, rb, disp16, 0, 0);

        break;

    case 0x2E:

        /* STL_C */

        ret = gen_store_conditional(ctx, ra, rb, disp16, 0);

        break;

    case 0x2F:

        /* STQ_C */

        ret = gen_store_conditional(ctx, ra, rb, disp16, 1);

        break;

    case 0x30:

        /* BR */

        ret = gen_bdirect(ctx, ra, disp21);

        break;

    case 0x31: /* FBEQ */

        ret = gen_fbcond(ctx, TCG_COND_EQ, ra, disp21);

        break;

    case 0x32: /* FBLT */

        ret = gen_fbcond(ctx, TCG_COND_LT, ra, disp21);

        break;

    case 0x33: /* FBLE */

        ret = gen_fbcond(ctx, TCG_COND_LE, ra, disp21);

        break;

    case 0x34:

        /* BSR */

        ret = gen_bdirect(ctx, ra, disp21);

        break;

    case 0x35: /* FBNE */

        ret = gen_fbcond(ctx, TCG_COND_NE, ra, disp21);

        break;

    case 0x36: /* FBGE */

        ret = gen_fbcond(ctx, TCG_COND_GE, ra, disp21);

        break;

    case 0x37: /* FBGT */

        ret = gen_fbcond(ctx, TCG_COND_GT, ra, disp21);

        break;

    case 0x38:

        /* BLBC */

        ret = gen_bcond(ctx, TCG_COND_EQ, ra, disp21, 1);

        break;

    case 0x39:

        /* BEQ */

        ret = gen_bcond(ctx, TCG_COND_EQ, ra, disp21, 0);

        break;

    case 0x3A:

        /* BLT */

        ret = gen_bcond(ctx, TCG_COND_LT, ra, disp21, 0);

        break;

    case 0x3B:

        /* BLE */

        ret = gen_bcond(ctx, TCG_COND_LE, ra, disp21, 0);

        break;

    case 0x3C:

        /* BLBS */

        ret = gen_bcond(ctx, TCG_COND_NE, ra, disp21, 1);

        break;

    case 0x3D:

        /* BNE */

        ret = gen_bcond(ctx, TCG_COND_NE, ra, disp21, 0);

        break;

    case 0x3E:

        /* BGE */

        ret = gen_bcond(ctx, TCG_COND_GE, ra, disp21, 0);

        break;

    case 0x3F:

        /* BGT */

        ret = gen_bcond(ctx, TCG_COND_GT, ra, disp21, 0);

        break;

    invalid_opc:

        ret = gen_invalid(ctx);

        break;

    }



    return ret;

}
