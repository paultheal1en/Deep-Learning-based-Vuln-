static void decode_opc(CPUMIPSState *env, DisasContext *ctx)

{

    int32_t offset;

    int rs, rt, rd, sa;

    uint32_t op, op1;

    int16_t imm;



    /* make sure instructions are on a word boundary */

    if (ctx->pc & 0x3) {

        env->CP0_BadVAddr = ctx->pc;

        generate_exception_err(ctx, EXCP_AdEL, EXCP_INST_NOTAVAIL);

        return;

    }



    /* Handle blikely not taken case */

    if ((ctx->hflags & MIPS_HFLAG_BMASK_BASE) == MIPS_HFLAG_BL) {

        TCGLabel *l1 = gen_new_label();



        tcg_gen_brcondi_tl(TCG_COND_NE, bcond, 0, l1);

        tcg_gen_movi_i32(hflags, ctx->hflags & ~MIPS_HFLAG_BMASK);

        gen_goto_tb(ctx, 1, ctx->pc + 4);

        gen_set_label(l1);

    }



    op = MASK_OP_MAJOR(ctx->opcode);

    rs = (ctx->opcode >> 21) & 0x1f;

    rt = (ctx->opcode >> 16) & 0x1f;

    rd = (ctx->opcode >> 11) & 0x1f;

    sa = (ctx->opcode >> 6) & 0x1f;

    imm = (int16_t)ctx->opcode;

    switch (op) {

    case OPC_SPECIAL:

        decode_opc_special(env, ctx);

        break;

    case OPC_SPECIAL2:

        decode_opc_special2_legacy(env, ctx);

        break;

    case OPC_SPECIAL3:

        decode_opc_special3(env, ctx);

        break;

    case OPC_REGIMM:

        op1 = MASK_REGIMM(ctx->opcode);

        switch (op1) {

        case OPC_BLTZL: /* REGIMM branches */

        case OPC_BGEZL:

        case OPC_BLTZALL:

        case OPC_BGEZALL:

            check_insn(ctx, ISA_MIPS2);

            check_insn_opc_removed(ctx, ISA_MIPS32R6);

            /* Fallthrough */

        case OPC_BLTZ:

        case OPC_BGEZ:

            gen_compute_branch(ctx, op1, 4, rs, -1, imm << 2, 4);

            break;

        case OPC_BLTZAL:

        case OPC_BGEZAL:

            if (ctx->insn_flags & ISA_MIPS32R6) {

                if (rs == 0) {

                    /* OPC_NAL, OPC_BAL */

                    gen_compute_branch(ctx, op1, 4, 0, -1, imm << 2, 4);

                } else {

                    generate_exception_end(ctx, EXCP_RI);

                }

            } else {

                gen_compute_branch(ctx, op1, 4, rs, -1, imm << 2, 4);

            }

            break;

        case OPC_TGEI ... OPC_TEQI: /* REGIMM traps */

        case OPC_TNEI:

            check_insn(ctx, ISA_MIPS2);

            check_insn_opc_removed(ctx, ISA_MIPS32R6);

            gen_trap(ctx, op1, rs, -1, imm);

            break;

        case OPC_SIGRIE:

            check_insn(ctx, ISA_MIPS32R6);

            generate_exception_end(ctx, EXCP_RI);

            break;

        case OPC_SYNCI:

            check_insn(ctx, ISA_MIPS32R2);

            /* Break the TB to be able to sync copied instructions

               immediately */

            ctx->bstate = BS_STOP;

            break;

        case OPC_BPOSGE32:    /* MIPS DSP branch */

#if defined(TARGET_MIPS64)

        case OPC_BPOSGE64:

#endif

            check_dsp(ctx);

            gen_compute_branch(ctx, op1, 4, -1, -2, (int32_t)imm << 2, 4);

            break;

#if defined(TARGET_MIPS64)

        case OPC_DAHI:

            check_insn(ctx, ISA_MIPS32R6);

            check_mips_64(ctx);

            if (rs != 0) {

                tcg_gen_addi_tl(cpu_gpr[rs], cpu_gpr[rs], (int64_t)imm << 32);

            }

            break;

        case OPC_DATI:

            check_insn(ctx, ISA_MIPS32R6);

            check_mips_64(ctx);

            if (rs != 0) {

                tcg_gen_addi_tl(cpu_gpr[rs], cpu_gpr[rs], (int64_t)imm << 48);

            }

            break;

#endif

        default:            /* Invalid */

            MIPS_INVAL("regimm");

            generate_exception_end(ctx, EXCP_RI);

            break;

        }

        break;

    case OPC_CP0:

        check_cp0_enabled(ctx);

        op1 = MASK_CP0(ctx->opcode);

        switch (op1) {

        case OPC_MFC0:

        case OPC_MTC0:

        case OPC_MFTR:

        case OPC_MTTR:

        case OPC_MFHC0:

        case OPC_MTHC0:

#if defined(TARGET_MIPS64)

        case OPC_DMFC0:

        case OPC_DMTC0:

#endif

#ifndef CONFIG_USER_ONLY

            gen_cp0(env, ctx, op1, rt, rd);

#endif /* !CONFIG_USER_ONLY */

            break;

        case OPC_C0_FIRST ... OPC_C0_LAST:

#ifndef CONFIG_USER_ONLY

            gen_cp0(env, ctx, MASK_C0(ctx->opcode), rt, rd);

#endif /* !CONFIG_USER_ONLY */

            break;

        case OPC_MFMC0:

#ifndef CONFIG_USER_ONLY

            {

                uint32_t op2;

                TCGv t0 = tcg_temp_new();



                op2 = MASK_MFMC0(ctx->opcode);

                switch (op2) {

                case OPC_DMT:

                    check_insn(ctx, ASE_MT);

                    gen_helper_dmt(t0);

                    gen_store_gpr(t0, rt);

                    break;

                case OPC_EMT:

                    check_insn(ctx, ASE_MT);

                    gen_helper_emt(t0);

                    gen_store_gpr(t0, rt);

                    break;

                case OPC_DVPE:

                    check_insn(ctx, ASE_MT);

                    gen_helper_dvpe(t0, cpu_env);

                    gen_store_gpr(t0, rt);

                    break;

                case OPC_EVPE:

                    check_insn(ctx, ASE_MT);

                    gen_helper_evpe(t0, cpu_env);

                    gen_store_gpr(t0, rt);

                    break;

                case OPC_DVP:

                    check_insn(ctx, ISA_MIPS32R6);

                    if (ctx->vp) {

                        gen_helper_dvp(t0, cpu_env);

                        gen_store_gpr(t0, rt);

                    }

                    break;

                case OPC_EVP:

                    check_insn(ctx, ISA_MIPS32R6);

                    if (ctx->vp) {

                        gen_helper_evp(t0, cpu_env);

                        gen_store_gpr(t0, rt);

                    }

                    break;

                case OPC_DI:

                    check_insn(ctx, ISA_MIPS32R2);

                    save_cpu_state(ctx, 1);

                    gen_helper_di(t0, cpu_env);

                    gen_store_gpr(t0, rt);

                    /* Stop translation as we may have switched

                       the execution mode.  */

                    ctx->bstate = BS_STOP;

                    break;

                case OPC_EI:

                    check_insn(ctx, ISA_MIPS32R2);

                    save_cpu_state(ctx, 1);

                    gen_helper_ei(t0, cpu_env);

                    gen_store_gpr(t0, rt);

                    /* Stop translation as we may have switched

                       the execution mode.  */

                    ctx->bstate = BS_STOP;

                    break;

                default:            /* Invalid */

                    MIPS_INVAL("mfmc0");

                    generate_exception_end(ctx, EXCP_RI);

                    break;

                }

                tcg_temp_free(t0);

            }

#endif /* !CONFIG_USER_ONLY */

            break;

        case OPC_RDPGPR:

            check_insn(ctx, ISA_MIPS32R2);

            gen_load_srsgpr(rt, rd);

            break;

        case OPC_WRPGPR:

            check_insn(ctx, ISA_MIPS32R2);

            gen_store_srsgpr(rt, rd);

            break;

        default:

            MIPS_INVAL("cp0");

            generate_exception_end(ctx, EXCP_RI);

            break;

        }

        break;

    case OPC_BOVC: /* OPC_BEQZALC, OPC_BEQC, OPC_ADDI */

        if (ctx->insn_flags & ISA_MIPS32R6) {

            /* OPC_BOVC, OPC_BEQZALC, OPC_BEQC */

            gen_compute_compact_branch(ctx, op, rs, rt, imm << 2);

        } else {

            /* OPC_ADDI */

            /* Arithmetic with immediate opcode */

            gen_arith_imm(ctx, op, rt, rs, imm);

        }

        break;

    case OPC_ADDIU:

         gen_arith_imm(ctx, op, rt, rs, imm);

         break;

    case OPC_SLTI: /* Set on less than with immediate opcode */

    case OPC_SLTIU:

         gen_slt_imm(ctx, op, rt, rs, imm);

         break;

    case OPC_ANDI: /* Arithmetic with immediate opcode */

    case OPC_LUI: /* OPC_AUI */

    case OPC_ORI:

    case OPC_XORI:

         gen_logic_imm(ctx, op, rt, rs, imm);

         break;

    case OPC_J ... OPC_JAL: /* Jump */

         offset = (int32_t)(ctx->opcode & 0x3FFFFFF) << 2;

         gen_compute_branch(ctx, op, 4, rs, rt, offset, 4);

         break;

    /* Branch */

    case OPC_BLEZC: /* OPC_BGEZC, OPC_BGEC, OPC_BLEZL */

        if (ctx->insn_flags & ISA_MIPS32R6) {

            if (rt == 0) {

                generate_exception_end(ctx, EXCP_RI);

                break;

            }

            /* OPC_BLEZC, OPC_BGEZC, OPC_BGEC */

            gen_compute_compact_branch(ctx, op, rs, rt, imm << 2);

        } else {

            /* OPC_BLEZL */

            gen_compute_branch(ctx, op, 4, rs, rt, imm << 2, 4);

        }

        break;

    case OPC_BGTZC: /* OPC_BLTZC, OPC_BLTC, OPC_BGTZL */

        if (ctx->insn_flags & ISA_MIPS32R6) {

            if (rt == 0) {

                generate_exception_end(ctx, EXCP_RI);

                break;

            }

            /* OPC_BGTZC, OPC_BLTZC, OPC_BLTC */

            gen_compute_compact_branch(ctx, op, rs, rt, imm << 2);

        } else {

            /* OPC_BGTZL */

            gen_compute_branch(ctx, op, 4, rs, rt, imm << 2, 4);

        }

        break;

    case OPC_BLEZALC: /* OPC_BGEZALC, OPC_BGEUC, OPC_BLEZ */

        if (rt == 0) {

            /* OPC_BLEZ */

            gen_compute_branch(ctx, op, 4, rs, rt, imm << 2, 4);

        } else {

            check_insn(ctx, ISA_MIPS32R6);

            /* OPC_BLEZALC, OPC_BGEZALC, OPC_BGEUC */

            gen_compute_compact_branch(ctx, op, rs, rt, imm << 2);

        }

        break;

    case OPC_BGTZALC: /* OPC_BLTZALC, OPC_BLTUC, OPC_BGTZ */

        if (rt == 0) {

            /* OPC_BGTZ */

            gen_compute_branch(ctx, op, 4, rs, rt, imm << 2, 4);

        } else {

            check_insn(ctx, ISA_MIPS32R6);

            /* OPC_BGTZALC, OPC_BLTZALC, OPC_BLTUC */

            gen_compute_compact_branch(ctx, op, rs, rt, imm << 2);

        }

        break;

    case OPC_BEQL:

    case OPC_BNEL:

        check_insn(ctx, ISA_MIPS2);

         check_insn_opc_removed(ctx, ISA_MIPS32R6);

        /* Fallthrough */

    case OPC_BEQ:

    case OPC_BNE:

         gen_compute_branch(ctx, op, 4, rs, rt, imm << 2, 4);

         break;

    case OPC_LL: /* Load and stores */

        check_insn(ctx, ISA_MIPS2);

        /* Fallthrough */

    case OPC_LWL:

    case OPC_LWR:

        check_insn_opc_removed(ctx, ISA_MIPS32R6);

         /* Fallthrough */

    case OPC_LB ... OPC_LH:

    case OPC_LW ... OPC_LHU:

         gen_ld(ctx, op, rt, rs, imm);

         break;

    case OPC_SWL:

    case OPC_SWR:

        check_insn_opc_removed(ctx, ISA_MIPS32R6);

        /* fall through */

    case OPC_SB ... OPC_SH:

    case OPC_SW:

         gen_st(ctx, op, rt, rs, imm);

         break;

    case OPC_SC:

        check_insn(ctx, ISA_MIPS2);

         check_insn_opc_removed(ctx, ISA_MIPS32R6);

         gen_st_cond(ctx, op, rt, rs, imm);

         break;

    case OPC_CACHE:

        check_insn_opc_removed(ctx, ISA_MIPS32R6);

        check_cp0_enabled(ctx);

        check_insn(ctx, ISA_MIPS3 | ISA_MIPS32);

        if (ctx->hflags & MIPS_HFLAG_ITC_CACHE) {

            gen_cache_operation(ctx, rt, rs, imm);

        }

        /* Treat as NOP. */

        break;

    case OPC_PREF:

        check_insn_opc_removed(ctx, ISA_MIPS32R6);

        check_insn(ctx, ISA_MIPS4 | ISA_MIPS32);

        /* Treat as NOP. */

        break;



    /* Floating point (COP1). */

    case OPC_LWC1:

    case OPC_LDC1:

    case OPC_SWC1:

    case OPC_SDC1:

        gen_cop1_ldst(ctx, op, rt, rs, imm);

        break;



    case OPC_CP1:

        op1 = MASK_CP1(ctx->opcode);



        switch (op1) {

        case OPC_MFHC1:

        case OPC_MTHC1:

            check_cp1_enabled(ctx);

            check_insn(ctx, ISA_MIPS32R2);

        case OPC_MFC1:

        case OPC_CFC1:

        case OPC_MTC1:

        case OPC_CTC1:

            check_cp1_enabled(ctx);

            gen_cp1(ctx, op1, rt, rd);

            break;

#if defined(TARGET_MIPS64)

        case OPC_DMFC1:

        case OPC_DMTC1:

            check_cp1_enabled(ctx);

            check_insn(ctx, ISA_MIPS3);

            check_mips_64(ctx);

            gen_cp1(ctx, op1, rt, rd);

            break;

#endif

        case OPC_BC1EQZ: /* OPC_BC1ANY2 */

            check_cp1_enabled(ctx);

            if (ctx->insn_flags & ISA_MIPS32R6) {

                /* OPC_BC1EQZ */

                gen_compute_branch1_r6(ctx, MASK_CP1(ctx->opcode),

                                       rt, imm << 2, 4);

            } else {

                /* OPC_BC1ANY2 */

                check_cop1x(ctx);

                check_insn(ctx, ASE_MIPS3D);

                gen_compute_branch1(ctx, MASK_BC1(ctx->opcode),

                                    (rt >> 2) & 0x7, imm << 2);

            }

            break;

        case OPC_BC1NEZ:

            check_cp1_enabled(ctx);

            check_insn(ctx, ISA_MIPS32R6);

            gen_compute_branch1_r6(ctx, MASK_CP1(ctx->opcode),

                                   rt, imm << 2, 4);

            break;

        case OPC_BC1ANY4:

            check_cp1_enabled(ctx);

            check_insn_opc_removed(ctx, ISA_MIPS32R6);

            check_cop1x(ctx);

            check_insn(ctx, ASE_MIPS3D);

            /* fall through */

        case OPC_BC1:

            check_cp1_enabled(ctx);

            check_insn_opc_removed(ctx, ISA_MIPS32R6);

            gen_compute_branch1(ctx, MASK_BC1(ctx->opcode),

                                (rt >> 2) & 0x7, imm << 2);

            break;

        case OPC_PS_FMT:

            check_ps(ctx);

            /* fall through */

        case OPC_S_FMT:

        case OPC_D_FMT:

            check_cp1_enabled(ctx);

            gen_farith(ctx, ctx->opcode & FOP(0x3f, 0x1f), rt, rd, sa,

                       (imm >> 8) & 0x7);

            break;

        case OPC_W_FMT:

        case OPC_L_FMT:

        {

            int r6_op = ctx->opcode & FOP(0x3f, 0x1f);

            check_cp1_enabled(ctx);

            if (ctx->insn_flags & ISA_MIPS32R6) {

                switch (r6_op) {

                case R6_OPC_CMP_AF_S:

                case R6_OPC_CMP_UN_S:

                case R6_OPC_CMP_EQ_S:

                case R6_OPC_CMP_UEQ_S:

                case R6_OPC_CMP_LT_S:

                case R6_OPC_CMP_ULT_S:

                case R6_OPC_CMP_LE_S:

                case R6_OPC_CMP_ULE_S:

                case R6_OPC_CMP_SAF_S:

                case R6_OPC_CMP_SUN_S:

                case R6_OPC_CMP_SEQ_S:

                case R6_OPC_CMP_SEUQ_S:

                case R6_OPC_CMP_SLT_S:

                case R6_OPC_CMP_SULT_S:

                case R6_OPC_CMP_SLE_S:

                case R6_OPC_CMP_SULE_S:

                case R6_OPC_CMP_OR_S:

                case R6_OPC_CMP_UNE_S:

                case R6_OPC_CMP_NE_S:

                case R6_OPC_CMP_SOR_S:

                case R6_OPC_CMP_SUNE_S:

                case R6_OPC_CMP_SNE_S:

                    gen_r6_cmp_s(ctx, ctx->opcode & 0x1f, rt, rd, sa);

                    break;

                case R6_OPC_CMP_AF_D:

                case R6_OPC_CMP_UN_D:

                case R6_OPC_CMP_EQ_D:

                case R6_OPC_CMP_UEQ_D:

                case R6_OPC_CMP_LT_D:

                case R6_OPC_CMP_ULT_D:

                case R6_OPC_CMP_LE_D:

                case R6_OPC_CMP_ULE_D:

                case R6_OPC_CMP_SAF_D:

                case R6_OPC_CMP_SUN_D:

                case R6_OPC_CMP_SEQ_D:

                case R6_OPC_CMP_SEUQ_D:

                case R6_OPC_CMP_SLT_D:

                case R6_OPC_CMP_SULT_D:

                case R6_OPC_CMP_SLE_D:

                case R6_OPC_CMP_SULE_D:

                case R6_OPC_CMP_OR_D:

                case R6_OPC_CMP_UNE_D:

                case R6_OPC_CMP_NE_D:

                case R6_OPC_CMP_SOR_D:

                case R6_OPC_CMP_SUNE_D:

                case R6_OPC_CMP_SNE_D:

                    gen_r6_cmp_d(ctx, ctx->opcode & 0x1f, rt, rd, sa);

                    break;

                default:

                    gen_farith(ctx, ctx->opcode & FOP(0x3f, 0x1f),

                               rt, rd, sa, (imm >> 8) & 0x7);



                    break;

                }

            } else {

                gen_farith(ctx, ctx->opcode & FOP(0x3f, 0x1f), rt, rd, sa,

                           (imm >> 8) & 0x7);

            }

            break;

        }

        case OPC_BZ_V:

        case OPC_BNZ_V:

        case OPC_BZ_B:

        case OPC_BZ_H:

        case OPC_BZ_W:

        case OPC_BZ_D:

        case OPC_BNZ_B:

        case OPC_BNZ_H:

        case OPC_BNZ_W:

        case OPC_BNZ_D:

            check_insn(ctx, ASE_MSA);

            gen_msa_branch(env, ctx, op1);

            break;

        default:

            MIPS_INVAL("cp1");

            generate_exception_end(ctx, EXCP_RI);

            break;

        }

        break;



    /* Compact branches [R6] and COP2 [non-R6] */

    case OPC_BC: /* OPC_LWC2 */

    case OPC_BALC: /* OPC_SWC2 */

        if (ctx->insn_flags & ISA_MIPS32R6) {

            /* OPC_BC, OPC_BALC */

            gen_compute_compact_branch(ctx, op, 0, 0,

                                       sextract32(ctx->opcode << 2, 0, 28));

        } else {

            /* OPC_LWC2, OPC_SWC2 */

            /* COP2: Not implemented. */

            generate_exception_err(ctx, EXCP_CpU, 2);

        }

        break;

    case OPC_BEQZC: /* OPC_JIC, OPC_LDC2 */

    case OPC_BNEZC: /* OPC_JIALC, OPC_SDC2 */

        if (ctx->insn_flags & ISA_MIPS32R6) {

            if (rs != 0) {

                /* OPC_BEQZC, OPC_BNEZC */

                gen_compute_compact_branch(ctx, op, rs, 0,

                                           sextract32(ctx->opcode << 2, 0, 23));

            } else {

                /* OPC_JIC, OPC_JIALC */

                gen_compute_compact_branch(ctx, op, 0, rt, imm);

            }

        } else {

            /* OPC_LWC2, OPC_SWC2 */

            /* COP2: Not implemented. */

            generate_exception_err(ctx, EXCP_CpU, 2);

        }

        break;

    case OPC_CP2:

        check_insn(ctx, INSN_LOONGSON2F);

        /* Note that these instructions use different fields.  */

        gen_loongson_multimedia(ctx, sa, rd, rt);

        break;



    case OPC_CP3:

        check_insn_opc_removed(ctx, ISA_MIPS32R6);

        if (ctx->CP0_Config1 & (1 << CP0C1_FP)) {

            check_cp1_enabled(ctx);

            op1 = MASK_CP3(ctx->opcode);

            switch (op1) {

            case OPC_LUXC1:

            case OPC_SUXC1:

                check_insn(ctx, ISA_MIPS5 | ISA_MIPS32R2);

                /* Fallthrough */

            case OPC_LWXC1:

            case OPC_LDXC1:

            case OPC_SWXC1:

            case OPC_SDXC1:

                check_insn(ctx, ISA_MIPS4 | ISA_MIPS32R2);

                gen_flt3_ldst(ctx, op1, sa, rd, rs, rt);

                break;

            case OPC_PREFX:

                check_insn(ctx, ISA_MIPS4 | ISA_MIPS32R2);

                /* Treat as NOP. */

                break;

            case OPC_ALNV_PS:

                check_insn(ctx, ISA_MIPS5 | ISA_MIPS32R2);

                /* Fallthrough */

            case OPC_MADD_S:

            case OPC_MADD_D:

            case OPC_MADD_PS:

            case OPC_MSUB_S:

            case OPC_MSUB_D:

            case OPC_MSUB_PS:

            case OPC_NMADD_S:

            case OPC_NMADD_D:

            case OPC_NMADD_PS:

            case OPC_NMSUB_S:

            case OPC_NMSUB_D:

            case OPC_NMSUB_PS:

                check_insn(ctx, ISA_MIPS4 | ISA_MIPS32R2);

                gen_flt3_arith(ctx, op1, sa, rs, rd, rt);

                break;

            default:

                MIPS_INVAL("cp3");

                generate_exception_end(ctx, EXCP_RI);

                break;

            }

        } else {

            generate_exception_err(ctx, EXCP_CpU, 1);

        }

        break;



#if defined(TARGET_MIPS64)

    /* MIPS64 opcodes */

    case OPC_LDL ... OPC_LDR:

    case OPC_LLD:

        check_insn_opc_removed(ctx, ISA_MIPS32R6);

        /* fall through */

    case OPC_LWU:

    case OPC_LD:

        check_insn(ctx, ISA_MIPS3);

        check_mips_64(ctx);

        gen_ld(ctx, op, rt, rs, imm);

        break;

    case OPC_SDL ... OPC_SDR:

        check_insn_opc_removed(ctx, ISA_MIPS32R6);

        /* fall through */

    case OPC_SD:

        check_insn(ctx, ISA_MIPS3);

        check_mips_64(ctx);

        gen_st(ctx, op, rt, rs, imm);

        break;

    case OPC_SCD:

        check_insn_opc_removed(ctx, ISA_MIPS32R6);

        check_insn(ctx, ISA_MIPS3);

        check_mips_64(ctx);

        gen_st_cond(ctx, op, rt, rs, imm);

        break;

    case OPC_BNVC: /* OPC_BNEZALC, OPC_BNEC, OPC_DADDI */

        if (ctx->insn_flags & ISA_MIPS32R6) {

            /* OPC_BNVC, OPC_BNEZALC, OPC_BNEC */

            gen_compute_compact_branch(ctx, op, rs, rt, imm << 2);

        } else {

            /* OPC_DADDI */

            check_insn(ctx, ISA_MIPS3);

            check_mips_64(ctx);

            gen_arith_imm(ctx, op, rt, rs, imm);

        }

        break;

    case OPC_DADDIU:

        check_insn(ctx, ISA_MIPS3);

        check_mips_64(ctx);

        gen_arith_imm(ctx, op, rt, rs, imm);

        break;

#else

    case OPC_BNVC: /* OPC_BNEZALC, OPC_BNEC */

        if (ctx->insn_flags & ISA_MIPS32R6) {

            gen_compute_compact_branch(ctx, op, rs, rt, imm << 2);

        } else {

            MIPS_INVAL("major opcode");

            generate_exception_end(ctx, EXCP_RI);

        }

        break;

#endif

    case OPC_DAUI: /* OPC_JALX */

        if (ctx->insn_flags & ISA_MIPS32R6) {

#if defined(TARGET_MIPS64)

            /* OPC_DAUI */

            check_mips_64(ctx);

            if (rs == 0) {

                generate_exception(ctx, EXCP_RI);

            } else if (rt != 0) {

                TCGv t0 = tcg_temp_new();

                gen_load_gpr(t0, rs);

                tcg_gen_addi_tl(cpu_gpr[rt], t0, imm << 16);

                tcg_temp_free(t0);

            }

#else

            generate_exception_end(ctx, EXCP_RI);

            MIPS_INVAL("major opcode");

#endif

        } else {

            /* OPC_JALX */

            check_insn(ctx, ASE_MIPS16 | ASE_MICROMIPS);

            offset = (int32_t)(ctx->opcode & 0x3FFFFFF) << 2;

            gen_compute_branch(ctx, op, 4, rs, rt, offset, 4);

        }

        break;

    case OPC_MSA: /* OPC_MDMX */

        /* MDMX: Not implemented. */

        gen_msa(env, ctx);

        break;

    case OPC_PCREL:

        check_insn(ctx, ISA_MIPS32R6);

        gen_pcrel(ctx, ctx->opcode, ctx->pc, rs);

        break;

    default:            /* Invalid */

        MIPS_INVAL("major opcode");

        generate_exception_end(ctx, EXCP_RI);

        break;

    }

}
