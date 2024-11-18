static int decode_mips16_opc (CPUMIPSState *env, DisasContext *ctx,

                              int *is_branch)

{

    int rx, ry;

    int sa;

    int op, cnvt_op, op1, offset;

    int funct;

    int n_bytes;



    op = (ctx->opcode >> 11) & 0x1f;

    sa = (ctx->opcode >> 2) & 0x7;

    sa = sa == 0 ? 8 : sa;

    rx = xlat((ctx->opcode >> 8) & 0x7);

    cnvt_op = (ctx->opcode >> 5) & 0x7;

    ry = xlat((ctx->opcode >> 5) & 0x7);

    op1 = offset = ctx->opcode & 0x1f;



    n_bytes = 2;



    switch (op) {

    case M16_OPC_ADDIUSP:

        {

            int16_t imm = ((uint8_t) ctx->opcode) << 2;



            gen_arith_imm(ctx, OPC_ADDIU, rx, 29, imm);

        }

        break;

    case M16_OPC_ADDIUPC:

        gen_addiupc(ctx, rx, ((uint8_t) ctx->opcode) << 2, 0, 0);

        break;

    case M16_OPC_B:

        offset = (ctx->opcode & 0x7ff) << 1;

        offset = (int16_t)(offset << 4) >> 4;

        gen_compute_branch(ctx, OPC_BEQ, 2, 0, 0, offset);

        /* No delay slot, so just process as a normal instruction */

        break;

    case M16_OPC_JAL:

        offset = cpu_lduw_code(env, ctx->pc + 2);

        offset = (((ctx->opcode & 0x1f) << 21)

                  | ((ctx->opcode >> 5) & 0x1f) << 16

                  | offset) << 2;

        op = ((ctx->opcode >> 10) & 0x1) ? OPC_JALXS : OPC_JALS;

        gen_compute_branch(ctx, op, 4, rx, ry, offset);

        n_bytes = 4;

        *is_branch = 1;

        break;

    case M16_OPC_BEQZ:

        gen_compute_branch(ctx, OPC_BEQ, 2, rx, 0, ((int8_t)ctx->opcode) << 1);

        /* No delay slot, so just process as a normal instruction */

        break;

    case M16_OPC_BNEQZ:

        gen_compute_branch(ctx, OPC_BNE, 2, rx, 0, ((int8_t)ctx->opcode) << 1);

        /* No delay slot, so just process as a normal instruction */

        break;

    case M16_OPC_SHIFT:

        switch (ctx->opcode & 0x3) {

        case 0x0:

            gen_shift_imm(ctx, OPC_SLL, rx, ry, sa);

            break;

        case 0x1:

#if defined(TARGET_MIPS64)

            check_mips_64(ctx);

            gen_shift_imm(ctx, OPC_DSLL, rx, ry, sa);

#else

            generate_exception(ctx, EXCP_RI);

#endif

            break;

        case 0x2:

            gen_shift_imm(ctx, OPC_SRL, rx, ry, sa);

            break;

        case 0x3:

            gen_shift_imm(ctx, OPC_SRA, rx, ry, sa);

            break;

        }

        break;

#if defined(TARGET_MIPS64)

    case M16_OPC_LD:

        check_mips_64(ctx);

        gen_ld(ctx, OPC_LD, ry, rx, offset << 3);

        break;

#endif

    case M16_OPC_RRIA:

        {

            int16_t imm = (int8_t)((ctx->opcode & 0xf) << 4) >> 4;



            if ((ctx->opcode >> 4) & 1) {

#if defined(TARGET_MIPS64)

                check_mips_64(ctx);

                gen_arith_imm(ctx, OPC_DADDIU, ry, rx, imm);

#else

                generate_exception(ctx, EXCP_RI);

#endif

            } else {

                gen_arith_imm(ctx, OPC_ADDIU, ry, rx, imm);

            }

        }

        break;

    case M16_OPC_ADDIU8:

        {

            int16_t imm = (int8_t) ctx->opcode;



            gen_arith_imm(ctx, OPC_ADDIU, rx, rx, imm);

        }

        break;

    case M16_OPC_SLTI:

        {

            int16_t imm = (uint8_t) ctx->opcode;

            gen_slt_imm(ctx, OPC_SLTI, 24, rx, imm);

        }

        break;

    case M16_OPC_SLTIU:

        {

            int16_t imm = (uint8_t) ctx->opcode;

            gen_slt_imm(ctx, OPC_SLTIU, 24, rx, imm);

        }

        break;

    case M16_OPC_I8:

        {

            int reg32;



            funct = (ctx->opcode >> 8) & 0x7;

            switch (funct) {

            case I8_BTEQZ:

                gen_compute_branch(ctx, OPC_BEQ, 2, 24, 0,

                                   ((int8_t)ctx->opcode) << 1);

                break;

            case I8_BTNEZ:

                gen_compute_branch(ctx, OPC_BNE, 2, 24, 0,

                                   ((int8_t)ctx->opcode) << 1);

                break;

            case I8_SWRASP:

                gen_st(ctx, OPC_SW, 31, 29, (ctx->opcode & 0xff) << 2);

                break;

            case I8_ADJSP:

                gen_arith_imm(ctx, OPC_ADDIU, 29, 29,

                              ((int8_t)ctx->opcode) << 3);

                break;

            case I8_SVRS:

                {

                    int do_ra = ctx->opcode & (1 << 6);

                    int do_s0 = ctx->opcode & (1 << 5);

                    int do_s1 = ctx->opcode & (1 << 4);

                    int framesize = ctx->opcode & 0xf;



                    if (framesize == 0) {

                        framesize = 128;

                    } else {

                        framesize = framesize << 3;

                    }



                    if (ctx->opcode & (1 << 7)) {

                        gen_mips16_save(ctx, 0, 0,

                                        do_ra, do_s0, do_s1, framesize);

                    } else {

                        gen_mips16_restore(ctx, 0, 0,

                                           do_ra, do_s0, do_s1, framesize);

                    }

                }

                break;

            case I8_MOV32R:

                {

                    int rz = xlat(ctx->opcode & 0x7);



                    reg32 = (((ctx->opcode >> 3) & 0x3) << 3) |

                        ((ctx->opcode >> 5) & 0x7);

                    gen_arith(ctx, OPC_ADDU, reg32, rz, 0);

                }

                break;

            case I8_MOVR32:

                reg32 = ctx->opcode & 0x1f;

                gen_arith(ctx, OPC_ADDU, ry, reg32, 0);

                break;

            default:

                generate_exception(ctx, EXCP_RI);

                break;

            }

        }

        break;

    case M16_OPC_LI:

        {

            int16_t imm = (uint8_t) ctx->opcode;



            gen_arith_imm(ctx, OPC_ADDIU, rx, 0, imm);

        }

        break;

    case M16_OPC_CMPI:

        {

            int16_t imm = (uint8_t) ctx->opcode;

            gen_logic_imm(ctx, OPC_XORI, 24, rx, imm);

        }

        break;

#if defined(TARGET_MIPS64)

    case M16_OPC_SD:

        check_mips_64(ctx);

        gen_st(ctx, OPC_SD, ry, rx, offset << 3);

        break;

#endif

    case M16_OPC_LB:

        gen_ld(ctx, OPC_LB, ry, rx, offset);

        break;

    case M16_OPC_LH:

        gen_ld(ctx, OPC_LH, ry, rx, offset << 1);

        break;

    case M16_OPC_LWSP:

        gen_ld(ctx, OPC_LW, rx, 29, ((uint8_t)ctx->opcode) << 2);

        break;

    case M16_OPC_LW:

        gen_ld(ctx, OPC_LW, ry, rx, offset << 2);

        break;

    case M16_OPC_LBU:

        gen_ld(ctx, OPC_LBU, ry, rx, offset);

        break;

    case M16_OPC_LHU:

        gen_ld(ctx, OPC_LHU, ry, rx, offset << 1);

        break;

    case M16_OPC_LWPC:

        gen_ld(ctx, OPC_LWPC, rx, 0, ((uint8_t)ctx->opcode) << 2);

        break;

#if defined (TARGET_MIPS64)

    case M16_OPC_LWU:

        check_mips_64(ctx);

        gen_ld(ctx, OPC_LWU, ry, rx, offset << 2);

        break;

#endif

    case M16_OPC_SB:

        gen_st(ctx, OPC_SB, ry, rx, offset);

        break;

    case M16_OPC_SH:

        gen_st(ctx, OPC_SH, ry, rx, offset << 1);

        break;

    case M16_OPC_SWSP:

        gen_st(ctx, OPC_SW, rx, 29, ((uint8_t)ctx->opcode) << 2);

        break;

    case M16_OPC_SW:

        gen_st(ctx, OPC_SW, ry, rx, offset << 2);

        break;

    case M16_OPC_RRR:

        {

            int rz = xlat((ctx->opcode >> 2) & 0x7);

            int mips32_op;



            switch (ctx->opcode & 0x3) {

            case RRR_ADDU:

                mips32_op = OPC_ADDU;

                break;

            case RRR_SUBU:

                mips32_op = OPC_SUBU;

                break;

#if defined(TARGET_MIPS64)

            case RRR_DADDU:

                mips32_op = OPC_DADDU;

                check_mips_64(ctx);

                break;

            case RRR_DSUBU:

                mips32_op = OPC_DSUBU;

                check_mips_64(ctx);

                break;

#endif

            default:

                generate_exception(ctx, EXCP_RI);

                goto done;

            }



            gen_arith(ctx, mips32_op, rz, rx, ry);

        done:

            ;

        }

        break;

    case M16_OPC_RR:

        switch (op1) {

        case RR_JR:

            {

                int nd = (ctx->opcode >> 7) & 0x1;

                int link = (ctx->opcode >> 6) & 0x1;

                int ra = (ctx->opcode >> 5) & 0x1;



                if (link) {

                    op = nd ? OPC_JALRC : OPC_JALRS;

                } else {

                    op = OPC_JR;

                }



                gen_compute_branch(ctx, op, 2, ra ? 31 : rx, 31, 0);

                if (!nd) {

                    *is_branch = 1;

                }

            }

            break;

        case RR_SDBBP:

            /* XXX: not clear which exception should be raised

             *      when in debug mode...

             */

            check_insn(ctx, ISA_MIPS32);

            if (!(ctx->hflags & MIPS_HFLAG_DM)) {

                generate_exception(ctx, EXCP_DBp);

            } else {

                generate_exception(ctx, EXCP_DBp);

            }

            break;

        case RR_SLT:

            gen_slt(ctx, OPC_SLT, 24, rx, ry);

            break;

        case RR_SLTU:

            gen_slt(ctx, OPC_SLTU, 24, rx, ry);

            break;

        case RR_BREAK:

            generate_exception(ctx, EXCP_BREAK);

            break;

        case RR_SLLV:

            gen_shift(ctx, OPC_SLLV, ry, rx, ry);

            break;

        case RR_SRLV:

            gen_shift(ctx, OPC_SRLV, ry, rx, ry);

            break;

        case RR_SRAV:

            gen_shift(ctx, OPC_SRAV, ry, rx, ry);

            break;

#if defined (TARGET_MIPS64)

        case RR_DSRL:

            check_mips_64(ctx);

            gen_shift_imm(ctx, OPC_DSRL, ry, ry, sa);

            break;

#endif

        case RR_CMP:

            gen_logic(ctx, OPC_XOR, 24, rx, ry);

            break;

        case RR_NEG:

            gen_arith(ctx, OPC_SUBU, rx, 0, ry);

            break;

        case RR_AND:

            gen_logic(ctx, OPC_AND, rx, rx, ry);

            break;

        case RR_OR:

            gen_logic(ctx, OPC_OR, rx, rx, ry);

            break;

        case RR_XOR:

            gen_logic(ctx, OPC_XOR, rx, rx, ry);

            break;

        case RR_NOT:

            gen_logic(ctx, OPC_NOR, rx, ry, 0);

            break;

        case RR_MFHI:

            gen_HILO(ctx, OPC_MFHI, 0, rx);

            break;

        case RR_CNVT:

            switch (cnvt_op) {

            case RR_RY_CNVT_ZEB:

                tcg_gen_ext8u_tl(cpu_gpr[rx], cpu_gpr[rx]);

                break;

            case RR_RY_CNVT_ZEH:

                tcg_gen_ext16u_tl(cpu_gpr[rx], cpu_gpr[rx]);

                break;

            case RR_RY_CNVT_SEB:

                tcg_gen_ext8s_tl(cpu_gpr[rx], cpu_gpr[rx]);

                break;

            case RR_RY_CNVT_SEH:

                tcg_gen_ext16s_tl(cpu_gpr[rx], cpu_gpr[rx]);

                break;

#if defined (TARGET_MIPS64)

            case RR_RY_CNVT_ZEW:

                check_mips_64(ctx);

                tcg_gen_ext32u_tl(cpu_gpr[rx], cpu_gpr[rx]);

                break;

            case RR_RY_CNVT_SEW:

                check_mips_64(ctx);

                tcg_gen_ext32s_tl(cpu_gpr[rx], cpu_gpr[rx]);

                break;

#endif

            default:

                generate_exception(ctx, EXCP_RI);

                break;

            }

            break;

        case RR_MFLO:

            gen_HILO(ctx, OPC_MFLO, 0, rx);

            break;

#if defined (TARGET_MIPS64)

        case RR_DSRA:

            check_mips_64(ctx);

            gen_shift_imm(ctx, OPC_DSRA, ry, ry, sa);

            break;

        case RR_DSLLV:

            check_mips_64(ctx);

            gen_shift(ctx, OPC_DSLLV, ry, rx, ry);

            break;

        case RR_DSRLV:

            check_mips_64(ctx);

            gen_shift(ctx, OPC_DSRLV, ry, rx, ry);

            break;

        case RR_DSRAV:

            check_mips_64(ctx);

            gen_shift(ctx, OPC_DSRAV, ry, rx, ry);

            break;

#endif

        case RR_MULT:

            gen_muldiv(ctx, OPC_MULT, 0, rx, ry);

            break;

        case RR_MULTU:

            gen_muldiv(ctx, OPC_MULTU, 0, rx, ry);

            break;

        case RR_DIV:

            gen_muldiv(ctx, OPC_DIV, 0, rx, ry);

            break;

        case RR_DIVU:

            gen_muldiv(ctx, OPC_DIVU, 0, rx, ry);

            break;

#if defined (TARGET_MIPS64)

        case RR_DMULT:

            check_mips_64(ctx);

            gen_muldiv(ctx, OPC_DMULT, 0, rx, ry);

            break;

        case RR_DMULTU:

            check_mips_64(ctx);

            gen_muldiv(ctx, OPC_DMULTU, 0, rx, ry);

            break;

        case RR_DDIV:

            check_mips_64(ctx);

            gen_muldiv(ctx, OPC_DDIV, 0, rx, ry);

            break;

        case RR_DDIVU:

            check_mips_64(ctx);

            gen_muldiv(ctx, OPC_DDIVU, 0, rx, ry);

            break;

#endif

        default:

            generate_exception(ctx, EXCP_RI);

            break;

        }

        break;

    case M16_OPC_EXTEND:

        decode_extended_mips16_opc(env, ctx, is_branch);

        n_bytes = 4;

        break;

#if defined(TARGET_MIPS64)

    case M16_OPC_I64:

        funct = (ctx->opcode >> 8) & 0x7;

        decode_i64_mips16(ctx, ry, funct, offset, 0);

        break;

#endif

    default:

        generate_exception(ctx, EXCP_RI);

        break;

    }



    return n_bytes;

}
