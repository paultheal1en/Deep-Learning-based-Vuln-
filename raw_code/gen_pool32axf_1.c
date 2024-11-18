static void gen_pool32axf (CPUMIPSState *env, DisasContext *ctx, int rt, int rs,

                           int *is_branch)

{

    int extension = (ctx->opcode >> 6) & 0x3f;

    int minor = (ctx->opcode >> 12) & 0xf;

    uint32_t mips32_op;



    switch (extension) {

    case TEQ:

        mips32_op = OPC_TEQ;

        goto do_trap;

    case TGE:

        mips32_op = OPC_TGE;

        goto do_trap;

    case TGEU:

        mips32_op = OPC_TGEU;

        goto do_trap;

    case TLT:

        mips32_op = OPC_TLT;

        goto do_trap;

    case TLTU:

        mips32_op = OPC_TLTU;

        goto do_trap;

    case TNE:

        mips32_op = OPC_TNE;

    do_trap:

        gen_trap(ctx, mips32_op, rs, rt, -1);

        break;

#ifndef CONFIG_USER_ONLY

    case MFC0:

    case MFC0 + 32:

        check_cp0_enabled(ctx);

        if (rt == 0) {

            /* Treat as NOP. */

            break;

        }

        gen_mfc0(ctx, cpu_gpr[rt], rs, (ctx->opcode >> 11) & 0x7);

        break;

    case MTC0:

    case MTC0 + 32:

        check_cp0_enabled(ctx);

        {

            TCGv t0 = tcg_temp_new();



            gen_load_gpr(t0, rt);

            gen_mtc0(ctx, t0, rs, (ctx->opcode >> 11) & 0x7);

            tcg_temp_free(t0);

        }

        break;

#endif

    case 0x2c:

        switch (minor) {

        case SEB:

            gen_bshfl(ctx, OPC_SEB, rs, rt);

            break;

        case SEH:

            gen_bshfl(ctx, OPC_SEH, rs, rt);

            break;

        case CLO:

            mips32_op = OPC_CLO;

            goto do_cl;

        case CLZ:

            mips32_op = OPC_CLZ;

        do_cl:

            check_insn(ctx, ISA_MIPS32);

            gen_cl(ctx, mips32_op, rt, rs);

            break;

        case RDHWR:

            gen_rdhwr(ctx, rt, rs);

            break;

        case WSBH:

            gen_bshfl(ctx, OPC_WSBH, rs, rt);

            break;

        case MULT:

            mips32_op = OPC_MULT;

            goto do_mul;

        case MULTU:

            mips32_op = OPC_MULTU;

            goto do_mul;

        case DIV:

            mips32_op = OPC_DIV;

            goto do_div;

        case DIVU:

            mips32_op = OPC_DIVU;

            goto do_div;

        do_div:

            check_insn(ctx, ISA_MIPS32);

            gen_muldiv(ctx, mips32_op, 0, rs, rt);

            break;

        case MADD:

            mips32_op = OPC_MADD;

            goto do_mul;

        case MADDU:

            mips32_op = OPC_MADDU;

            goto do_mul;

        case MSUB:

            mips32_op = OPC_MSUB;

            goto do_mul;

        case MSUBU:

            mips32_op = OPC_MSUBU;

        do_mul:

            check_insn(ctx, ISA_MIPS32);

            gen_muldiv(ctx, mips32_op, (ctx->opcode >> 14) & 3, rs, rt);

            break;

        default:

            goto pool32axf_invalid;

        }

        break;

    case 0x34:

        switch (minor) {

        case MFC2:

        case MTC2:

        case MFHC2:

        case MTHC2:

        case CFC2:

        case CTC2:

            generate_exception_err(ctx, EXCP_CpU, 2);

            break;

        default:

            goto pool32axf_invalid;

        }

        break;

    case 0x3c:

        switch (minor) {

        case JALR:

        case JALR_HB:

            gen_compute_branch (ctx, OPC_JALR, 4, rs, rt, 0);

            *is_branch = 1;

            break;

        case JALRS:

        case JALRS_HB:

            gen_compute_branch (ctx, OPC_JALRS, 4, rs, rt, 0);

            *is_branch = 1;

            break;

        default:

            goto pool32axf_invalid;

        }

        break;

    case 0x05:

        switch (minor) {

        case RDPGPR:

            check_cp0_enabled(ctx);

            check_insn(ctx, ISA_MIPS32R2);

            gen_load_srsgpr(rt, rs);

            break;

        case WRPGPR:

            check_cp0_enabled(ctx);

            check_insn(ctx, ISA_MIPS32R2);

            gen_store_srsgpr(rt, rs);

            break;

        default:

            goto pool32axf_invalid;

        }

        break;

#ifndef CONFIG_USER_ONLY

    case 0x0d:

        switch (minor) {

        case TLBP:

            mips32_op = OPC_TLBP;

            goto do_cp0;

        case TLBR:

            mips32_op = OPC_TLBR;

            goto do_cp0;

        case TLBWI:

            mips32_op = OPC_TLBWI;

            goto do_cp0;

        case TLBWR:

            mips32_op = OPC_TLBWR;

            goto do_cp0;

        case WAIT:

            mips32_op = OPC_WAIT;

            goto do_cp0;

        case DERET:

            mips32_op = OPC_DERET;

            goto do_cp0;

        case ERET:

            mips32_op = OPC_ERET;

        do_cp0:

            gen_cp0(env, ctx, mips32_op, rt, rs);

            break;

        default:

            goto pool32axf_invalid;

        }

        break;

    case 0x1d:

        switch (minor) {

        case DI:

            check_cp0_enabled(ctx);

            {

                TCGv t0 = tcg_temp_new();



                save_cpu_state(ctx, 1);

                gen_helper_di(t0, cpu_env);

                gen_store_gpr(t0, rs);

                /* Stop translation as we may have switched the execution mode */

                ctx->bstate = BS_STOP;

                tcg_temp_free(t0);

            }

            break;

        case EI:

            check_cp0_enabled(ctx);

            {

                TCGv t0 = tcg_temp_new();



                save_cpu_state(ctx, 1);

                gen_helper_ei(t0, cpu_env);

                gen_store_gpr(t0, rs);

                /* Stop translation as we may have switched the execution mode */

                ctx->bstate = BS_STOP;

                tcg_temp_free(t0);

            }

            break;

        default:

            goto pool32axf_invalid;

        }

        break;

#endif

    case 0x2d:

        switch (minor) {

        case SYNC:

            /* NOP */

            break;

        case SYSCALL:

            generate_exception(ctx, EXCP_SYSCALL);

            ctx->bstate = BS_STOP;

            break;

        case SDBBP:

            check_insn(ctx, ISA_MIPS32);

            if (!(ctx->hflags & MIPS_HFLAG_DM)) {

                generate_exception(ctx, EXCP_DBp);

            } else {

                generate_exception(ctx, EXCP_DBp);

            }

            break;

        default:

            goto pool32axf_invalid;

        }

        break;

    case 0x35:

        switch (minor & 3) {

        case MFHI32:

            gen_HILO(ctx, OPC_MFHI, minor >> 2, rs);

            break;

        case MFLO32:

            gen_HILO(ctx, OPC_MFLO, minor >> 2, rs);

            break;

        case MTHI32:

            gen_HILO(ctx, OPC_MTHI, minor >> 2, rs);

            break;

        case MTLO32:

            gen_HILO(ctx, OPC_MTLO, minor >> 2, rs);

            break;

        default:

            goto pool32axf_invalid;

        }

        break;

    default:

    pool32axf_invalid:

        MIPS_INVAL("pool32axf");

        generate_exception(ctx, EXCP_RI);

        break;

    }

}
