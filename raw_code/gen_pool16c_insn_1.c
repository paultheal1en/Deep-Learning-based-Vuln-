static void gen_pool16c_insn(DisasContext *ctx, int *is_branch)

{

    int rd = mmreg((ctx->opcode >> 3) & 0x7);

    int rs = mmreg(ctx->opcode & 0x7);

    int opc;



    switch (((ctx->opcode) >> 4) & 0x3f) {

    case NOT16 + 0:

    case NOT16 + 1:

    case NOT16 + 2:

    case NOT16 + 3:

        gen_logic(ctx, OPC_NOR, rd, rs, 0);

        break;

    case XOR16 + 0:

    case XOR16 + 1:

    case XOR16 + 2:

    case XOR16 + 3:

        gen_logic(ctx, OPC_XOR, rd, rd, rs);

        break;

    case AND16 + 0:

    case AND16 + 1:

    case AND16 + 2:

    case AND16 + 3:

        gen_logic(ctx, OPC_AND, rd, rd, rs);

        break;

    case OR16 + 0:

    case OR16 + 1:

    case OR16 + 2:

    case OR16 + 3:

        gen_logic(ctx, OPC_OR, rd, rd, rs);

        break;

    case LWM16 + 0:

    case LWM16 + 1:

    case LWM16 + 2:

    case LWM16 + 3:

        {

            static const int lwm_convert[] = { 0x11, 0x12, 0x13, 0x14 };

            int offset = ZIMM(ctx->opcode, 0, 4);



            gen_ldst_multiple(ctx, LWM32, lwm_convert[(ctx->opcode >> 4) & 0x3],

                              29, offset << 2);

        }

        break;

    case SWM16 + 0:

    case SWM16 + 1:

    case SWM16 + 2:

    case SWM16 + 3:

        {

            static const int swm_convert[] = { 0x11, 0x12, 0x13, 0x14 };

            int offset = ZIMM(ctx->opcode, 0, 4);



            gen_ldst_multiple(ctx, SWM32, swm_convert[(ctx->opcode >> 4) & 0x3],

                              29, offset << 2);

        }

        break;

    case JR16 + 0:

    case JR16 + 1:

        {

            int reg = ctx->opcode & 0x1f;



            gen_compute_branch(ctx, OPC_JR, 2, reg, 0, 0);

        }

        *is_branch = 1;

        break;

    case JRC16 + 0:

    case JRC16 + 1:

        {

            int reg = ctx->opcode & 0x1f;



            gen_compute_branch(ctx, OPC_JR, 2, reg, 0, 0);

            /* Let normal delay slot handling in our caller take us

               to the branch target.  */

        }

        break;

    case JALR16 + 0:

    case JALR16 + 1:

        opc = OPC_JALR;

        goto do_jalr;

    case JALR16S + 0:

    case JALR16S + 1:

        opc = OPC_JALRS;

    do_jalr:

        {

            int reg = ctx->opcode & 0x1f;



            gen_compute_branch(ctx, opc, 2, reg, 31, 0);

        }

        *is_branch = 1;

        break;

    case MFHI16 + 0:

    case MFHI16 + 1:

        gen_HILO(ctx, OPC_MFHI, 0, uMIPS_RS5(ctx->opcode));

        break;

    case MFLO16 + 0:

    case MFLO16 + 1:

        gen_HILO(ctx, OPC_MFLO, 0, uMIPS_RS5(ctx->opcode));

        break;

    case BREAK16:

        generate_exception(ctx, EXCP_BREAK);

        break;

    case SDBBP16:

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

    case JRADDIUSP + 0:

    case JRADDIUSP + 1:

        {

            int imm = ZIMM(ctx->opcode, 0, 5);



            gen_compute_branch(ctx, OPC_JR, 2, 31, 0, 0);

            gen_arith_imm(ctx, OPC_ADDIU, 29, 29, imm << 2);

            /* Let normal delay slot handling in our caller take us

               to the branch target.  */

        }

        break;

    default:

        generate_exception(ctx, EXCP_RI);

        break;

    }

}
