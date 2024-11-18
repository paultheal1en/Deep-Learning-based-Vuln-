static void gen_compute_branch(DisasContext *ctx, uint32_t opc, int r1,

                               int r2 , int32_t constant , int32_t offset)

{

    TCGv temp, temp2;

    int n;



    switch (opc) {

/* SB-format jumps */

    case OPC1_16_SB_J:

    case OPC1_32_B_J:

        gen_goto_tb(ctx, 0, ctx->pc + offset * 2);

        break;

    case OPC1_32_B_CALL:

    case OPC1_16_SB_CALL:

        gen_helper_1arg(call, ctx->next_pc);

        gen_goto_tb(ctx, 0, ctx->pc + offset * 2);

        break;

    case OPC1_16_SB_JZ:

        gen_branch_condi(ctx, TCG_COND_EQ, cpu_gpr_d[15], 0, offset);

        break;

    case OPC1_16_SB_JNZ:

        gen_branch_condi(ctx, TCG_COND_NE, cpu_gpr_d[15], 0, offset);

        break;

/* SBC-format jumps */

    case OPC1_16_SBC_JEQ:

        gen_branch_condi(ctx, TCG_COND_EQ, cpu_gpr_d[15], constant, offset);

        break;

    case OPC1_16_SBC_JNE:

        gen_branch_condi(ctx, TCG_COND_NE, cpu_gpr_d[15], constant, offset);

        break;

/* SBRN-format jumps */

    case OPC1_16_SBRN_JZ_T:

        temp = tcg_temp_new();

        tcg_gen_andi_tl(temp, cpu_gpr_d[15], 0x1u << constant);

        gen_branch_condi(ctx, TCG_COND_EQ, temp, 0, offset);

        tcg_temp_free(temp);

        break;

    case OPC1_16_SBRN_JNZ_T:

        temp = tcg_temp_new();

        tcg_gen_andi_tl(temp, cpu_gpr_d[15], 0x1u << constant);

        gen_branch_condi(ctx, TCG_COND_NE, temp, 0, offset);

        tcg_temp_free(temp);

        break;

/* SBR-format jumps */

    case OPC1_16_SBR_JEQ:

        gen_branch_cond(ctx, TCG_COND_EQ, cpu_gpr_d[r1], cpu_gpr_d[15],

                        offset);

        break;

    case OPC1_16_SBR_JNE:

        gen_branch_cond(ctx, TCG_COND_NE, cpu_gpr_d[r1], cpu_gpr_d[15],

                        offset);

        break;

    case OPC1_16_SBR_JNZ:

        gen_branch_condi(ctx, TCG_COND_NE, cpu_gpr_d[r1], 0, offset);

        break;

    case OPC1_16_SBR_JNZ_A:

        gen_branch_condi(ctx, TCG_COND_NE, cpu_gpr_a[r1], 0, offset);

        break;

    case OPC1_16_SBR_JGEZ:

        gen_branch_condi(ctx, TCG_COND_GE, cpu_gpr_d[r1], 0, offset);

        break;

    case OPC1_16_SBR_JGTZ:

        gen_branch_condi(ctx, TCG_COND_GT, cpu_gpr_d[r1], 0, offset);

        break;

    case OPC1_16_SBR_JLEZ:

        gen_branch_condi(ctx, TCG_COND_LE, cpu_gpr_d[r1], 0, offset);

        break;

    case OPC1_16_SBR_JLTZ:

        gen_branch_condi(ctx, TCG_COND_LT, cpu_gpr_d[r1], 0, offset);

        break;

    case OPC1_16_SBR_JZ:

        gen_branch_condi(ctx, TCG_COND_EQ, cpu_gpr_d[r1], 0, offset);

        break;

    case OPC1_16_SBR_JZ_A:

        gen_branch_condi(ctx, TCG_COND_EQ, cpu_gpr_a[r1], 0, offset);

        break;

    case OPC1_16_SBR_LOOP:

        gen_loop(ctx, r1, offset * 2 - 32);

        break;

/* SR-format jumps */

    case OPC1_16_SR_JI:

        tcg_gen_andi_tl(cpu_PC, cpu_gpr_a[r1], 0xfffffffe);

        tcg_gen_exit_tb(0);

        break;

    case OPC2_32_SYS_RET:

    case OPC2_16_SR_RET:

        gen_helper_ret(cpu_env);

        tcg_gen_exit_tb(0);

        break;

/* B-format */

    case OPC1_32_B_CALLA:

        gen_helper_1arg(call, ctx->next_pc);

        gen_goto_tb(ctx, 0, EA_B_ABSOLUT(offset));

        break;

    case OPC1_32_B_FCALL:

        gen_fcall_save_ctx(ctx);

        gen_goto_tb(ctx, 0, ctx->pc + offset * 2);

        break;

    case OPC1_32_B_FCALLA:

        gen_fcall_save_ctx(ctx);

        gen_goto_tb(ctx, 0, EA_B_ABSOLUT(offset));

        break;

    case OPC1_32_B_JLA:

        tcg_gen_movi_tl(cpu_gpr_a[11], ctx->next_pc);

        /* fall through */

    case OPC1_32_B_JA:

        gen_goto_tb(ctx, 0, EA_B_ABSOLUT(offset));

        break;

    case OPC1_32_B_JL:

        tcg_gen_movi_tl(cpu_gpr_a[11], ctx->next_pc);

        gen_goto_tb(ctx, 0, ctx->pc + offset * 2);

        break;

/* BOL format */

    case OPCM_32_BRC_EQ_NEQ:

         if (MASK_OP_BRC_OP2(ctx->opcode) == OPC2_32_BRC_JEQ) {

            gen_branch_condi(ctx, TCG_COND_EQ, cpu_gpr_d[r1], constant, offset);

         } else {

            gen_branch_condi(ctx, TCG_COND_NE, cpu_gpr_d[r1], constant, offset);

         }

         break;

    case OPCM_32_BRC_GE:

         if (MASK_OP_BRC_OP2(ctx->opcode) == OP2_32_BRC_JGE) {

            gen_branch_condi(ctx, TCG_COND_GE, cpu_gpr_d[r1], constant, offset);

         } else {

            constant = MASK_OP_BRC_CONST4(ctx->opcode);

            gen_branch_condi(ctx, TCG_COND_GEU, cpu_gpr_d[r1], constant,

                             offset);

         }

         break;

    case OPCM_32_BRC_JLT:

         if (MASK_OP_BRC_OP2(ctx->opcode) == OPC2_32_BRC_JLT) {

            gen_branch_condi(ctx, TCG_COND_LT, cpu_gpr_d[r1], constant, offset);

         } else {

            constant = MASK_OP_BRC_CONST4(ctx->opcode);

            gen_branch_condi(ctx, TCG_COND_LTU, cpu_gpr_d[r1], constant,

                             offset);

         }

         break;

    case OPCM_32_BRC_JNE:

        temp = tcg_temp_new();

        if (MASK_OP_BRC_OP2(ctx->opcode) == OPC2_32_BRC_JNED) {

            tcg_gen_mov_tl(temp, cpu_gpr_d[r1]);

            /* subi is unconditional */

            tcg_gen_subi_tl(cpu_gpr_d[r1], cpu_gpr_d[r1], 1);

            gen_branch_condi(ctx, TCG_COND_NE, temp, constant, offset);

        } else {

            tcg_gen_mov_tl(temp, cpu_gpr_d[r1]);

            /* addi is unconditional */

            tcg_gen_addi_tl(cpu_gpr_d[r1], cpu_gpr_d[r1], 1);

            gen_branch_condi(ctx, TCG_COND_NE, temp, constant, offset);

        }

        tcg_temp_free(temp);

        break;

/* BRN format */

    case OPCM_32_BRN_JTT:

        n = MASK_OP_BRN_N(ctx->opcode);



        temp = tcg_temp_new();

        tcg_gen_andi_tl(temp, cpu_gpr_d[r1], (1 << n));



        if (MASK_OP_BRN_OP2(ctx->opcode) == OPC2_32_BRN_JNZ_T) {

            gen_branch_condi(ctx, TCG_COND_NE, temp, 0, offset);

        } else {

            gen_branch_condi(ctx, TCG_COND_EQ, temp, 0, offset);

        }

        tcg_temp_free(temp);

        break;

/* BRR Format */

    case OPCM_32_BRR_EQ_NEQ:

        if (MASK_OP_BRR_OP2(ctx->opcode) == OPC2_32_BRR_JEQ) {

            gen_branch_cond(ctx, TCG_COND_EQ, cpu_gpr_d[r1], cpu_gpr_d[r2],

                            offset);

        } else {

            gen_branch_cond(ctx, TCG_COND_NE, cpu_gpr_d[r1], cpu_gpr_d[r2],

                            offset);

        }

        break;

    case OPCM_32_BRR_ADDR_EQ_NEQ:

        if (MASK_OP_BRR_OP2(ctx->opcode) == OPC2_32_BRR_JEQ_A) {

            gen_branch_cond(ctx, TCG_COND_EQ, cpu_gpr_a[r1], cpu_gpr_a[r2],

                            offset);

        } else {

            gen_branch_cond(ctx, TCG_COND_NE, cpu_gpr_a[r1], cpu_gpr_a[r2],

                            offset);

        }

        break;

    case OPCM_32_BRR_GE:

        if (MASK_OP_BRR_OP2(ctx->opcode) == OPC2_32_BRR_JGE) {

            gen_branch_cond(ctx, TCG_COND_GE, cpu_gpr_d[r1], cpu_gpr_d[r2],

                            offset);

        } else {

            gen_branch_cond(ctx, TCG_COND_GEU, cpu_gpr_d[r1], cpu_gpr_d[r2],

                            offset);

        }

        break;

    case OPCM_32_BRR_JLT:

        if (MASK_OP_BRR_OP2(ctx->opcode) == OPC2_32_BRR_JLT) {

            gen_branch_cond(ctx, TCG_COND_LT, cpu_gpr_d[r1], cpu_gpr_d[r2],

                            offset);

        } else {

            gen_branch_cond(ctx, TCG_COND_LTU, cpu_gpr_d[r1], cpu_gpr_d[r2],

                            offset);

        }

        break;

    case OPCM_32_BRR_LOOP:

        if (MASK_OP_BRR_OP2(ctx->opcode) == OPC2_32_BRR_LOOP) {

            gen_loop(ctx, r2, offset * 2);

        } else {

            /* OPC2_32_BRR_LOOPU */

            gen_goto_tb(ctx, 0, ctx->pc + offset * 2);

        }

        break;

    case OPCM_32_BRR_JNE:

        temp = tcg_temp_new();

        temp2 = tcg_temp_new();

        if (MASK_OP_BRC_OP2(ctx->opcode) == OPC2_32_BRR_JNED) {

            tcg_gen_mov_tl(temp, cpu_gpr_d[r1]);

            /* also save r2, in case of r1 == r2, so r2 is not decremented */

            tcg_gen_mov_tl(temp2, cpu_gpr_d[r2]);

            /* subi is unconditional */

            tcg_gen_subi_tl(cpu_gpr_d[r1], cpu_gpr_d[r1], 1);

            gen_branch_cond(ctx, TCG_COND_NE, temp, temp2, offset);

        } else {

            tcg_gen_mov_tl(temp, cpu_gpr_d[r1]);

            /* also save r2, in case of r1 == r2, so r2 is not decremented */

            tcg_gen_mov_tl(temp2, cpu_gpr_d[r2]);

            /* addi is unconditional */

            tcg_gen_addi_tl(cpu_gpr_d[r1], cpu_gpr_d[r1], 1);

            gen_branch_cond(ctx, TCG_COND_NE, temp, temp2, offset);

        }

        tcg_temp_free(temp);

        tcg_temp_free(temp2);

        break;

    case OPCM_32_BRR_JNZ:

        if (MASK_OP_BRR_OP2(ctx->opcode) == OPC2_32_BRR_JNZ_A) {

            gen_branch_condi(ctx, TCG_COND_NE, cpu_gpr_a[r1], 0, offset);

        } else {

            gen_branch_condi(ctx, TCG_COND_EQ, cpu_gpr_a[r1], 0, offset);

        }

        break;

    default:

        printf("Branch Error at %x\n", ctx->pc);

    }

    ctx->bstate = BS_BRANCH;

}
