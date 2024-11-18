static void decode_rrr_divide(CPUTriCoreState *env, DisasContext *ctx)

{

    uint32_t op2;



    int r1, r2, r3, r4;



    op2 = MASK_OP_RRR_OP2(ctx->opcode);

    r1 = MASK_OP_RRR_S1(ctx->opcode);

    r2 = MASK_OP_RRR_S2(ctx->opcode);

    r3 = MASK_OP_RRR_S3(ctx->opcode);

    r4 = MASK_OP_RRR_D(ctx->opcode);



    CHECK_REG_PAIR(r3);



    switch (op2) {

    case OPC2_32_RRR_DVADJ:

        CHECK_REG_PAIR(r4);

        GEN_HELPER_RRR(dvadj, cpu_gpr_d[r4], cpu_gpr_d[r4+1], cpu_gpr_d[r3],

                       cpu_gpr_d[r3+1], cpu_gpr_d[r2]);

        break;

    case OPC2_32_RRR_DVSTEP:

        CHECK_REG_PAIR(r4);

        GEN_HELPER_RRR(dvstep, cpu_gpr_d[r4], cpu_gpr_d[r4+1], cpu_gpr_d[r3],

                       cpu_gpr_d[r3+1], cpu_gpr_d[r2]);

        break;

    case OPC2_32_RRR_DVSTEP_U:

        CHECK_REG_PAIR(r4);

        GEN_HELPER_RRR(dvstep_u, cpu_gpr_d[r4], cpu_gpr_d[r4+1], cpu_gpr_d[r3],

                       cpu_gpr_d[r3+1], cpu_gpr_d[r2]);

        break;

    case OPC2_32_RRR_IXMAX:

        CHECK_REG_PAIR(r4);

        GEN_HELPER_RRR(ixmax, cpu_gpr_d[r4], cpu_gpr_d[r4+1], cpu_gpr_d[r3],

                       cpu_gpr_d[r3+1], cpu_gpr_d[r2]);

        break;

    case OPC2_32_RRR_IXMAX_U:

        CHECK_REG_PAIR(r4);

        GEN_HELPER_RRR(ixmax_u, cpu_gpr_d[r4], cpu_gpr_d[r4+1], cpu_gpr_d[r3],

                       cpu_gpr_d[r3+1], cpu_gpr_d[r2]);

        break;

    case OPC2_32_RRR_IXMIN:

        CHECK_REG_PAIR(r4);

        GEN_HELPER_RRR(ixmin, cpu_gpr_d[r4], cpu_gpr_d[r4+1], cpu_gpr_d[r3],

                       cpu_gpr_d[r3+1], cpu_gpr_d[r2]);

        break;

    case OPC2_32_RRR_IXMIN_U:

        CHECK_REG_PAIR(r4);

        GEN_HELPER_RRR(ixmin_u, cpu_gpr_d[r4], cpu_gpr_d[r4+1], cpu_gpr_d[r3],

                       cpu_gpr_d[r3+1], cpu_gpr_d[r2]);

        break;

    case OPC2_32_RRR_PACK:

        gen_helper_pack(cpu_gpr_d[r4], cpu_PSW_C, cpu_gpr_d[r3],

                        cpu_gpr_d[r3+1], cpu_gpr_d[r1]);

        break;

    default:

        generate_trap(ctx, TRAPC_INSN_ERR, TIN2_IOPC);

    }

}
