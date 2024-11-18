static void dec_div(DisasContext *dc)

{

    unsigned int u;



    u = dc->imm & 2; 

    LOG_DIS("div\n");



    if (!(dc->env->pvr.regs[2] & PVR2_ILL_OPCODE_EXC_MASK)

          && !((dc->env->pvr.regs[0] & PVR0_USE_DIV_MASK))) {

        tcg_gen_movi_tl(cpu_SR[SR_ESR], ESR_EC_ILLEGAL_OP);

        t_gen_raise_exception(dc, EXCP_HW_EXCP);

    }



    if (u)

        gen_helper_divu(cpu_R[dc->rd], *(dec_alu_op_b(dc)), cpu_R[dc->ra]);

    else

        gen_helper_divs(cpu_R[dc->rd], *(dec_alu_op_b(dc)), cpu_R[dc->ra]);

    if (!dc->rd)

        tcg_gen_movi_tl(cpu_R[dc->rd], 0);

}
