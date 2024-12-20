static void dec_store(DisasContext *dc)

{

    TCGv t, *addr;

    unsigned int size;



    size = 1 << (dc->opcode & 3);



    if (size > 4 && (dc->tb_flags & MSR_EE_FLAG)

          && !(dc->env->pvr.regs[2] & PVR2_ILL_OPCODE_EXC_MASK)) {

        tcg_gen_movi_tl(cpu_SR[SR_ESR], ESR_EC_ILLEGAL_OP);

        t_gen_raise_exception(dc, EXCP_HW_EXCP);

        return;

    }



    LOG_DIS("s%d%s\n", size, dc->type_b ? "i" : "");

    t_sync_flags(dc);

    /* If we get a fault on a dslot, the jmpstate better be in sync.  */

    sync_jmpstate(dc);

    addr = compute_ldst_addr(dc, &t);



    /* Verify alignment if needed.  */

    if ((dc->env->pvr.regs[2] & PVR2_UNALIGNED_EXC_MASK) && size > 1) {

        gen_helper_memalign(*addr, tcg_const_tl(dc->rd),

                            tcg_const_tl(1), tcg_const_tl(size - 1));

    }



    gen_store(dc, *addr, cpu_R[dc->rd], size);

    if (addr == &t)

        tcg_temp_free(t);

}
