static void do_branch_reg(DisasContext *dc, int32_t offset, uint32_t insn,

                          TCGv r_cond, TCGv r_reg)

{

    unsigned int cond = GET_FIELD_SP(insn, 25, 27), a = (insn & (1 << 29));

    target_ulong target = dc->pc + offset;



    flush_cond(dc, r_cond);

    gen_cond_reg(r_cond, cond, r_reg);

    if (a) {

        gen_branch_a(dc, target, dc->npc, r_cond);

        dc->is_br = 1;

    } else {

        dc->pc = dc->npc;

        dc->jump_pc[0] = target;

        dc->jump_pc[1] = dc->npc + 4;

        dc->npc = JUMP_PC;

    }

}
