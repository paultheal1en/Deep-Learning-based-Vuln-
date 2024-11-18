static bool gen_check_loop_end(DisasContext *dc, int slot)

{

    if (option_enabled(dc, XTENSA_OPTION_LOOP) &&

            !(dc->tb->flags & XTENSA_TBFLAG_EXCM) &&

            dc->next_pc == dc->lend) {

        int label = gen_new_label();



        gen_advance_ccount(dc);

        tcg_gen_brcondi_i32(TCG_COND_EQ, cpu_SR[LCOUNT], 0, label);

        tcg_gen_subi_i32(cpu_SR[LCOUNT], cpu_SR[LCOUNT], 1);

        gen_jumpi(dc, dc->lbeg, slot);

        gen_set_label(label);

        gen_jumpi(dc, dc->next_pc, -1);

        return true;

    }

    return false;

}
