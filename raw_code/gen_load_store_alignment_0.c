static void gen_load_store_alignment(DisasContext *dc, int shift,

        TCGv_i32 addr, bool no_hw_alignment)

{

    if (!option_enabled(dc, XTENSA_OPTION_UNALIGNED_EXCEPTION)) {

        tcg_gen_andi_i32(addr, addr, ~0 << shift);

    } else if (option_enabled(dc, XTENSA_OPTION_HW_ALIGNMENT) &&

            no_hw_alignment) {

        int label = gen_new_label();

        TCGv_i32 tmp = tcg_temp_new_i32();

        tcg_gen_andi_i32(tmp, addr, ~(~0 << shift));

        tcg_gen_brcondi_i32(TCG_COND_EQ, tmp, 0, label);

        gen_exception_cause_vaddr(dc, LOAD_STORE_ALIGNMENT_CAUSE, addr);

        gen_set_label(label);

        tcg_temp_free(tmp);

    }

}
