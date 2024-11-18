static inline void gen_branch_slot(uint32_t delayed_pc, int t)

{

    TCGv sr;

    int label = gen_new_label();

    tcg_gen_movi_i32(cpu_delayed_pc, delayed_pc);

    sr = tcg_temp_new();

    tcg_gen_andi_i32(sr, cpu_sr, SR_T);

    tcg_gen_brcondi_i32(t ? TCG_COND_EQ:TCG_COND_NE, sr, 0, label);

    tcg_gen_ori_i32(cpu_flags, cpu_flags, DELAY_SLOT_TRUE);

    gen_set_label(label);

}
