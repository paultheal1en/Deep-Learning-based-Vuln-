static inline void wb_SR_F(void)

{

    int label;



    label = gen_new_label();

    tcg_gen_andi_tl(cpu_sr, cpu_sr, ~SR_F);

    tcg_gen_brcondi_tl(TCG_COND_EQ, env_btaken, 0, label);

    tcg_gen_ori_tl(cpu_sr, cpu_sr, SR_F);

    gen_set_label(label);

}
