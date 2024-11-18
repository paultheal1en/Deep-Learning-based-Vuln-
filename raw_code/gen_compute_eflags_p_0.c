static void gen_compute_eflags_p(DisasContext *s, TCGv reg)

{

    gen_compute_eflags(s);

    tcg_gen_shri_tl(reg, cpu_cc_src, 2);

    tcg_gen_andi_tl(reg, reg, 1);

}
