static inline void _t_gen_mov_TN_env(TCGv tn, int offset)

{

    if (offset > sizeof(CPUCRISState)) {

        fprintf(stderr, "wrong load from env from off=%d\n", offset);

    }

    tcg_gen_ld_tl(tn, cpu_env, offset);

}
