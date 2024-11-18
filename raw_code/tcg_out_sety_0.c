static inline void tcg_out_sety(TCGContext *s, tcg_target_long val)

{

    if (val == 0 || val == -1)

        tcg_out32(s, WRY | INSN_IMM13(val));

    else

        fprintf(stderr, "unimplemented sety %ld\n", (long)val);

}
