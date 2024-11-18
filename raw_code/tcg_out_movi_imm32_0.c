static inline void tcg_out_movi_imm32(TCGContext *s, int ret, uint32_t arg)

{

    if (check_fit_tl(arg, 12))

        tcg_out_movi_imm13(s, ret, arg);

    else {

        tcg_out_sethi(s, ret, arg);

        if (arg & 0x3ff)

            tcg_out_arithi(s, ret, ret, arg & 0x3ff, ARITH_OR);

    }

}
