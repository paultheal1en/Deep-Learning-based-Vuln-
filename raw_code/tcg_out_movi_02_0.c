static inline void tcg_out_movi(TCGContext *s, TCGType type,

                                int ret, tcg_target_long arg)

{

#if defined(__sparc_v9__) && !defined(__sparc_v8plus__)

    if (arg != (arg & 0xffffffff))

        fprintf(stderr, "unimplemented %s with constant %ld\n", __func__, arg);

#endif

    if (arg == (arg & 0xfff))

        tcg_out32(s, ARITH_OR | INSN_RD(ret) | INSN_RS1(TCG_REG_G0) |

                  INSN_IMM13(arg));

    else {

        tcg_out32(s, SETHI | INSN_RD(ret) | ((arg & 0xfffffc00) >> 10));

        if (arg & 0x3ff)

            tcg_out32(s, ARITH_OR | INSN_RD(ret) | INSN_RS1(ret) |

                      INSN_IMM13(arg & 0x3ff));

    }

}
