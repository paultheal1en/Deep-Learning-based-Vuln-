static inline void tcg_out_ldst(TCGContext *s, int ret, int addr, int offset, int op)

{

    if (offset == (offset & 0xfff))

        tcg_out32(s, op | INSN_RD(ret) | INSN_RS1(addr) |

                  INSN_IMM13(offset));

    else

        fprintf(stderr, "unimplemented %s with offset %d\n", __func__, offset);

}
