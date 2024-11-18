static inline void gen_op_mov_reg_v(int ot, int reg, TCGv t0)

{

    switch(ot) {

    case OT_BYTE:

        if (reg < 4 X86_64_DEF( || reg >= 8 || x86_64_hregs)) {

            tcg_gen_deposit_tl(cpu_regs[reg], cpu_regs[reg], t0, 0, 8);

        } else {

            tcg_gen_deposit_tl(cpu_regs[reg - 4], cpu_regs[reg - 4], t0, 8, 8);

        }

        break;

    case OT_WORD:

        tcg_gen_deposit_tl(cpu_regs[reg], cpu_regs[reg], t0, 0, 16);

        break;

    default: /* XXX this shouldn't be reached;  abort? */

    case OT_LONG:

        /* For x86_64, this sets the higher half of register to zero.

           For i386, this is equivalent to a mov. */

        tcg_gen_ext32u_tl(cpu_regs[reg], t0);

        break;

#ifdef TARGET_X86_64

    case OT_QUAD:

        tcg_gen_mov_tl(cpu_regs[reg], t0);

        break;

#endif

    }

}
