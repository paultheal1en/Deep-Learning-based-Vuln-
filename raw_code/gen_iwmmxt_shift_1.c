static inline int gen_iwmmxt_shift(uint32_t insn, uint32_t mask, TCGv dest)

{

    int rd = (insn >> 0) & 0xf;

    TCGv tmp;



    if (insn & (1 << 8)) {

        if (rd < ARM_IWMMXT_wCGR0 || rd > ARM_IWMMXT_wCGR3) {

            return 1;

        } else {

            tmp = iwmmxt_load_creg(rd);

        }

    } else {

        tmp = new_tmp();

        iwmmxt_load_reg(cpu_V0, rd);

        tcg_gen_trunc_i64_i32(tmp, cpu_V0);

    }

    tcg_gen_andi_i32(tmp, tmp, mask);

    tcg_gen_mov_i32(dest, tmp);

    dead_tmp(tmp);

    return 0;

}
