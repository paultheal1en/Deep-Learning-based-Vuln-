static inline void gen_add_data_offset(DisasContext *s, unsigned int insn,

                                       TCGv var)

{

    int val, rm, shift, shiftop;

    TCGv offset;



    if (!(insn & (1 << 25))) {

        /* immediate */

        val = insn & 0xfff;

        if (!(insn & (1 << 23)))

            val = -val;

        if (val != 0)

            tcg_gen_addi_i32(var, var, val);

    } else {

        /* shift/register */

        rm = (insn) & 0xf;

        shift = (insn >> 7) & 0x1f;

        shiftop = (insn >> 5) & 3;

        offset = load_reg(s, rm);

        gen_arm_shift_im(offset, shiftop, shift, 0);

        if (!(insn & (1 << 23)))

            tcg_gen_sub_i32(var, var, offset);

        else

            tcg_gen_add_i32(var, var, offset);

        dead_tmp(offset);

    }

}
