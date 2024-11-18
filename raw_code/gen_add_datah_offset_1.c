static inline void gen_add_datah_offset(DisasContext *s, unsigned int insn,

                                        int extra, TCGv var)

{

    int val, rm;

    TCGv offset;



    if (insn & (1 << 22)) {

        /* immediate */

        val = (insn & 0xf) | ((insn >> 4) & 0xf0);

        if (!(insn & (1 << 23)))

            val = -val;

        val += extra;

        if (val != 0)

            tcg_gen_addi_i32(var, var, val);

    } else {

        /* register */

        if (extra)

            tcg_gen_addi_i32(var, var, extra);

        rm = (insn) & 0xf;

        offset = load_reg(s, rm);

        if (!(insn & (1 << 23)))

            tcg_gen_sub_i32(var, var, offset);

        else

            tcg_gen_add_i32(var, var, offset);

        dead_tmp(offset);

    }

}
