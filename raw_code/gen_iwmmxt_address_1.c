static inline int gen_iwmmxt_address(DisasContext *s, uint32_t insn, TCGv dest)

{

    int rd;

    uint32_t offset;

    TCGv tmp;



    rd = (insn >> 16) & 0xf;

    tmp = load_reg(s, rd);



    offset = (insn & 0xff) << ((insn >> 7) & 2);

    if (insn & (1 << 24)) {

        /* Pre indexed */

        if (insn & (1 << 23))

            tcg_gen_addi_i32(tmp, tmp, offset);

        else

            tcg_gen_addi_i32(tmp, tmp, -offset);

        tcg_gen_mov_i32(dest, tmp);

        if (insn & (1 << 21))

            store_reg(s, rd, tmp);

        else

            dead_tmp(tmp);

    } else if (insn & (1 << 21)) {

        /* Post indexed */

        tcg_gen_mov_i32(dest, tmp);

        if (insn & (1 << 23))

            tcg_gen_addi_i32(tmp, tmp, offset);

        else

            tcg_gen_addi_i32(tmp, tmp, -offset);

        store_reg(s, rd, tmp);

    } else if (!(insn & (1 << 23)))

        return 1;

    return 0;

}
