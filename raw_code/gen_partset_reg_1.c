static void gen_partset_reg(int opsize, TCGv reg, TCGv val)

{

    TCGv tmp;

    switch (opsize) {

    case OS_BYTE:

        tcg_gen_andi_i32(reg, reg, 0xffffff00);

        tmp = tcg_temp_new();

        tcg_gen_ext8u_i32(tmp, val);

        tcg_gen_or_i32(reg, reg, tmp);

        break;

    case OS_WORD:

        tcg_gen_andi_i32(reg, reg, 0xffff0000);

        tmp = tcg_temp_new();

        tcg_gen_ext16u_i32(tmp, val);

        tcg_gen_or_i32(reg, reg, tmp);

        break;

    case OS_LONG:

    case OS_SINGLE:

        tcg_gen_mov_i32(reg, val);

        break;

    default:

        qemu_assert(0, "Bad operand size");

        break;

    }

}
