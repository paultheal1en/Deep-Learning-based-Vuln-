static void gen_load_exclusive(DisasContext *s, int rt, int rt2,

                               TCGv addr, int size)

{

    TCGv tmp;



    switch (size) {

    case 0:

        tmp = gen_ld8u(addr, IS_USER(s));

        break;

    case 1:

        tmp = gen_ld16u(addr, IS_USER(s));

        break;

    case 2:

    case 3:

        tmp = gen_ld32(addr, IS_USER(s));

        break;

    default:

        abort();

    }

    tcg_gen_mov_i32(cpu_exclusive_val, tmp);

    store_reg(s, rt, tmp);

    if (size == 3) {

        TCGv tmp2 = new_tmp();

        tcg_gen_addi_i32(tmp2, addr, 4);

        tmp = gen_ld32(tmp2, IS_USER(s));

        dead_tmp(tmp2);

        tcg_gen_mov_i32(cpu_exclusive_high, tmp);

        store_reg(s, rt2, tmp);

    }

    tcg_gen_mov_i32(cpu_exclusive_addr, addr);

}
