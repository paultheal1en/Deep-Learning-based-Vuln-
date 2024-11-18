static void gen_store_exclusive(DisasContext *s, int rd, int rt, int rt2,

                                TCGv addr, int size)

{

    TCGv tmp;

    int done_label;

    int fail_label;



    /* if (env->exclusive_addr == addr && env->exclusive_val == [addr]) {

         [addr] = {Rt};

         {Rd} = 0;

       } else {

         {Rd} = 1;

       } */

    fail_label = gen_new_label();

    done_label = gen_new_label();

    tcg_gen_brcond_i32(TCG_COND_NE, addr, cpu_exclusive_addr, fail_label);

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

    tcg_gen_brcond_i32(TCG_COND_NE, tmp, cpu_exclusive_val, fail_label);

    dead_tmp(tmp);

    if (size == 3) {

        TCGv tmp2 = new_tmp();

        tcg_gen_addi_i32(tmp2, addr, 4);

        tmp = gen_ld32(tmp2, IS_USER(s));

        dead_tmp(tmp2);

        tcg_gen_brcond_i32(TCG_COND_NE, tmp, cpu_exclusive_high, fail_label);

        dead_tmp(tmp);

    }

    tmp = load_reg(s, rt);

    switch (size) {

    case 0:

        gen_st8(tmp, addr, IS_USER(s));

        break;

    case 1:

        gen_st16(tmp, addr, IS_USER(s));

        break;

    case 2:

    case 3:

        gen_st32(tmp, addr, IS_USER(s));

        break;

    default:

        abort();

    }

    if (size == 3) {

        tcg_gen_addi_i32(addr, addr, 4);

        tmp = load_reg(s, rt2);

        gen_st32(tmp, addr, IS_USER(s));

    }

    tcg_gen_movi_i32(cpu_R[rd], 0);

    tcg_gen_br(done_label);

    gen_set_label(fail_label);

    tcg_gen_movi_i32(cpu_R[rd], 1);

    gen_set_label(done_label);

    tcg_gen_movi_i32(cpu_exclusive_addr, -1);

}
