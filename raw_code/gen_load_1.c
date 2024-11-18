static inline TCGv gen_load(DisasContext * s, int opsize, TCGv addr, int sign)

{

    TCGv tmp;

    int index = IS_USER(s);

    s->is_mem = 1;

    tmp = tcg_temp_new_i32();

    switch(opsize) {

    case OS_BYTE:

        if (sign)

            tcg_gen_qemu_ld8s(tmp, addr, index);

        else

            tcg_gen_qemu_ld8u(tmp, addr, index);

        break;

    case OS_WORD:

        if (sign)

            tcg_gen_qemu_ld16s(tmp, addr, index);

        else

            tcg_gen_qemu_ld16u(tmp, addr, index);

        break;

    case OS_LONG:

    case OS_SINGLE:

        tcg_gen_qemu_ld32u(tmp, addr, index);

        break;

    default:

        qemu_assert(0, "bad load size");

    }

    gen_throws_exception = gen_last_qop;

    return tmp;

}
