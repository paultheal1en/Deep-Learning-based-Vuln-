static inline TCGv gen_ld32(TCGv addr, int index)

{

    TCGv tmp = new_tmp();

    tcg_gen_qemu_ld32u(tmp, addr, index);

    return tmp;

}
