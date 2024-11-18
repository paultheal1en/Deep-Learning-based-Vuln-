static inline TCGv gen_ld8u(TCGv addr, int index)

{

    TCGv tmp = new_tmp();

    tcg_gen_qemu_ld8u(tmp, addr, index);

    return tmp;

}
