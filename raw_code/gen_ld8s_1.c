static inline TCGv gen_ld8s(TCGv addr, int index)

{

    TCGv tmp = new_tmp();

    tcg_gen_qemu_ld8s(tmp, addr, index);

    return tmp;

}
