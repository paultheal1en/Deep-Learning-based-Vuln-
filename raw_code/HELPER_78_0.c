uint32_t HELPER(rer)(CPUXtensaState *env, uint32_t addr)

{

    return address_space_ldl(env->address_space_er, addr,

                             (MemTxAttrs){0}, NULL);

}
