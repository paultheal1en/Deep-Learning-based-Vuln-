void HELPER(wer)(CPUXtensaState *env, uint32_t data, uint32_t addr)

{

    address_space_stl(env->address_space_er, addr, data,

                      (MemTxAttrs){0}, NULL);

}
