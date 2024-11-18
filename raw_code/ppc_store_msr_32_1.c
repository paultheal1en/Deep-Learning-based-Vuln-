void ppc_store_msr_32 (CPUPPCState *env, uint32_t value)

{

    do_store_msr(env,

                 (do_load_msr(env) & ~0xFFFFFFFFULL) | (value & 0xFFFFFFFF));

}
