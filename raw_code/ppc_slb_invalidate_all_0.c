void ppc_slb_invalidate_all (CPUPPCState *env)

{

    /* XXX: TODO */

    tlb_flush(env, 1);

}
