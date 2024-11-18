hwaddr ppc_hash64_get_phys_page_debug(CPUPPCState *env, target_ulong addr)

{

    struct mmu_ctx_hash64 ctx;



    if (unlikely(ppc_hash64_get_physical_address(env, &ctx, addr, 0, ACCESS_INT)

                 != 0)) {

        return -1;

    }



    return ctx.raddr & TARGET_PAGE_MASK;

}
