void helper_icbi(target_ulong addr)

{

    addr &= ~(env->dcache_line_size - 1);

    /* Invalidate one cache line :

     * PowerPC specification says this is to be treated like a load

     * (not a fetch) by the MMU. To be sure it will be so,

     * do the load "by hand".

     */

    ldl(addr);

    tb_invalidate_page_range(addr, addr + env->icache_line_size);

}
