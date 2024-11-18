void do_POWER_rac (void)

{

#if 0

    mmu_ctx_t ctx;



    /* We don't have to generate many instances of this instruction,

     * as rac is supervisor only.

     */

    if (get_physical_address(env, &ctx, T0, 0, ACCESS_INT, 1) == 0)

        T0 = ctx.raddr;

#endif

}
