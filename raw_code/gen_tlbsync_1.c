static void gen_tlbsync(DisasContext *ctx)

{

#if defined(CONFIG_USER_ONLY)

    GEN_PRIV;

#else

    CHK_HV;



    /* tlbsync is a nop for server, ptesync handles delayed tlb flush,

     * embedded however needs to deal with tlbsync. We don't try to be

     * fancy and swallow the overhead of checking for both.

     */

    gen_check_tlb_flush(ctx);

#endif /* defined(CONFIG_USER_ONLY) */

}
