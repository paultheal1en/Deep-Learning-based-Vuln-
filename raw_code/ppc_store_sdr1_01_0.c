void ppc_store_sdr1 (CPUPPCState *env, target_ulong value)

{

    LOG_MMU("%s: " TARGET_FMT_lx "\n", __func__, value);

    if (env->sdr1 != value) {

        /* XXX: for PowerPC 64, should check that the HTABSIZE value

         *      is <= 28

         */

        env->sdr1 = value;

        tlb_flush(env, 1);

    }

}
