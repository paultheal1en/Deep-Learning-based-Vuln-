static int cpu_post_load(void *opaque, int version_id)

{

    PowerPCCPU *cpu = opaque;

    CPUPPCState *env = &cpu->env;

    int i;

    target_ulong msr;



    /*

     * If we're operating in compat mode, we should be ok as long as

     * the destination supports the same compatiblity mode.

     *

     * Otherwise, however, we require that the destination has exactly

     * the same CPU model as the source.

     */



#if defined(TARGET_PPC64)

    if (cpu->compat_pvr) {

        Error *local_err = NULL;



        ppc_set_compat(cpu, cpu->compat_pvr, &local_err);

        if (local_err) {

            error_report_err(local_err);

            error_free(local_err);

            return -1;

        }

    } else

#endif

    {

        if (!pvr_match(cpu, env->spr[SPR_PVR])) {

            return -1;

        }

    }



    env->lr = env->spr[SPR_LR];

    env->ctr = env->spr[SPR_CTR];

    cpu_write_xer(env, env->spr[SPR_XER]);

#if defined(TARGET_PPC64)

    env->cfar = env->spr[SPR_CFAR];

#endif

    env->spe_fscr = env->spr[SPR_BOOKE_SPEFSCR];



    for (i = 0; (i < 4) && (i < env->nb_BATs); i++) {

        env->DBAT[0][i] = env->spr[SPR_DBAT0U + 2*i];

        env->DBAT[1][i] = env->spr[SPR_DBAT0U + 2*i + 1];

        env->IBAT[0][i] = env->spr[SPR_IBAT0U + 2*i];

        env->IBAT[1][i] = env->spr[SPR_IBAT0U + 2*i + 1];

    }

    for (i = 0; (i < 4) && ((i+4) < env->nb_BATs); i++) {

        env->DBAT[0][i+4] = env->spr[SPR_DBAT4U + 2*i];

        env->DBAT[1][i+4] = env->spr[SPR_DBAT4U + 2*i + 1];

        env->IBAT[0][i+4] = env->spr[SPR_IBAT4U + 2*i];

        env->IBAT[1][i+4] = env->spr[SPR_IBAT4U + 2*i + 1];

    }



    if (!cpu->vhyp) {

        ppc_store_sdr1(env, env->spr[SPR_SDR1]);

    }



    /* Invalidate all msr bits except MSR_TGPR/MSR_HVB before restoring */

    msr = env->msr;

    env->msr ^= ~((1ULL << MSR_TGPR) | MSR_HVB);

    ppc_store_msr(env, msr);



    hreg_compute_mem_idx(env);



    return 0;

}
