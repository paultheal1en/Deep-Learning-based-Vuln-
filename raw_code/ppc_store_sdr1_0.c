void ppc_store_sdr1(CPUPPCState *env, target_ulong value)

{

    qemu_log_mask(CPU_LOG_MMU, "%s: " TARGET_FMT_lx "\n", __func__, value);

    assert(!env->external_htab);

    env->spr[SPR_SDR1] = value;

#if defined(TARGET_PPC64)

    if (env->mmu_model & POWERPC_MMU_64) {

        PowerPCCPU *cpu = ppc_env_get_cpu(env);

        Error *local_err = NULL;



        ppc_hash64_set_sdr1(cpu, value, &local_err);

        if (local_err) {

            error_report_err(local_err);

            error_free(local_err);

        }

    } else

#endif /* defined(TARGET_PPC64) */

    {

        /* FIXME: Should check for valid HTABMASK values */

        env->htab_mask = ((value & SDR_32_HTABMASK) << 16) | 0xFFFF;

        env->htab_base = value & SDR_32_HTABORG;

    }

}
