void ppc_store_sdr1(CPUPPCState *env, target_ulong value)

{

    LOG_MMU("%s: " TARGET_FMT_lx "\n", __func__, value);

    if (env->spr[SPR_SDR1] != value) {

        env->spr[SPR_SDR1] = value;

#if defined(TARGET_PPC64)

        if (env->mmu_model & POWERPC_MMU_64) {

            target_ulong htabsize = value & SDR_64_HTABSIZE;



            if (htabsize > 28) {

                fprintf(stderr, "Invalid HTABSIZE 0x" TARGET_FMT_lx

                        " stored in SDR1\n", htabsize);

                htabsize = 28;

            }

            env->htab_mask = (1ULL << (htabsize + 18)) - 1;

            env->htab_base = value & SDR_64_HTABORG;

        } else

#endif /* defined(TARGET_PPC64) */

        {

            /* FIXME: Should check for valid HTABMASK values */

            env->htab_mask = ((value & SDR_32_HTABMASK) << 16) | 0xFFFF;

            env->htab_base = value & SDR_32_HTABORG;

        }

        tlb_flush(env, 1);

    }

}
