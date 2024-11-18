void ppc_hash64_set_sdr1(PowerPCCPU *cpu, target_ulong value,

                         Error **errp)

{

    CPUPPCState *env = &cpu->env;

    target_ulong htabsize = value & SDR_64_HTABSIZE;



    env->spr[SPR_SDR1] = value;

    if (htabsize > 28) {

        error_setg(errp,

                   "Invalid HTABSIZE 0x" TARGET_FMT_lx" stored in SDR1",

                   htabsize);

        htabsize = 28;

    }

    env->htab_mask = (1ULL << (htabsize + 18 - 7)) - 1;

    env->htab_base = value & SDR_64_HTABORG;

}
