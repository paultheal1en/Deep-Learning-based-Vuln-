static void ppc_hash64_set_isi(CPUState *cs, CPUPPCState *env,

                               uint64_t error_code)

{

    bool vpm;



    if (msr_ir) {

        vpm = !!(env->spr[SPR_LPCR] & LPCR_VPM1);

    } else {

        vpm = !!(env->spr[SPR_LPCR] & LPCR_VPM0);

    }

    if (vpm && !msr_hv) {

        cs->exception_index = POWERPC_EXCP_HISI;

    } else {

        cs->exception_index = POWERPC_EXCP_ISI;

    }

    env->error_code = error_code;

}
