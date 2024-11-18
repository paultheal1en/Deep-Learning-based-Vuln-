static inline void do_rfi(CPUPPCState *env, target_ulong nip, target_ulong msr,
                          target_ulong msrm, int keep_msrh)
{
    CPUState *cs = CPU(ppc_env_get_cpu(env));
#if defined(TARGET_PPC64)
    if (msr_is_64bit(env, msr)) {
        nip = (uint64_t)nip;
        msr &= (uint64_t)msrm;
    } else {
        nip = (uint32_t)nip;
        msr = (uint32_t)(msr & msrm);
        if (keep_msrh) {
            msr |= env->msr & ~((uint64_t)0xFFFFFFFF);
        }
    }
#else
    nip = (uint32_t)nip;
    msr &= (uint32_t)msrm;
#endif
    /* XXX: beware: this is false if VLE is supported */
    env->nip = nip & ~((target_ulong)0x00000003);
    hreg_store_msr(env, msr, 1);
#if defined(DEBUG_OP)
    cpu_dump_rfi(env->nip, env->msr);
#endif
    /* No need to raise an exception here,
     * as rfi is always the last insn of a TB
     */
    cs->interrupt_request |= CPU_INTERRUPT_EXITTB;
}