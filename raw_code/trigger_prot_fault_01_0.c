static void trigger_prot_fault(CPUS390XState *env, target_ulong vaddr,

                               uint64_t mode)

{

    CPUState *cs = CPU(s390_env_get_cpu(env));

    int ilen = ILEN_LATER_INC;

    int bits = trans_bits(env, mode) | 4;



    DPRINTF("%s: vaddr=%016" PRIx64 " bits=%d\n", __func__, vaddr, bits);



    stq_phys(cs->as,

             env->psa + offsetof(LowCore, trans_exc_code), vaddr | bits);

    trigger_pgm_exception(env, PGM_PROTECTION, ilen);

}
