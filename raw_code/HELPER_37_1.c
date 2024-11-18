uint32_t HELPER(tprot)(CPUS390XState *env, uint64_t a1, uint64_t a2)

{

    S390CPU *cpu = s390_env_get_cpu(env);

    CPUState *cs = CPU(cpu);



    /*

     * TODO: we currently don't handle all access protection types

     * (including access-list and key-controlled) as well as AR mode.

     */

    if (!s390_cpu_virt_mem_check_write(cpu, a1, 0, 1)) {

        /* Fetching permitted; storing permitted */

        return 0;

    }

    switch (env->int_pgm_code) {

    case PGM_PROTECTION:

        /* Fetching permitted; storing not permitted */

        cs->exception_index = 0;

        return 1;

    case PGM_ADDRESSING:

        /* Fetching not permitted; storing not permitted */

        cs->exception_index = 0;

        return 2;

    case PGM_ASCE_TYPE:

    case PGM_REG_FIRST_TRANS:

    case PGM_REG_SEC_TRANS:

    case PGM_REG_THIRD_TRANS:

    case PGM_SEGMENT_TRANS:

    case PGM_PAGE_TRANS:

    case PGM_ALET_SPEC:

    case PGM_ALEN_SPEC:

    case PGM_ALE_SEQ:

    case PGM_ASTE_VALID:

    case PGM_ASTE_SEQ:

    case PGM_EXT_AUTH:

        /* Translation not available */

        cs->exception_index = 0;

        return 3;

    }

    /* any other exception is forwarded to the guest */

    s390_cpu_virt_mem_handle_exc(cpu, GETPC());

    return 0;

}
