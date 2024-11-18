static void do_svc_interrupt(CPUS390XState *env)

{

    uint64_t mask, addr;

    LowCore *lowcore;

    hwaddr len = TARGET_PAGE_SIZE;



    lowcore = cpu_physical_memory_map(env->psa, &len, 1);



    lowcore->svc_code = cpu_to_be16(env->int_svc_code);

    lowcore->svc_ilen = cpu_to_be16(env->int_svc_ilen);

    lowcore->svc_old_psw.mask = cpu_to_be64(get_psw_mask(env));

    lowcore->svc_old_psw.addr = cpu_to_be64(env->psw.addr + env->int_svc_ilen);

    mask = be64_to_cpu(lowcore->svc_new_psw.mask);

    addr = be64_to_cpu(lowcore->svc_new_psw.addr);



    cpu_physical_memory_unmap(lowcore, len, 1, len);



    load_psw(env, mask, addr);

}
