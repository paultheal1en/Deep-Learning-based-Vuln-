static void mvc_fast_memmove(CPUS390XState *env, uint32_t l, uint64_t dest,

                             uint64_t src)

{

    S390CPU *cpu = s390_env_get_cpu(env);

    hwaddr dest_phys;

    hwaddr src_phys;

    hwaddr len = l;

    void *dest_p;

    void *src_p;

    uint64_t asc = env->psw.mask & PSW_MASK_ASC;

    int flags;



    if (mmu_translate(env, dest, 1, asc, &dest_phys, &flags, true)) {

        cpu_stb_data(env, dest, 0);

        cpu_abort(CPU(cpu), "should never reach here");

    }

    dest_phys |= dest & ~TARGET_PAGE_MASK;



    if (mmu_translate(env, src, 0, asc, &src_phys, &flags, true)) {

        cpu_ldub_data(env, src);

        cpu_abort(CPU(cpu), "should never reach here");

    }

    src_phys |= src & ~TARGET_PAGE_MASK;



    dest_p = cpu_physical_memory_map(dest_phys, &len, 1);

    src_p = cpu_physical_memory_map(src_phys, &len, 0);



    memmove(dest_p, src_p, len);



    cpu_physical_memory_unmap(dest_p, 1, len, len);

    cpu_physical_memory_unmap(src_p, 0, len, len);

}
