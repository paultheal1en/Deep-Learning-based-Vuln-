hwaddr s390_cpu_get_phys_page_debug(CPUState *cs, vaddr vaddr)

{

    S390CPU *cpu = S390_CPU(cs);

    CPUS390XState *env = &cpu->env;

    target_ulong raddr;

    int prot;

    uint64_t asc = env->psw.mask & PSW_MASK_ASC;



    /* 31-Bit mode */

    if (!(env->psw.mask & PSW_MASK_64)) {

        vaddr &= 0x7fffffff;

    }



    mmu_translate(env, vaddr, MMU_INST_FETCH, asc, &raddr, &prot, false);



    return raddr;

}
