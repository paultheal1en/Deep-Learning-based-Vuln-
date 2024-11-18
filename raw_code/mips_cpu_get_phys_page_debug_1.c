hwaddr mips_cpu_get_phys_page_debug(CPUState *cs, vaddr addr)

{

    MIPSCPU *cpu = MIPS_CPU(cs);

    hwaddr phys_addr;

    int prot;



    if (get_physical_address(&cpu->env, &phys_addr, &prot, addr, 0,

                             ACCESS_INT) != 0) {

        return -1;

    }

    return phys_addr;

}
