hwaddr uc32_cpu_get_phys_page_debug(CPUState *cs, vaddr addr)

{

    UniCore32CPU *cpu = UNICORE32_CPU(cs);



    cpu_abort(CPU(cpu), "%s not supported yet\n", __func__);

    return addr;

}
