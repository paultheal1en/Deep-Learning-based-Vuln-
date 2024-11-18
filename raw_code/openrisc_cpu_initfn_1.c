static void openrisc_cpu_initfn(Object *obj)

{

    CPUState *cs = CPU(obj);

    OpenRISCCPU *cpu = OPENRISC_CPU(obj);

    static int inited;



    cs->env_ptr = &cpu->env;

    cpu_exec_init(cs, &error_abort);



#ifndef CONFIG_USER_ONLY

    cpu_openrisc_mmu_init(cpu);

#endif



    if (tcg_enabled() && !inited) {

        inited = 1;

        openrisc_translate_init();

    }

}
