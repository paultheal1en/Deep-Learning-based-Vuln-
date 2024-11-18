static void m68k_cpu_initfn(Object *obj)

{

    CPUState *cs = CPU(obj);

    M68kCPU *cpu = M68K_CPU(obj);

    CPUM68KState *env = &cpu->env;

    static bool inited;



    cs->env_ptr = env;

    cpu_exec_init(cs, &error_abort);



    if (tcg_enabled() && !inited) {

        inited = true;

        m68k_tcg_init();

    }

}
