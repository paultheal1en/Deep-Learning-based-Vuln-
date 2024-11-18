static void moxie_cpu_initfn(Object *obj)

{

    CPUState *cs = CPU(obj);

    MoxieCPU *cpu = MOXIE_CPU(obj);

    static int inited;



    cs->env_ptr = &cpu->env;

    cpu_exec_init(cs, &error_abort);



    if (tcg_enabled() && !inited) {

        inited = 1;

        moxie_translate_init();

    }

}
