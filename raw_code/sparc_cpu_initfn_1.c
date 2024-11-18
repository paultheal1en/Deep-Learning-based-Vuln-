static void sparc_cpu_initfn(Object *obj)

{

    CPUState *cs = CPU(obj);

    SPARCCPU *cpu = SPARC_CPU(obj);

    CPUSPARCState *env = &cpu->env;



    cs->env_ptr = env;

    cpu_exec_init(cs, &error_abort);



    if (tcg_enabled()) {

        gen_intermediate_code_init(env);

    }

}
