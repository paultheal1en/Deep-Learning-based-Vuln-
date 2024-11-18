CPUSPARCState *cpu_sparc_init(const char *cpu_model)

{

    SPARCCPU *cpu;

    CPUSPARCState *env;



    cpu = SPARC_CPU(object_new(TYPE_SPARC_CPU));

    env = &cpu->env;



    gen_intermediate_code_init(env);



    if (cpu_sparc_register(env, cpu_model) < 0) {

        object_delete(OBJECT(cpu));

        return NULL;

    }

    qemu_init_vcpu(env);



    return env;

}
