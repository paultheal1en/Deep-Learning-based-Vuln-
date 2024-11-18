CPUMIPSState *cpu_mips_init (const char *cpu_model)

{

    CPUMIPSState *env;

    const mips_def_t *def;



    def = cpu_mips_find_by_name(cpu_model);

    if (!def)

        return NULL;

    env = qemu_mallocz(sizeof(CPUMIPSState));

    env->cpu_model = def;



    cpu_exec_init(env);

    env->cpu_model_str = cpu_model;

    mips_tcg_init();

    cpu_reset(env);

    qemu_init_vcpu(env);

    return env;

}
