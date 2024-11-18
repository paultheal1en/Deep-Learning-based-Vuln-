static void lm32_cpu_initfn(Object *obj)

{

    CPUState *cs = CPU(obj);

    LM32CPU *cpu = LM32_CPU(obj);

    CPULM32State *env = &cpu->env;

    static bool tcg_initialized;



    cs->env_ptr = env;

    cpu_exec_init(cs, &error_abort);



    env->flags = 0;



    if (tcg_enabled() && !tcg_initialized) {

        tcg_initialized = true;

        lm32_translate_init();

    }

}
