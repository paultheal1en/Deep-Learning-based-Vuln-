CPUSPARCState *cpu_sparc_init(void)

{

    CPUSPARCState *env;



    cpu_exec_init();



    if (!(env = malloc(sizeof(CPUSPARCState))))

	return (NULL);

    memset(env, 0, sizeof(*env));

    env->cwp = 0;

    env->wim = 1;

    env->regwptr = env->regbase + (env->cwp * 16);

    env->access_type = ACCESS_DATA;

#if defined(CONFIG_USER_ONLY)

    env->user_mode_only = 1;

#else

    /* Emulate Prom */

    env->psrs = 1;

    env->pc = 0x4000;

    env->npc = env->pc + 4;

    env->mmuregs[0] = (0x10<<24) | MMU_E; /* Impl 1, ver 0, MMU Enabled */

    env->mmuregs[1] = 0x3000 >> 4; /* MMU Context table */

#endif

    cpu_single_env = env;

    return (env);

}
