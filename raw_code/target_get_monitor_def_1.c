int target_get_monitor_def(CPUState *cs, const char *name, uint64_t *pval)

{

    int i, regnum;

    PowerPCCPU *cpu = POWERPC_CPU(cs);

    CPUPPCState *env = &cpu->env;



    /* General purpose registers */

    if ((tolower(name[0]) == 'r') &&

        ppc_cpu_get_reg_num(name + 1, ARRAY_SIZE(env->gpr), &regnum)) {

        *pval = env->gpr[regnum];

        return 0;

    }



    /* Floating point registers */

    if ((tolower(name[0]) == 'f') &&

        ppc_cpu_get_reg_num(name + 1, ARRAY_SIZE(env->fpr), &regnum)) {

        *pval = env->fpr[regnum];

        return 0;

    }



    /* Special purpose registers */

    for (i = 0; i < ARRAY_SIZE(env->spr_cb); ++i) {

        ppc_spr_t *spr = &env->spr_cb[i];



        if (spr->name && (strcasecmp(name, spr->name) == 0)) {

            *pval = env->spr[i];

            return 0;

        }

    }



    /* Segment registers */

#if !defined(CONFIG_USER_ONLY)

    if ((strncasecmp(name, "sr", 2) == 0) &&

        ppc_cpu_get_reg_num(name + 2, ARRAY_SIZE(env->sr), &regnum)) {

        *pval = env->sr[regnum];

        return 0;

    }

#endif



    return -EINVAL;

}
