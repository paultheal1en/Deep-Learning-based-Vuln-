uint32_t HELPER(v7m_mrs)(CPUARMState *env, uint32_t reg)

{

    ARMCPU *cpu = arm_env_get_cpu(env);



    switch (reg) {

    case 0: /* APSR */

        return xpsr_read(env) & 0xf8000000;

    case 1: /* IAPSR */

        return xpsr_read(env) & 0xf80001ff;

    case 2: /* EAPSR */

        return xpsr_read(env) & 0xff00fc00;

    case 3: /* xPSR */

        return xpsr_read(env) & 0xff00fdff;

    case 5: /* IPSR */

        return xpsr_read(env) & 0x000001ff;

    case 6: /* EPSR */

        return xpsr_read(env) & 0x0700fc00;

    case 7: /* IEPSR */

        return xpsr_read(env) & 0x0700edff;

    case 8: /* MSP */

        return env->v7m.current_sp ? env->v7m.other_sp : env->regs[13];

    case 9: /* PSP */

        return env->v7m.current_sp ? env->regs[13] : env->v7m.other_sp;

    case 16: /* PRIMASK */

        return (env->daif & PSTATE_I) != 0;

    case 17: /* BASEPRI */

    case 18: /* BASEPRI_MAX */

        return env->v7m.basepri;

    case 19: /* FAULTMASK */

        return (env->daif & PSTATE_F) != 0;

    case 20: /* CONTROL */

        return env->v7m.control;

    default:

        /* ??? For debugging only.  */

        cpu_abort(CPU(cpu), "Unimplemented system register read (%d)\n", reg);

        return 0;

    }

}
