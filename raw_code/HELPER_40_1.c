void HELPER(v7m_msr)(CPUARMState *env, uint32_t reg, uint32_t val)

{

    ARMCPU *cpu = arm_env_get_cpu(env);



    switch (reg) {

    case 0: /* APSR */

        xpsr_write(env, val, 0xf8000000);

        break;

    case 1: /* IAPSR */

        xpsr_write(env, val, 0xf8000000);

        break;

    case 2: /* EAPSR */

        xpsr_write(env, val, 0xfe00fc00);

        break;

    case 3: /* xPSR */

        xpsr_write(env, val, 0xfe00fc00);

        break;

    case 5: /* IPSR */

        /* IPSR bits are readonly.  */

        break;

    case 6: /* EPSR */

        xpsr_write(env, val, 0x0600fc00);

        break;

    case 7: /* IEPSR */

        xpsr_write(env, val, 0x0600fc00);

        break;

    case 8: /* MSP */

        if (env->v7m.current_sp)

            env->v7m.other_sp = val;

        else

            env->regs[13] = val;

        break;

    case 9: /* PSP */

        if (env->v7m.current_sp)

            env->regs[13] = val;

        else

            env->v7m.other_sp = val;

        break;

    case 16: /* PRIMASK */

        if (val & 1) {

            env->daif |= PSTATE_I;

        } else {

            env->daif &= ~PSTATE_I;

        }

        break;

    case 17: /* BASEPRI */

        env->v7m.basepri = val & 0xff;

        break;

    case 18: /* BASEPRI_MAX */

        val &= 0xff;

        if (val != 0 && (val < env->v7m.basepri || env->v7m.basepri == 0))

            env->v7m.basepri = val;

        break;

    case 19: /* FAULTMASK */

        if (val & 1) {

            env->daif |= PSTATE_F;

        } else {

            env->daif &= ~PSTATE_F;

        }

        break;

    case 20: /* CONTROL */

        env->v7m.control = val & 3;

        switch_v7m_sp(env, (val & 2) != 0);

        break;

    default:

        /* ??? For debugging only.  */

        cpu_abort(CPU(cpu), "Unimplemented system register write (%d)\n", reg);

        return;

    }

}
