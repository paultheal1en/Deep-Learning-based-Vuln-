static void do_interrupt_v7m(CPUARMState *env)

{

    uint32_t xpsr = xpsr_read(env);

    uint32_t lr;

    uint32_t addr;



    lr = 0xfffffff1;

    if (env->v7m.current_sp)

        lr |= 4;

    if (env->v7m.exception == 0)

        lr |= 8;



    /* For exceptions we just mark as pending on the NVIC, and let that

       handle it.  */

    /* TODO: Need to escalate if the current priority is higher than the

       one we're raising.  */

    switch (env->exception_index) {

    case EXCP_UDEF:

        armv7m_nvic_set_pending(env->nvic, ARMV7M_EXCP_USAGE);

        return;

    case EXCP_SWI:

        env->regs[15] += 2;

        armv7m_nvic_set_pending(env->nvic, ARMV7M_EXCP_SVC);

        return;

    case EXCP_PREFETCH_ABORT:

    case EXCP_DATA_ABORT:

        armv7m_nvic_set_pending(env->nvic, ARMV7M_EXCP_MEM);

        return;

    case EXCP_BKPT:

        if (semihosting_enabled) {

            int nr;

            nr = lduw_code(env->regs[15]) & 0xff;

            if (nr == 0xab) {

                env->regs[15] += 2;

                env->regs[0] = do_arm_semihosting(env);

                return;

            }

        }

        armv7m_nvic_set_pending(env->nvic, ARMV7M_EXCP_DEBUG);

        return;

    case EXCP_IRQ:

        env->v7m.exception = armv7m_nvic_acknowledge_irq(env->nvic);

        break;

    case EXCP_EXCEPTION_EXIT:

        do_v7m_exception_exit(env);

        return;

    default:

        cpu_abort(env, "Unhandled exception 0x%x\n", env->exception_index);

        return; /* Never happens.  Keep compiler happy.  */

    }



    /* Align stack pointer.  */

    /* ??? Should only do this if Configuration Control Register

       STACKALIGN bit is set.  */

    if (env->regs[13] & 4) {

        env->regs[13] -= 4;

        xpsr |= 0x200;

    }

    /* Switch to the handler mode.  */

    v7m_push(env, xpsr);

    v7m_push(env, env->regs[15]);

    v7m_push(env, env->regs[14]);

    v7m_push(env, env->regs[12]);

    v7m_push(env, env->regs[3]);

    v7m_push(env, env->regs[2]);

    v7m_push(env, env->regs[1]);

    v7m_push(env, env->regs[0]);

    switch_v7m_sp(env, 0);

    /* Clear IT bits */

    env->condexec_bits = 0;

    env->regs[14] = lr;

    addr = ldl_phys(env->v7m.vecbase + env->v7m.exception * 4);

    env->regs[15] = addr & 0xfffffffe;

    env->thumb = addr & 1;

}
