void cpu_loop(CPUTLGState *env)

{

    CPUState *cs = CPU(tilegx_env_get_cpu(env));

    int trapnr;



    while (1) {

        cpu_exec_start(cs);

        trapnr = cpu_tilegx_exec(cs);

        cpu_exec_end(cs);

        switch (trapnr) {

        case TILEGX_EXCP_SYSCALL:

            env->regs[TILEGX_R_RE] = do_syscall(env, env->regs[TILEGX_R_NR],

                                                env->regs[0], env->regs[1],

                                                env->regs[2], env->regs[3],

                                                env->regs[4], env->regs[5],

                                                env->regs[6], env->regs[7]);

            env->regs[TILEGX_R_ERR] = TILEGX_IS_ERRNO(env->regs[TILEGX_R_RE])

                                                      ? - env->regs[TILEGX_R_RE]

                                                      : 0;

            break;

        case TILEGX_EXCP_OPCODE_EXCH:

            do_exch(env, true, false);

            break;

        case TILEGX_EXCP_OPCODE_EXCH4:

            do_exch(env, false, false);

            break;

        case TILEGX_EXCP_OPCODE_CMPEXCH:

            do_exch(env, true, true);

            break;

        case TILEGX_EXCP_OPCODE_CMPEXCH4:

            do_exch(env, false, true);

            break;

        case TILEGX_EXCP_OPCODE_FETCHADD:

        case TILEGX_EXCP_OPCODE_FETCHADDGEZ:

        case TILEGX_EXCP_OPCODE_FETCHAND:

        case TILEGX_EXCP_OPCODE_FETCHOR:

            do_fetch(env, trapnr, true);

            break;

        case TILEGX_EXCP_OPCODE_FETCHADD4:

        case TILEGX_EXCP_OPCODE_FETCHADDGEZ4:

        case TILEGX_EXCP_OPCODE_FETCHAND4:

        case TILEGX_EXCP_OPCODE_FETCHOR4:

            do_fetch(env, trapnr, false);

            break;

        case TILEGX_EXCP_SIGNAL:

            do_signal(env, env->signo, env->sigcode);

            break;

        case TILEGX_EXCP_REG_IDN_ACCESS:

        case TILEGX_EXCP_REG_UDN_ACCESS:

            gen_sigill_reg(env);

            break;

        default:

            fprintf(stderr, "trapnr is %d[0x%x].\n", trapnr, trapnr);

            g_assert_not_reached();

        }

        process_pending_signals(env);

    }

}
