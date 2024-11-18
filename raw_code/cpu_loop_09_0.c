void cpu_loop(CPUCRISState *env)

{

    CPUState *cs = CPU(cris_env_get_cpu(env));

    int trapnr, ret;

    target_siginfo_t info;

    

    while (1) {

        cpu_exec_start(cs);

        trapnr = cpu_cris_exec(cs);

        cpu_exec_end(cs);

        switch (trapnr) {

        case 0xaa:

            {

                info.si_signo = TARGET_SIGSEGV;

                info.si_errno = 0;

                /* XXX: check env->error_code */

                info.si_code = TARGET_SEGV_MAPERR;

                info._sifields._sigfault._addr = env->pregs[PR_EDA];

                queue_signal(env, info.si_signo, &info);

            }

            break;

	case EXCP_INTERRUPT:

	  /* just indicate that signals should be handled asap */

	  break;

        case EXCP_BREAK:

            ret = do_syscall(env, 

                             env->regs[9], 

                             env->regs[10], 

                             env->regs[11], 

                             env->regs[12], 

                             env->regs[13], 

                             env->pregs[7], 

                             env->pregs[11],

                             0, 0);

            env->regs[10] = ret;

            break;

        case EXCP_DEBUG:

            {

                int sig;



                sig = gdb_handlesig(cs, TARGET_SIGTRAP);

                if (sig)

                  {

                    info.si_signo = sig;

                    info.si_errno = 0;

                    info.si_code = TARGET_TRAP_BRKPT;

                    queue_signal(env, info.si_signo, &info);

                  }

            }

            break;

        default:

            printf ("Unhandled trap: 0x%x\n", trapnr);

            cpu_dump_state(cs, stderr, fprintf, 0);

            exit(EXIT_FAILURE);

        }

        process_pending_signals (env);

    }

}
