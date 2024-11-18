void cpu_loop(CPUSH4State *env)

{

    CPUState *cs = CPU(sh_env_get_cpu(env));

    int trapnr, ret;

    target_siginfo_t info;



    while (1) {

        cpu_exec_start(cs);

        trapnr = cpu_sh4_exec(cs);

        cpu_exec_end(cs);



        switch (trapnr) {

        case 0x160:

            env->pc += 2;

            ret = do_syscall(env,

                             env->gregs[3],

                             env->gregs[4],

                             env->gregs[5],

                             env->gregs[6],

                             env->gregs[7],

                             env->gregs[0],

                             env->gregs[1],

                             0, 0);

            env->gregs[0] = ret;

            break;

        case EXCP_INTERRUPT:

            /* just indicate that signals should be handled asap */

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

	case 0xa0:

	case 0xc0:

            info.si_signo = TARGET_SIGSEGV;

            info.si_errno = 0;

            info.si_code = TARGET_SEGV_MAPERR;

            info._sifields._sigfault._addr = env->tea;

            queue_signal(env, info.si_signo, &info);

	    break;



        default:

            printf ("Unhandled trap: 0x%x\n", trapnr);

            cpu_dump_state(cs, stderr, fprintf, 0);

            exit(EXIT_FAILURE);

        }

        process_pending_signals (env);

    }

}
