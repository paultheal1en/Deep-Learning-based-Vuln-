void cpu_loop(CPUMBState *env)

{

    int trapnr, ret;

    target_siginfo_t info;

    

    while (1) {

        trapnr = cpu_mb_exec (env);

        switch (trapnr) {

        case 0xaa:

            {

                info.si_signo = SIGSEGV;

                info.si_errno = 0;

                /* XXX: check env->error_code */

                info.si_code = TARGET_SEGV_MAPERR;

                info._sifields._sigfault._addr = 0;

                queue_signal(env, info.si_signo, &info);

            }

            break;

	case EXCP_INTERRUPT:

	  /* just indicate that signals should be handled asap */

	  break;

        case EXCP_BREAK:

            /* Return address is 4 bytes after the call.  */

            env->regs[14] += 4;

            ret = do_syscall(env, 

                             env->regs[12], 

                             env->regs[5], 

                             env->regs[6], 

                             env->regs[7], 

                             env->regs[8], 

                             env->regs[9], 

                             env->regs[10],

                             0, 0);

            env->regs[3] = ret;

            env->sregs[SR_PC] = env->regs[14];

            break;

        case EXCP_HW_EXCP:

            env->regs[17] = env->sregs[SR_PC] + 4;

            if (env->iflags & D_FLAG) {

                env->sregs[SR_ESR] |= 1 << 12;

                env->sregs[SR_PC] -= 4;

                /* FIXME: if branch was immed, replay the imm as well.  */

            }



            env->iflags &= ~(IMM_FLAG | D_FLAG);



            switch (env->sregs[SR_ESR] & 31) {

                case ESR_EC_DIVZERO:

                    info.si_signo = SIGFPE;

                    info.si_errno = 0;

                    info.si_code = TARGET_FPE_FLTDIV;

                    info._sifields._sigfault._addr = 0;

                    queue_signal(env, info.si_signo, &info);

                    break;

                case ESR_EC_FPU:

                    info.si_signo = SIGFPE;

                    info.si_errno = 0;

                    if (env->sregs[SR_FSR] & FSR_IO) {

                        info.si_code = TARGET_FPE_FLTINV;

                    }

                    if (env->sregs[SR_FSR] & FSR_DZ) {

                        info.si_code = TARGET_FPE_FLTDIV;

                    }

                    info._sifields._sigfault._addr = 0;

                    queue_signal(env, info.si_signo, &info);

                    break;

                default:

                    printf ("Unhandled hw-exception: 0x%x\n",

                            env->sregs[SR_ESR] & ESR_EC_MASK);

                    cpu_dump_state(env, stderr, fprintf, 0);

                    exit (1);

                    break;

            }

            break;

        case EXCP_DEBUG:

            {

                int sig;



                sig = gdb_handlesig (env, TARGET_SIGTRAP);

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

            cpu_dump_state(env, stderr, fprintf, 0);

            exit (1);

        }

        process_pending_signals (env);

    }

}
