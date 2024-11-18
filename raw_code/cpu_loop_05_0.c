void cpu_loop(CPUMIPSState *env)

{

    CPUState *cs = CPU(mips_env_get_cpu(env));

    target_siginfo_t info;

    int trapnr;

    abi_long ret;

# ifdef TARGET_ABI_MIPSO32

    unsigned int syscall_num;

# endif



    for(;;) {

        cpu_exec_start(cs);

        trapnr = cpu_mips_exec(env);

        cpu_exec_end(cs);

        switch(trapnr) {

        case EXCP_SYSCALL:

            env->active_tc.PC += 4;

# ifdef TARGET_ABI_MIPSO32

            syscall_num = env->active_tc.gpr[2] - 4000;

            if (syscall_num >= sizeof(mips_syscall_args)) {

                ret = -TARGET_ENOSYS;

            } else {

                int nb_args;

                abi_ulong sp_reg;

                abi_ulong arg5 = 0, arg6 = 0, arg7 = 0, arg8 = 0;



                nb_args = mips_syscall_args[syscall_num];

                sp_reg = env->active_tc.gpr[29];

                switch (nb_args) {

                /* these arguments are taken from the stack */

                case 8:

                    if ((ret = get_user_ual(arg8, sp_reg + 28)) != 0) {

                        goto done_syscall;

                    }

                case 7:

                    if ((ret = get_user_ual(arg7, sp_reg + 24)) != 0) {

                        goto done_syscall;

                    }

                case 6:

                    if ((ret = get_user_ual(arg6, sp_reg + 20)) != 0) {

                        goto done_syscall;

                    }

                case 5:

                    if ((ret = get_user_ual(arg5, sp_reg + 16)) != 0) {

                        goto done_syscall;

                    }

                default:

                    break;

                }

                ret = do_syscall(env, env->active_tc.gpr[2],

                                 env->active_tc.gpr[4],

                                 env->active_tc.gpr[5],

                                 env->active_tc.gpr[6],

                                 env->active_tc.gpr[7],

                                 arg5, arg6, arg7, arg8);

            }

done_syscall:

# else

            ret = do_syscall(env, env->active_tc.gpr[2],

                             env->active_tc.gpr[4], env->active_tc.gpr[5],

                             env->active_tc.gpr[6], env->active_tc.gpr[7],

                             env->active_tc.gpr[8], env->active_tc.gpr[9],

                             env->active_tc.gpr[10], env->active_tc.gpr[11]);

# endif /* O32 */

            if (ret == -TARGET_QEMU_ESIGRETURN) {

                /* Returning from a successful sigreturn syscall.

                   Avoid clobbering register state.  */

                break;

            }

            if ((abi_ulong)ret >= (abi_ulong)-1133) {

                env->active_tc.gpr[7] = 1; /* error flag */

                ret = -ret;

            } else {

                env->active_tc.gpr[7] = 0; /* error flag */

            }

            env->active_tc.gpr[2] = ret;

            break;

        case EXCP_TLBL:

        case EXCP_TLBS:

        case EXCP_AdEL:

        case EXCP_AdES:

            info.si_signo = TARGET_SIGSEGV;

            info.si_errno = 0;

            /* XXX: check env->error_code */

            info.si_code = TARGET_SEGV_MAPERR;

            info._sifields._sigfault._addr = env->CP0_BadVAddr;

            queue_signal(env, info.si_signo, &info);

            break;

        case EXCP_CpU:

        case EXCP_RI:

            info.si_signo = TARGET_SIGILL;

            info.si_errno = 0;

            info.si_code = 0;

            queue_signal(env, info.si_signo, &info);

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

        case EXCP_SC:

            if (do_store_exclusive(env)) {

                info.si_signo = TARGET_SIGSEGV;

                info.si_errno = 0;

                info.si_code = TARGET_SEGV_MAPERR;

                info._sifields._sigfault._addr = env->active_tc.PC;

                queue_signal(env, info.si_signo, &info);

            }

            break;

        case EXCP_DSPDIS:

            info.si_signo = TARGET_SIGILL;

            info.si_errno = 0;

            info.si_code = TARGET_ILL_ILLOPC;

            queue_signal(env, info.si_signo, &info);

            break;

        /* The code below was inspired by the MIPS Linux kernel trap

         * handling code in arch/mips/kernel/traps.c.

         */

        case EXCP_BREAK:

            {

                abi_ulong trap_instr;

                unsigned int code;



                if (env->hflags & MIPS_HFLAG_M16) {

                    if (env->insn_flags & ASE_MICROMIPS) {

                        /* microMIPS mode */

                        abi_ulong instr[2];



                        ret = get_user_u16(instr[0], env->active_tc.PC) ||

                              get_user_u16(instr[1], env->active_tc.PC + 2);



                        trap_instr = (instr[0] << 16) | instr[1];

                    } else {

                        /* MIPS16e mode */

                        ret = get_user_u16(trap_instr, env->active_tc.PC);

                        if (ret != 0) {

                            goto error;

                        }

                        code = (trap_instr >> 6) & 0x3f;

                        if (do_break(env, &info, code) != 0) {

                            goto error;

                        }

                        break;

                    }

                } else {

                    ret = get_user_ual(trap_instr, env->active_tc.PC);

                }



                if (ret != 0) {

                    goto error;

                }



                /* As described in the original Linux kernel code, the

                 * below checks on 'code' are to work around an old

                 * assembly bug.

                 */

                code = ((trap_instr >> 6) & ((1 << 20) - 1));

                if (code >= (1 << 10)) {

                    code >>= 10;

                }



                if (do_break(env, &info, code) != 0) {

                    goto error;

                }

            }

            break;

        case EXCP_TRAP:

            {

                abi_ulong trap_instr;

                unsigned int code = 0;



                if (env->hflags & MIPS_HFLAG_M16) {

                    /* microMIPS mode */

                    abi_ulong instr[2];



                    ret = get_user_u16(instr[0], env->active_tc.PC) ||

                          get_user_u16(instr[1], env->active_tc.PC + 2);



                    trap_instr = (instr[0] << 16) | instr[1];

                } else {

                    ret = get_user_ual(trap_instr, env->active_tc.PC);

                }



                if (ret != 0) {

                    goto error;

                }



                /* The immediate versions don't provide a code.  */

                if (!(trap_instr & 0xFC000000)) {

                    if (env->hflags & MIPS_HFLAG_M16) {

                        /* microMIPS mode */

                        code = ((trap_instr >> 12) & ((1 << 4) - 1));

                    } else {

                        code = ((trap_instr >> 6) & ((1 << 10) - 1));

                    }

                }



                if (do_break(env, &info, code) != 0) {

                    goto error;

                }

            }

            break;

        default:

error:

            fprintf(stderr, "qemu: unhandled CPU exception 0x%x - aborting\n",

                    trapnr);

            cpu_dump_state(cs, stderr, fprintf, 0);

            abort();

        }

        process_pending_signals(env);

    }

}
