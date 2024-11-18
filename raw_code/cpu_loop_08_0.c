void cpu_loop (CPUState *env)

{

    int trapnr;

    target_siginfo_t info;



    while (1) {

        trapnr = cpu_alpha_exec (env);



        switch (trapnr) {

        case EXCP_RESET:

            fprintf(stderr, "Reset requested. Exit\n");

            exit(1);

            break;

        case EXCP_MCHK:

            fprintf(stderr, "Machine check exception. Exit\n");

            exit(1);

            break;

        case EXCP_ARITH:

            fprintf(stderr, "Arithmetic trap.\n");

            exit(1);

            break;

        case EXCP_HW_INTERRUPT:

            fprintf(stderr, "External interrupt. Exit\n");

            exit(1);

            break;

        case EXCP_DFAULT:

            fprintf(stderr, "MMU data fault\n");

            exit(1);

            break;

        case EXCP_DTB_MISS_PAL:

            fprintf(stderr, "MMU data TLB miss in PALcode\n");

            exit(1);

            break;

        case EXCP_ITB_MISS:

            fprintf(stderr, "MMU instruction TLB miss\n");

            exit(1);

            break;

        case EXCP_ITB_ACV:

            fprintf(stderr, "MMU instruction access violation\n");

            exit(1);

            break;

        case EXCP_DTB_MISS_NATIVE:

            fprintf(stderr, "MMU data TLB miss\n");

            exit(1);

            break;

        case EXCP_UNALIGN:

            fprintf(stderr, "Unaligned access\n");

            exit(1);

            break;

        case EXCP_OPCDEC:

            fprintf(stderr, "Invalid instruction\n");

            exit(1);

            break;

        case EXCP_FEN:

            fprintf(stderr, "Floating-point not allowed\n");

            exit(1);

            break;

        case EXCP_CALL_PAL ... (EXCP_CALL_PALP - 1):

            call_pal(env, (trapnr >> 6) | 0x80);

            break;

        case EXCP_CALL_PALP ... (EXCP_CALL_PALE - 1):

            fprintf(stderr, "Privileged call to PALcode\n");

            exit(1);

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
