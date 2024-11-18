static void setup_frame(int sig, struct target_sigaction *ka,

                        target_sigset_t *set, CPUMBState *env)

{

    struct target_signal_frame *frame;

    abi_ulong frame_addr;

    int err = 0;

    int i;



    frame_addr = get_sigframe(ka, env, sizeof *frame);

    if (!lock_user_struct(VERIFY_WRITE, frame, frame_addr, 0))

        goto badframe;



    /* Save the mask.  */

    __put_user(set->sig[0], &frame->uc.tuc_mcontext.oldmask);

    if (err)

        goto badframe;



    for(i = 1; i < TARGET_NSIG_WORDS; i++) {

        if (__put_user(set->sig[i], &frame->extramask[i - 1]))

            goto badframe;

    }



    setup_sigcontext(&frame->uc.tuc_mcontext, env);



    /* Set up to return from userspace. If provided, use a stub

       already in userspace. */

    /* minus 8 is offset to cater for "rtsd r15,8" offset */

    if (ka->sa_flags & TARGET_SA_RESTORER) {

        env->regs[15] = ((unsigned long)ka->sa_restorer)-8;

    } else {

        uint32_t t;

        /* Note, these encodings are _big endian_! */

        /* addi r12, r0, __NR_sigreturn */

        t = 0x31800000UL | TARGET_NR_sigreturn;

        __put_user(t, frame->tramp + 0);

        /* brki r14, 0x8 */

        t = 0xb9cc0008UL;

        __put_user(t, frame->tramp + 1);



        /* Return from sighandler will jump to the tramp.

           Negative 8 offset because return is rtsd r15, 8 */

        env->regs[15] = ((unsigned long)frame->tramp) - 8;

    }



    if (err)

        goto badframe;



    /* Set up registers for signal handler */

    env->regs[1] = frame_addr;

    /* Signal handler args: */

    env->regs[5] = sig; /* Arg 0: signum */

    env->regs[6] = 0;

    /* arg 1: sigcontext */

    env->regs[7] = frame_addr += offsetof(typeof(*frame), uc);



    /* Offset of 4 to handle microblaze rtid r14, 0 */

    env->sregs[SR_PC] = (unsigned long)ka->_sa_handler;



    unlock_user_struct(frame, frame_addr, 1);

    return;

  badframe:

    unlock_user_struct(frame, frame_addr, 1);

    force_sig(TARGET_SIGSEGV);

}
