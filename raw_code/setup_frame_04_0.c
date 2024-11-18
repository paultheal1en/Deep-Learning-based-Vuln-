static void setup_frame(int sig, struct emulated_sigaction * ka,

                        target_sigset_t *set, CPUState *regs)

{

    struct sigframe *frame;

    abi_ulong frame_addr;

    int i;



    frame_addr = get_sigframe(ka, regs, sizeof(*frame));

    if (!lock_user_struct(VERIFY_WRITE, frame, frame_addr, 0))

	goto give_sigsegv;



    install_sigtramp(frame->sf_code, TARGET_NR_sigreturn);



    if(setup_sigcontext(regs, &frame->sf_sc))

	goto give_sigsegv;



    for(i = 0; i < TARGET_NSIG_WORDS; i++) {

	if(__put_user(set->sig[i], &frame->sf_mask.sig[i]))

	    goto give_sigsegv;

    }



    /*

    * Arguments to signal handler:

    *

    *   a0 = signal number

    *   a1 = 0 (should be cause)

    *   a2 = pointer to struct sigcontext

    *

    * $25 and PC point to the signal handler, $29 points to the

    * struct sigframe.

    */

    regs->gpr[ 4][regs->current_tc] = sig;

    regs->gpr[ 5][regs->current_tc] = 0;

    regs->gpr[ 6][regs->current_tc] = h2g(&frame->sf_sc);

    regs->gpr[29][regs->current_tc] = h2g(frame);

    regs->gpr[31][regs->current_tc] = h2g(frame->sf_code);

    /* The original kernel code sets CP0_EPC to the handler

    * since it returns to userland using eret

    * we cannot do this here, and we must set PC directly */

    regs->PC[regs->current_tc] = regs->gpr[25][regs->current_tc] = ka->sa._sa_handler;

    unlock_user_struct(frame, frame_addr, 1);

    return;



give_sigsegv:

    unlock_user_struct(frame, frame_addr, 1);

    force_sig(TARGET_SIGSEGV/*, current*/);

    return;

}
