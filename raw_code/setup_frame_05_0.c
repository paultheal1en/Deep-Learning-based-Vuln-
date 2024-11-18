static void setup_frame(int sig, struct emulated_sigaction *ka,

			target_sigset_t *set, CPUX86State *env)

{

	struct sigframe *frame;

	int err = 0;



	frame = get_sigframe(ka, env, sizeof(*frame));



#if 0

	if (!access_ok(VERIFY_WRITE, frame, sizeof(*frame)))

		goto give_sigsegv;

#endif

	err |= __put_user((/*current->exec_domain

		           && current->exec_domain->signal_invmap

		           && sig < 32

		           ? current->exec_domain->signal_invmap[sig]

		           : */ sig),

		          &frame->sig);

	if (err)

		goto give_sigsegv;



	setup_sigcontext(&frame->sc, &frame->fpstate, env, set->sig[0]);

	if (err)

		goto give_sigsegv;



	if (TARGET_NSIG_WORDS > 1) {

		err |= __copy_to_user(frame->extramask, &set->sig[1],

				      sizeof(frame->extramask));

	}

	if (err)

		goto give_sigsegv;



	/* Set up to return from userspace.  If provided, use a stub

	   already in userspace.  */

	if (ka->sa.sa_flags & TARGET_SA_RESTORER) {

		err |= __put_user(ka->sa.sa_restorer, &frame->pretcode);

	} else {

		err |= __put_user(frame->retcode, &frame->pretcode);

		/* This is popl %eax ; movl $,%eax ; int $0x80 */

		err |= __put_user(0xb858, (short *)(frame->retcode+0));

		err |= __put_user(TARGET_NR_sigreturn, (int *)(frame->retcode+2));

		err |= __put_user(0x80cd, (short *)(frame->retcode+6));

	}



	if (err)

		goto give_sigsegv;



	/* Set up registers for signal handler */

	env->regs[R_ESP] = (unsigned long) frame;

	env->eip = (unsigned long) ka->sa._sa_handler;



        cpu_x86_load_seg(env, R_DS, __USER_DS);

        cpu_x86_load_seg(env, R_ES, __USER_DS);

        cpu_x86_load_seg(env, R_SS, __USER_DS);

        cpu_x86_load_seg(env, R_CS, __USER_CS);

	env->eflags &= ~TF_MASK;



	return;



give_sigsegv:

	if (sig == TARGET_SIGSEGV)

		ka->sa._sa_handler = TARGET_SIG_DFL;

	force_sig(TARGET_SIGSEGV /* , current */);

}
