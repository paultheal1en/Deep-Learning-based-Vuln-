long do_rt_sigreturn(CPUX86State *env)

{

        abi_ulong frame_addr;

	struct rt_sigframe *frame;

        sigset_t set;

	int eax;



        frame_addr = env->regs[R_ESP] - 4;

        if (!lock_user_struct(VERIFY_READ, frame, frame_addr, 1))

                goto badframe;

        target_to_host_sigset(&set, &frame->uc.tuc_sigmask);

        sigprocmask(SIG_SETMASK, &set, NULL);



	if (restore_sigcontext(env, &frame->uc.tuc_mcontext, &eax))

		goto badframe;



	if (do_sigaltstack(frame_addr + offsetof(struct rt_sigframe, uc.tuc_stack), 0, 

                           get_sp_from_cpustate(env)) == -EFAULT)

		goto badframe;



        unlock_user_struct(frame, frame_addr, 0);

	return eax;



badframe:

        unlock_user_struct(frame, frame_addr, 0);

        force_sig(TARGET_SIGSEGV);

	return 0;

}
