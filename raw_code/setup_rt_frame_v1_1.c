static void setup_rt_frame_v1(int usig, struct target_sigaction *ka,

                              target_siginfo_t *info,

                              target_sigset_t *set, CPUARMState *env)

{

	struct rt_sigframe_v1 *frame;

	abi_ulong frame_addr = get_sigframe(ka, env, sizeof(*frame));

	struct target_sigaltstack stack;

	int i;

        abi_ulong info_addr, uc_addr;



	if (!lock_user_struct(VERIFY_WRITE, frame, frame_addr, 0))

            return /* 1 */;



        info_addr = frame_addr + offsetof(struct rt_sigframe_v1, info);

	__put_user(info_addr, &frame->pinfo);

        uc_addr = frame_addr + offsetof(struct rt_sigframe_v1, uc);

	__put_user(uc_addr, &frame->puc);

	copy_siginfo_to_user(&frame->info, info);



	/* Clear all the bits of the ucontext we don't use.  */

	memset(&frame->uc, 0, offsetof(struct target_ucontext_v1, tuc_mcontext));



        memset(&stack, 0, sizeof(stack));

        __put_user(target_sigaltstack_used.ss_sp, &stack.ss_sp);

        __put_user(target_sigaltstack_used.ss_size, &stack.ss_size);

        __put_user(sas_ss_flags(get_sp_from_cpustate(env)), &stack.ss_flags);

        memcpy(&frame->uc.tuc_stack, &stack, sizeof(stack));



	setup_sigcontext(&frame->uc.tuc_mcontext, env, set->sig[0]);

        for(i = 0; i < TARGET_NSIG_WORDS; i++) {

            if (__put_user(set->sig[i], &frame->uc.tuc_sigmask.sig[i]))

                goto end;

        }



        setup_return(env, ka, &frame->retcode, frame_addr, usig,

                     frame_addr + offsetof(struct rt_sigframe_v1, retcode));



        env->regs[1] = info_addr;

        env->regs[2] = uc_addr;



end:

	unlock_user_struct(frame, frame_addr, 1);

}
