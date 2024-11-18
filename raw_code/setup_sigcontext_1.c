setup_sigcontext(struct target_sigcontext *sc, struct target_fpstate *fpstate,

		 CPUX86State *env, unsigned long mask)

{

	int err = 0;



	err |= __put_user(env->segs[R_GS], (unsigned int *)&sc->gs);

	err |= __put_user(env->segs[R_FS], (unsigned int *)&sc->fs);

	err |= __put_user(env->segs[R_ES], (unsigned int *)&sc->es);

	err |= __put_user(env->segs[R_DS], (unsigned int *)&sc->ds);

	err |= __put_user(env->regs[R_EDI], &sc->edi);

	err |= __put_user(env->regs[R_ESI], &sc->esi);

	err |= __put_user(env->regs[R_EBP], &sc->ebp);

	err |= __put_user(env->regs[R_ESP], &sc->esp);

	err |= __put_user(env->regs[R_EBX], &sc->ebx);

	err |= __put_user(env->regs[R_EDX], &sc->edx);

	err |= __put_user(env->regs[R_ECX], &sc->ecx);

	err |= __put_user(env->regs[R_EAX], &sc->eax);

	err |= __put_user(env->exception_index, &sc->trapno);

	err |= __put_user(env->error_code, &sc->err);

	err |= __put_user(env->eip, &sc->eip);

	err |= __put_user(env->segs[R_CS], (unsigned int *)&sc->cs);

	err |= __put_user(env->eflags, &sc->eflags);

	err |= __put_user(env->regs[R_ESP], &sc->esp_at_signal);

	err |= __put_user(env->segs[R_SS], (unsigned int *)&sc->ss);

#if 0

	tmp = save_i387(fpstate);

	if (tmp < 0)

	  err = 1;

	else

	  err |= __put_user(tmp ? fpstate : NULL, &sc->fpstate);

#else

        err |= __put_user(0, &sc->fpstate);

#endif

	/* non-iBCS2 extensions.. */

	err |= __put_user(mask, &sc->oldmask);

	err |= __put_user(/*current->thread.cr2*/ 0, &sc->cr2);

	return err;

}
