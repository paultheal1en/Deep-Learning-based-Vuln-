setup_sigcontext(struct target_sigcontext *sc, /*struct _fpstate *fpstate,*/

		 CPUState *env, unsigned long mask)

{

	int err = 0;



	__put_user_error(env->regs[0], &sc->arm_r0, err);

	__put_user_error(env->regs[1], &sc->arm_r1, err);

	__put_user_error(env->regs[2], &sc->arm_r2, err);

	__put_user_error(env->regs[3], &sc->arm_r3, err);

	__put_user_error(env->regs[4], &sc->arm_r4, err);

	__put_user_error(env->regs[5], &sc->arm_r5, err);

	__put_user_error(env->regs[6], &sc->arm_r6, err);

	__put_user_error(env->regs[7], &sc->arm_r7, err);

	__put_user_error(env->regs[8], &sc->arm_r8, err);

	__put_user_error(env->regs[9], &sc->arm_r9, err);

	__put_user_error(env->regs[10], &sc->arm_r10, err);

	__put_user_error(env->regs[11], &sc->arm_fp, err);

	__put_user_error(env->regs[12], &sc->arm_ip, err);

	__put_user_error(env->regs[13], &sc->arm_sp, err);

	__put_user_error(env->regs[14], &sc->arm_lr, err);

	__put_user_error(env->regs[15], &sc->arm_pc, err);

#ifdef TARGET_CONFIG_CPU_32

	__put_user_error(cpsr_read(env), &sc->arm_cpsr, err);

#endif



	__put_user_error(/* current->thread.trap_no */ 0, &sc->trap_no, err);

	__put_user_error(/* current->thread.error_code */ 0, &sc->error_code, err);

	__put_user_error(/* current->thread.address */ 0, &sc->fault_address, err);

	__put_user_error(mask, &sc->oldmask, err);



	return err;

}
