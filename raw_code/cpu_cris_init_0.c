CPUCRISState *cpu_cris_init (const char *cpu_model)

{

	CPUCRISState *env;

	static int tcg_initialized = 0;

	int i;



	env = qemu_mallocz(sizeof(CPUCRISState));

	if (!env)

		return NULL;



	cpu_exec_init(env);

	cpu_reset(env);



	if (tcg_initialized)

		return env;



	tcg_initialized = 1;



	cpu_env = tcg_global_reg_new(TCG_TYPE_PTR, TCG_AREG0, "env");

	cc_x = tcg_global_mem_new(TCG_TYPE_TL, TCG_AREG0,

				  offsetof(CPUState, cc_x), "cc_x");

	cc_src = tcg_global_mem_new(TCG_TYPE_TL, TCG_AREG0,

				    offsetof(CPUState, cc_src), "cc_src");

	cc_dest = tcg_global_mem_new(TCG_TYPE_TL, TCG_AREG0,

				     offsetof(CPUState, cc_dest),

				     "cc_dest");

	cc_result = tcg_global_mem_new(TCG_TYPE_TL, TCG_AREG0,

				       offsetof(CPUState, cc_result),

				       "cc_result");

	cc_op = tcg_global_mem_new(TCG_TYPE_TL, TCG_AREG0,

				   offsetof(CPUState, cc_op), "cc_op");

	cc_size = tcg_global_mem_new(TCG_TYPE_TL, TCG_AREG0,

				     offsetof(CPUState, cc_size),

				     "cc_size");

	cc_mask = tcg_global_mem_new(TCG_TYPE_TL, TCG_AREG0,

				     offsetof(CPUState, cc_mask),

				     "cc_mask");



	env_pc = tcg_global_mem_new(TCG_TYPE_TL, TCG_AREG0, 

				    offsetof(CPUState, pc),

				    "pc");

	env_btarget = tcg_global_mem_new(TCG_TYPE_TL, TCG_AREG0,

					 offsetof(CPUState, btarget),

					 "btarget");

	env_btaken = tcg_global_mem_new(TCG_TYPE_TL, TCG_AREG0,

					 offsetof(CPUState, btaken),

					 "btaken");

	for (i = 0; i < 16; i++) {

		cpu_R[i] = tcg_global_mem_new(TCG_TYPE_TL, TCG_AREG0,

					      offsetof(CPUState, regs[i]),

					      regnames[i]);

	}

	for (i = 0; i < 16; i++) {

		cpu_PR[i] = tcg_global_mem_new(TCG_TYPE_TL, TCG_AREG0,

					       offsetof(CPUState, pregs[i]),

					       pregnames[i]);

	}



	TCG_HELPER(helper_raise_exception);

	TCG_HELPER(helper_dump);



	TCG_HELPER(helper_tlb_flush_pid);

	TCG_HELPER(helper_movl_sreg_reg);

	TCG_HELPER(helper_movl_reg_sreg);

	TCG_HELPER(helper_rfe);

	TCG_HELPER(helper_rfn);



	TCG_HELPER(helper_evaluate_flags_muls);

	TCG_HELPER(helper_evaluate_flags_mulu);

	TCG_HELPER(helper_evaluate_flags_mcp);

	TCG_HELPER(helper_evaluate_flags_alu_4);

	TCG_HELPER(helper_evaluate_flags_move_4);

	TCG_HELPER(helper_evaluate_flags_move_2);

	TCG_HELPER(helper_evaluate_flags);

	TCG_HELPER(helper_top_evaluate_flags);

	return env;

}
