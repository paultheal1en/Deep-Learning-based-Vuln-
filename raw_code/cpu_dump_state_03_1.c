void cpu_dump_state (CPUCRISState *env, FILE *f, fprintf_function cpu_fprintf,

                     int flags)

{

	int i;

	uint32_t srs;



	if (!env || !f)

		return;



	cpu_fprintf(f, "PC=%x CCS=%x btaken=%d btarget=%x\n"

		    "cc_op=%d cc_src=%d cc_dest=%d cc_result=%x cc_mask=%x\n",

		    env->pc, env->pregs[PR_CCS], env->btaken, env->btarget,

		    env->cc_op,

		    env->cc_src, env->cc_dest, env->cc_result, env->cc_mask);





	for (i = 0; i < 16; i++) {

		cpu_fprintf(f, "%s=%8.8x ",regnames[i], env->regs[i]);

		if ((i + 1) % 4 == 0)

			cpu_fprintf(f, "\n");

	}

	cpu_fprintf(f, "\nspecial regs:\n");

	for (i = 0; i < 16; i++) {

		cpu_fprintf(f, "%s=%8.8x ", pregnames[i], env->pregs[i]);

		if ((i + 1) % 4 == 0)

			cpu_fprintf(f, "\n");

	}

	srs = env->pregs[PR_SRS];

	cpu_fprintf(f, "\nsupport function regs bank %x:\n", srs);

	if (srs < 256) {

		for (i = 0; i < 16; i++) {

			cpu_fprintf(f, "s%2.2d=%8.8x ",

				    i, env->sregs[srs][i]);

			if ((i + 1) % 4 == 0)

				cpu_fprintf(f, "\n");

		}

	}

	cpu_fprintf(f, "\n\n");



}
