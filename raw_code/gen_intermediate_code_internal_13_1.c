gen_intermediate_code_internal(CPUState *env, TranslationBlock *tb,

                               int search_pc)

{

	uint16_t *gen_opc_end;

   	uint32_t pc_start;

	unsigned int insn_len;

	int j, lj;

	struct DisasContext *dc = &ctx;

	uint32_t next_page_start;



	pc_start = tb->pc;

	dc->env = env;

	dc->tb = tb;



	gen_opc_end = gen_opc_buf + OPC_MAX_SIZE;



	dc->is_jmp = DISAS_NEXT;

	dc->pc = pc_start;

	dc->singlestep_enabled = env->singlestep_enabled;

	dc->flagx_live = 0;

	dc->flags_x = 0;

	next_page_start = (pc_start & TARGET_PAGE_MASK) + TARGET_PAGE_SIZE;

	lj = -1;

	do

	{

		check_breakpoint(env, dc);

		if (dc->is_jmp == DISAS_JUMP)

			goto done;



		if (search_pc) {

			j = gen_opc_ptr - gen_opc_buf;

			if (lj < j) {

				lj++;

				while (lj < j)

					gen_opc_instr_start[lj++] = 0;

			}

			gen_opc_pc[lj] = dc->pc;

			gen_opc_instr_start[lj] = 1;

		}



		insn_len = cris_decoder(dc);

		STATS(gen_op_exec_insn());

		dc->pc += insn_len;

		if (!dc->flagx_live

		    || (dc->flagx_live &&

			!(dc->cc_op == CC_OP_FLAGS && dc->flags_x))) {

			gen_movl_T0_preg[PR_CCS]();

			gen_op_andl_T0_im(~X_FLAG);

			gen_movl_preg_T0[PR_CCS]();

			dc->flagx_live = 1;

			dc->flags_x = 0;

		}



		/* Check for delayed branches here. If we do it before

		   actually genereating any host code, the simulator will just

		   loop doing nothing for on this program location.  */

		if (dc->delayed_branch) {

			dc->delayed_branch--;

			if (dc->delayed_branch == 0)

			{

				if (dc->bcc == CC_A) {

					gen_op_jmp1 ();

					dc->is_jmp = DISAS_UPDATE;

				}

				else {

					/* Conditional jmp.  */

					gen_op_cc_jmp (dc->delayed_pc, dc->pc);

					dc->is_jmp = DISAS_UPDATE;

				}

			}

		}



		if (env->singlestep_enabled)

			break;

	} while (!dc->is_jmp && gen_opc_ptr < gen_opc_end

		 && dc->pc < next_page_start);



	if (!dc->is_jmp) {

		gen_op_movl_T0_im((long)dc->pc);

		gen_op_movl_pc_T0();

	}



	cris_evaluate_flags (dc);

  done:

	if (__builtin_expect(env->singlestep_enabled, 0)) {

		gen_op_debug();

	} else {

		switch(dc->is_jmp) {

			case DISAS_NEXT:

				gen_goto_tb(dc, 1, dc->pc);

				break;

			default:

			case DISAS_JUMP:

			case DISAS_UPDATE:

				/* indicate that the hash table must be used

				   to find the next TB */

				tcg_gen_exit_tb(0);

				break;

			case DISAS_TB_JUMP:

				/* nothing more to generate */

				break;

		}

	}

	*gen_opc_ptr = INDEX_op_end;

	if (search_pc) {

		j = gen_opc_ptr - gen_opc_buf;

		lj++;

		while (lj <= j)

			gen_opc_instr_start[lj++] = 0;

	} else {

		tb->size = dc->pc - pc_start;

	}



#ifdef DEBUG_DISAS

	if (loglevel & CPU_LOG_TB_IN_ASM) {

		fprintf(logfile, "--------------\n");

		fprintf(logfile, "IN: %s\n", lookup_symbol(pc_start));

		target_disas(logfile, pc_start, dc->pc + 4 - pc_start, 0);

		fprintf(logfile, "\n");

	}

#endif

	return 0;

}
