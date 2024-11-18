int cpu_cris_handle_mmu_fault (CPUState *env, target_ulong address, int rw,

                               int mmu_idx, int is_softmmu)

{

	struct cris_mmu_result res;

	int prot, miss;

	int r = -1;

	target_ulong phy;



	D(printf ("%s addr=%x pc=%x rw=%x\n", __func__, address, env->pc, rw));

	miss = cris_mmu_translate(&res, env, address & TARGET_PAGE_MASK,

				  rw, mmu_idx, 0);

	if (miss)

	{

		if (env->exception_index == EXCP_BUSFAULT)

			cpu_abort(env,

				  "CRIS: Illegal recursive bus fault."

				 "addr=%x rw=%d\n",

				 address, rw);



		env->pregs[PR_EDA] = address;

		env->exception_index = EXCP_BUSFAULT;

		env->fault_vector = res.bf_vec;

		r = 1;

	}

	else

	{

		/*

		 * Mask off the cache selection bit. The ETRAX busses do not

		 * see the top bit.

		 */

		phy = res.phy & ~0x80000000;

		prot = res.prot;

		tlb_set_page(env, address & TARGET_PAGE_MASK, phy,

                             prot | PAGE_EXEC, mmu_idx, TARGET_PAGE_SIZE);

                r = 0;

	}

	if (r > 0)

		D_LOG("%s returns %d irqreq=%x addr=%x"

			  " phy=%x ismmu=%d vec=%x pc=%x\n", 

			  __func__, r, env->interrupt_request, 

			  address, res.phy, is_softmmu, res.bf_vec, env->pc);

	return r;

}
