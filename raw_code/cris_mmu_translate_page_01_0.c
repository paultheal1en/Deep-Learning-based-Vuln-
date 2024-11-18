static int cris_mmu_translate_page(struct cris_mmu_result_t *res,

				   CPUState *env, uint32_t vaddr,

				   int rw, int usermode)

{

	unsigned int vpage;

	unsigned int idx;

	uint32_t lo, hi;

	uint32_t tlb_vpn, tlb_pfn = 0;

	int tlb_pid, tlb_g, tlb_v, tlb_k, tlb_w, tlb_x;

	int cfg_v, cfg_k, cfg_w, cfg_x;	

	int set, match = 0;

	uint32_t r_cause;

	uint32_t r_cfg;

	int rwcause;

	int mmu = 1; /* Data mmu is default.  */

	int vect_base;



	r_cause = env->sregs[SFR_R_MM_CAUSE];

	r_cfg = env->sregs[SFR_RW_MM_CFG];



	switch (rw) {

		case 2: rwcause = CRIS_MMU_ERR_EXEC; mmu = 0; break;

		case 1: rwcause = CRIS_MMU_ERR_WRITE; break;

		default:

		case 0: rwcause = CRIS_MMU_ERR_READ; break;

	}



	/* I exception vectors 4 - 7, D 8 - 11.  */

	vect_base = (mmu + 1) * 4;



	vpage = vaddr >> 13;



	/* We know the index which to check on each set.

	   Scan both I and D.  */

#if 0

	for (set = 0; set < 4; set++) {

		for (idx = 0; idx < 16; idx++) {

			lo = env->tlbsets[mmu][set][idx].lo;

			hi = env->tlbsets[mmu][set][idx].hi;

			tlb_vpn = EXTRACT_FIELD(hi, 13, 31);

			tlb_pfn = EXTRACT_FIELD(lo, 13, 31);



			printf ("TLB: [%d][%d] hi=%x lo=%x v=%x p=%x\n", 

					set, idx, hi, lo, tlb_vpn, tlb_pfn);

		}

	}

#endif



	idx = vpage & 15;

	for (set = 0; set < 4; set++)

	{

		lo = env->tlbsets[mmu][set][idx].lo;

		hi = env->tlbsets[mmu][set][idx].hi;



		tlb_vpn = EXTRACT_FIELD(hi, 13, 31);

		tlb_pfn = EXTRACT_FIELD(lo, 13, 31);



		D(printf("TLB[%d][%d] v=%x vpage=%x -> pfn=%x lo=%x hi=%x\n", 

				i, idx, tlb_vpn, vpage, tlb_pfn, lo, hi));

		if (tlb_vpn == vpage) {

			match = 1;

			break;

		}

	}



	res->bf_vec = vect_base;

	if (match) {

		cfg_w  = EXTRACT_FIELD(r_cfg, 19, 19);

		cfg_k  = EXTRACT_FIELD(r_cfg, 18, 18);

		cfg_x  = EXTRACT_FIELD(r_cfg, 17, 17);

		cfg_v  = EXTRACT_FIELD(r_cfg, 16, 16);



		tlb_pid = EXTRACT_FIELD(hi, 0, 7);

		tlb_pfn = EXTRACT_FIELD(lo, 13, 31);

		tlb_g  = EXTRACT_FIELD(lo, 4, 4);

		tlb_v = EXTRACT_FIELD(lo, 3, 3);

		tlb_k = EXTRACT_FIELD(lo, 2, 2);

		tlb_w = EXTRACT_FIELD(lo, 1, 1);

		tlb_x = EXTRACT_FIELD(lo, 0, 0);



		/*

		set_exception_vector(0x04, i_mmu_refill);

		set_exception_vector(0x05, i_mmu_invalid);

		set_exception_vector(0x06, i_mmu_access);

		set_exception_vector(0x07, i_mmu_execute);

		set_exception_vector(0x08, d_mmu_refill);

		set_exception_vector(0x09, d_mmu_invalid);

		set_exception_vector(0x0a, d_mmu_access);

		set_exception_vector(0x0b, d_mmu_write);

		*/

		if (!tlb_g 

		    && tlb_pid != (env->pregs[PR_PID] & 0xff)) {

			D(printf ("tlb: wrong pid %x %x pc=%x\n", 

				 tlb_pid, env->pregs[PR_PID], env->pc));

			match = 0;

			res->bf_vec = vect_base;

		} else if (rw == 1 && cfg_w && !tlb_w) {

			D(printf ("tlb: write protected %x lo=%x\n", 

				vaddr, lo));

			match = 0;

			res->bf_vec = vect_base + 3;

		} else if (cfg_v && !tlb_v) {

			D(printf ("tlb: invalid %x\n", vaddr));

			set_field(&r_cause, rwcause, 8, 9);

			match = 0;

			res->bf_vec = vect_base + 1;

		}



		res->prot = 0;

		if (match) {

			res->prot |= PAGE_READ;

			if (tlb_w)

				res->prot |= PAGE_WRITE;

			if (tlb_x)

				res->prot |= PAGE_EXEC;

		}

		else

			D(dump_tlb(env, mmu));



		env->sregs[SFR_RW_MM_TLB_HI] = hi;

		env->sregs[SFR_RW_MM_TLB_LO] = lo;

	}



	if (!match) {

		/* miss.  */

		idx = vpage & 15;

		set = 0;



		/* Update RW_MM_TLB_SEL.  */

		env->sregs[SFR_RW_MM_TLB_SEL] = 0;

		set_field(&env->sregs[SFR_RW_MM_TLB_SEL], idx, 0, 4);

		set_field(&env->sregs[SFR_RW_MM_TLB_SEL], set, 4, 5);



		/* Update RW_MM_CAUSE.  */

		set_field(&r_cause, rwcause, 8, 2);

		set_field(&r_cause, vpage, 13, 19);

		set_field(&r_cause, env->pregs[PR_PID], 0, 8);

		env->sregs[SFR_R_MM_CAUSE] = r_cause;

		D(printf("refill vaddr=%x pc=%x\n", vaddr, env->pc));

	}





	D(printf ("%s rw=%d mtch=%d pc=%x va=%x vpn=%x tlbvpn=%x pfn=%x pid=%x"

		  " %x cause=%x sel=%x sp=%x %x %x\n",

		  __func__, rw, match, env->pc,

		  vaddr, vpage,

		  tlb_vpn, tlb_pfn, tlb_pid, 

		  env->pregs[PR_PID],

		  r_cause,

		  env->sregs[SFR_RW_MM_TLB_SEL],

		  env->regs[R_SP], env->pregs[PR_USP], env->ksp));



	res->pfn = tlb_pfn;

	return !match;

}
