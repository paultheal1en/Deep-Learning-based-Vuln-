static int cris_mmu_translate_page(struct cris_mmu_result *res,

				   CPUState *env, uint32_t vaddr,

				   int rw, int usermode, int debug)

{

	unsigned int vpage;

	unsigned int idx;

	uint32_t pid, lo, hi;

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

	pid = env->pregs[PR_PID] & 0xff;



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



		tlb_vpn = hi >> 13;

		tlb_pid = EXTRACT_FIELD(hi, 0, 7);

		tlb_g  = EXTRACT_FIELD(lo, 4, 4);



		D_LOG("TLB[%d][%d][%d] v=%x vpage=%x lo=%x hi=%x\n", 

			 mmu, set, idx, tlb_vpn, vpage, lo, hi);

		if ((tlb_g || (tlb_pid == pid))

		    && tlb_vpn == vpage) {

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



		tlb_pfn = EXTRACT_FIELD(lo, 13, 31);

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

		if (cfg_k && tlb_k && usermode) {

			D(printf ("tlb: kernel protected %x lo=%x pc=%x\n", 

				  vaddr, lo, env->pc));

			match = 0;

			res->bf_vec = vect_base + 2;

		} else if (rw == 1 && cfg_w && !tlb_w) {

			D(printf ("tlb: write protected %x lo=%x pc=%x\n", 

				  vaddr, lo, env->pc));

			match = 0;

			/* write accesses never go through the I mmu.  */

			res->bf_vec = vect_base + 3;

		} else if (rw == 2 && cfg_x && !tlb_x) {

			D(printf ("tlb: exec protected %x lo=%x pc=%x\n", 

				 vaddr, lo, env->pc));

			match = 0;

			res->bf_vec = vect_base + 3;

		} else if (cfg_v && !tlb_v) {

			D(printf ("tlb: invalid %x\n", vaddr));

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

	} else {

		/* If refill, provide a randomized set.  */

		set = env->mmu_rand_lfsr & 3;

	}



	if (!match && !debug) {

		cris_mmu_update_rand_lfsr(env);



		/* Compute index.  */

		idx = vpage & 15;



		/* Update RW_MM_TLB_SEL.  */

		env->sregs[SFR_RW_MM_TLB_SEL] = 0;

		set_field(&env->sregs[SFR_RW_MM_TLB_SEL], idx, 0, 4);

		set_field(&env->sregs[SFR_RW_MM_TLB_SEL], set, 4, 2);



		/* Update RW_MM_CAUSE.  */

		set_field(&r_cause, rwcause, 8, 2);

		set_field(&r_cause, vpage, 13, 19);

		set_field(&r_cause, pid, 0, 8);

		env->sregs[SFR_R_MM_CAUSE] = r_cause;

		D(printf("refill vaddr=%x pc=%x\n", vaddr, env->pc));

	}



	D(printf ("%s rw=%d mtch=%d pc=%x va=%x vpn=%x tlbvpn=%x pfn=%x pid=%x"

		  " %x cause=%x sel=%x sp=%x %x %x\n",

		  __func__, rw, match, env->pc,

		  vaddr, vpage,

		  tlb_vpn, tlb_pfn, tlb_pid, 

		  pid,

		  r_cause,

		  env->sregs[SFR_RW_MM_TLB_SEL],

		  env->regs[R_SP], env->pregs[PR_USP], env->ksp));



	res->phy = tlb_pfn << TARGET_PAGE_BITS;

	return !match;

}
