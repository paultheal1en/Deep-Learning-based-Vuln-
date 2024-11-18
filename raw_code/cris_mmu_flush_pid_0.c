void cris_mmu_flush_pid(CPUState *env, uint32_t pid)

{

	target_ulong vaddr;

	unsigned int idx;

	uint32_t lo, hi;

	uint32_t tlb_vpn;

	int tlb_pid, tlb_g, tlb_v, tlb_k;

	unsigned int set;

	unsigned int mmu;



	pid &= 0xff;

	for (mmu = 0; mmu < 2; mmu++) {

		for (set = 0; set < 4; set++)

		{

			for (idx = 0; idx < 16; idx++) {

				lo = env->tlbsets[mmu][set][idx].lo;

				hi = env->tlbsets[mmu][set][idx].hi;

				

				tlb_vpn = EXTRACT_FIELD(hi, 13, 31);

				tlb_pid = EXTRACT_FIELD(hi, 0, 7);

				tlb_g  = EXTRACT_FIELD(lo, 4, 4);

				tlb_v = EXTRACT_FIELD(lo, 3, 3);

				tlb_k = EXTRACT_FIELD(lo, 2, 2);



				/* Kernel protected areas need to be flushed

				   as well.  */

				if (tlb_v && !tlb_g && (tlb_pid == pid || tlb_k)) {

					vaddr = tlb_vpn << TARGET_PAGE_BITS;

					D(fprintf(logfile,

						  "flush pid=%x vaddr=%x\n", 

						  pid, vaddr));

					tlb_flush_page(env, vaddr);

				}

			}

		}

	}

}
