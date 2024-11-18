int ppc_hash64_handle_mmu_fault(PowerPCCPU *cpu, target_ulong eaddr,

                                int rwx, int mmu_idx)

{

    CPUState *cs = CPU(cpu);

    CPUPPCState *env = &cpu->env;

    ppc_slb_t *slb;

    hwaddr pte_offset;

    ppc_hash_pte64_t pte;

    int pp_prot, amr_prot, prot;

    uint64_t new_pte1;

    const int need_prot[] = {PAGE_READ, PAGE_WRITE, PAGE_EXEC};

    hwaddr raddr;



    assert((rwx == 0) || (rwx == 1) || (rwx == 2));



    /* 1. Handle real mode accesses */

    if (((rwx == 2) && (msr_ir == 0)) || ((rwx != 2) && (msr_dr == 0))) {

        /* Translation is off */

        /* In real mode the top 4 effective address bits are ignored */

        raddr = eaddr & 0x0FFFFFFFFFFFFFFFULL;

        tlb_set_page(cs, eaddr & TARGET_PAGE_MASK, raddr & TARGET_PAGE_MASK,

                     PAGE_READ | PAGE_WRITE | PAGE_EXEC, mmu_idx,

                     TARGET_PAGE_SIZE);

        return 0;

    }



    /* 2. Translation is on, so look up the SLB */

    slb = slb_lookup(cpu, eaddr);



    if (!slb) {

        if (rwx == 2) {

            cs->exception_index = POWERPC_EXCP_ISEG;

            env->error_code = 0;

        } else {

            cs->exception_index = POWERPC_EXCP_DSEG;

            env->error_code = 0;

            env->spr[SPR_DAR] = eaddr;

        }

        return 1;

    }



    /* 3. Check for segment level no-execute violation */

    if ((rwx == 2) && (slb->vsid & SLB_VSID_N)) {

        cs->exception_index = POWERPC_EXCP_ISI;

        env->error_code = 0x10000000;

        return 1;

    }



    /* 4. Locate the PTE in the hash table */

    pte_offset = ppc_hash64_htab_lookup(cpu, slb, eaddr, &pte);

    if (pte_offset == -1) {

        if (rwx == 2) {

            cs->exception_index = POWERPC_EXCP_ISI;

            env->error_code = 0x40000000;

        } else {

            cs->exception_index = POWERPC_EXCP_DSI;

            env->error_code = 0;

            env->spr[SPR_DAR] = eaddr;

            if (rwx == 1) {

                env->spr[SPR_DSISR] = 0x42000000;

            } else {

                env->spr[SPR_DSISR] = 0x40000000;

            }

        }

        return 1;

    }

    qemu_log_mask(CPU_LOG_MMU,

                "found PTE at offset %08" HWADDR_PRIx "\n", pte_offset);



    /* 5. Check access permissions */



    pp_prot = ppc_hash64_pte_prot(cpu, slb, pte);

    amr_prot = ppc_hash64_amr_prot(cpu, pte);

    prot = pp_prot & amr_prot;



    if ((need_prot[rwx] & ~prot) != 0) {

        /* Access right violation */

        qemu_log_mask(CPU_LOG_MMU, "PTE access rejected\n");

        if (rwx == 2) {

            cs->exception_index = POWERPC_EXCP_ISI;

            env->error_code = 0x08000000;

        } else {

            target_ulong dsisr = 0;



            cs->exception_index = POWERPC_EXCP_DSI;

            env->error_code = 0;

            env->spr[SPR_DAR] = eaddr;

            if (need_prot[rwx] & ~pp_prot) {

                dsisr |= 0x08000000;

            }

            if (rwx == 1) {

                dsisr |= 0x02000000;

            }

            if (need_prot[rwx] & ~amr_prot) {

                dsisr |= 0x00200000;

            }

            env->spr[SPR_DSISR] = dsisr;

        }

        return 1;

    }



    qemu_log_mask(CPU_LOG_MMU, "PTE access granted !\n");



    /* 6. Update PTE referenced and changed bits if necessary */



    new_pte1 = pte.pte1 | HPTE64_R_R; /* set referenced bit */

    if (rwx == 1) {

        new_pte1 |= HPTE64_R_C; /* set changed (dirty) bit */

    } else {

        /* Treat the page as read-only for now, so that a later write

         * will pass through this function again to set the C bit */

        prot &= ~PAGE_WRITE;

    }



    if (new_pte1 != pte.pte1) {

        ppc_hash64_store_hpte(cpu, pte_offset / HASH_PTE_SIZE_64,

                              pte.pte0, new_pte1);

    }



    /* 7. Determine the real address from the PTE */



    raddr = deposit64(pte.pte1 & HPTE64_R_RPN, 0, slb->sps->page_shift, eaddr);



    tlb_set_page(cs, eaddr & TARGET_PAGE_MASK, raddr & TARGET_PAGE_MASK,

                 prot, mmu_idx, TARGET_PAGE_SIZE);



    return 0;

}
