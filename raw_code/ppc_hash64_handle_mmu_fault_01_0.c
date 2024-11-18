int ppc_hash64_handle_mmu_fault(PowerPCCPU *cpu, vaddr eaddr,

                                int rwx, int mmu_idx)

{

    CPUState *cs = CPU(cpu);

    CPUPPCState *env = &cpu->env;

    ppc_slb_t *slb;

    unsigned apshift;

    hwaddr ptex;

    ppc_hash_pte64_t pte;

    int pp_prot, amr_prot, prot;

    uint64_t new_pte1, dsisr;

    const int need_prot[] = {PAGE_READ, PAGE_WRITE, PAGE_EXEC};

    hwaddr raddr;



    assert((rwx == 0) || (rwx == 1) || (rwx == 2));



    /* Note on LPCR usage: 970 uses HID4, but our special variant

     * of store_spr copies relevant fields into env->spr[SPR_LPCR].

     * Similarily we filter unimplemented bits when storing into

     * LPCR depending on the MMU version. This code can thus just

     * use the LPCR "as-is".

     */



    /* 1. Handle real mode accesses */

    if (((rwx == 2) && (msr_ir == 0)) || ((rwx != 2) && (msr_dr == 0))) {

        /* Translation is supposedly "off"  */

        /* In real mode the top 4 effective address bits are (mostly) ignored */

        raddr = eaddr & 0x0FFFFFFFFFFFFFFFULL;



        /* In HV mode, add HRMOR if top EA bit is clear */

        if (msr_hv || !env->has_hv_mode) {

            if (!(eaddr >> 63)) {

                raddr |= env->spr[SPR_HRMOR];

            }

        } else {

            /* Otherwise, check VPM for RMA vs VRMA */

            if (env->spr[SPR_LPCR] & LPCR_VPM0) {

                slb = &env->vrma_slb;

                if (slb->sps) {

                    goto skip_slb_search;

                }

                /* Not much else to do here */

                cs->exception_index = POWERPC_EXCP_MCHECK;

                env->error_code = 0;

                return 1;

            } else if (raddr < env->rmls) {

                /* RMA. Check bounds in RMLS */

                raddr |= env->spr[SPR_RMOR];

            } else {

                /* The access failed, generate the approriate interrupt */

                if (rwx == 2) {

                    ppc_hash64_set_isi(cs, env, 0x08000000);

                } else {

                    dsisr = 0x08000000;

                    if (rwx == 1) {

                        dsisr |= 0x02000000;

                    }

                    ppc_hash64_set_dsi(cs, env, eaddr, dsisr);

                }

                return 1;

            }

        }

        tlb_set_page(cs, eaddr & TARGET_PAGE_MASK, raddr & TARGET_PAGE_MASK,

                     PAGE_READ | PAGE_WRITE | PAGE_EXEC, mmu_idx,

                     TARGET_PAGE_SIZE);

        return 0;

    }



    /* 2. Translation is on, so look up the SLB */

    slb = slb_lookup(cpu, eaddr);

    if (!slb) {

        /* No entry found, check if in-memory segment tables are in use */

        if ((env->mmu_model & POWERPC_MMU_V3) && ppc64_use_proc_tbl(cpu)) {

            /* TODO - Unsupported */

            error_report("Segment Table Support Unimplemented");

            exit(1);

        }

        /* Segment still not found, generate the appropriate interrupt */

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



skip_slb_search:



    /* 3. Check for segment level no-execute violation */

    if ((rwx == 2) && (slb->vsid & SLB_VSID_N)) {

        ppc_hash64_set_isi(cs, env, 0x10000000);

        return 1;

    }



    /* 4. Locate the PTE in the hash table */

    ptex = ppc_hash64_htab_lookup(cpu, slb, eaddr, &pte, &apshift);

    if (ptex == -1) {

        dsisr = 0x40000000;

        if (rwx == 2) {

            ppc_hash64_set_isi(cs, env, dsisr);

        } else {

            if (rwx == 1) {

                dsisr |= 0x02000000;

            }

            ppc_hash64_set_dsi(cs, env, eaddr, dsisr);

        }

        return 1;

    }

    qemu_log_mask(CPU_LOG_MMU,

                  "found PTE at index %08" HWADDR_PRIx "\n", ptex);



    /* 5. Check access permissions */



    pp_prot = ppc_hash64_pte_prot(cpu, slb, pte);

    amr_prot = ppc_hash64_amr_prot(cpu, pte);

    prot = pp_prot & amr_prot;



    if ((need_prot[rwx] & ~prot) != 0) {

        /* Access right violation */

        qemu_log_mask(CPU_LOG_MMU, "PTE access rejected\n");

        if (rwx == 2) {

            ppc_hash64_set_isi(cs, env, 0x08000000);

        } else {

            dsisr = 0;

            if (need_prot[rwx] & ~pp_prot) {

                dsisr |= 0x08000000;

            }

            if (rwx == 1) {

                dsisr |= 0x02000000;

            }

            if (need_prot[rwx] & ~amr_prot) {

                dsisr |= 0x00200000;

            }

            ppc_hash64_set_dsi(cs, env, eaddr, dsisr);

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

        ppc_hash64_store_hpte(cpu, ptex, pte.pte0, new_pte1);

    }



    /* 7. Determine the real address from the PTE */



    raddr = deposit64(pte.pte1 & HPTE64_R_RPN, 0, apshift, eaddr);



    tlb_set_page(cs, eaddr & TARGET_PAGE_MASK, raddr & TARGET_PAGE_MASK,

                 prot, mmu_idx, 1ULL << apshift);



    return 0;

}
