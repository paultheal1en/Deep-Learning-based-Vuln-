int cpu_x86_handle_mmu_fault(CPUX86State *env, uint32_t addr, 

                             int is_write, int is_user, int is_softmmu)

{

    uint8_t *pde_ptr, *pte_ptr;

    uint32_t pde, pte, virt_addr, ptep;

    int error_code, is_dirty, prot, page_size, ret;

    unsigned long paddr, vaddr, page_offset;

    

#if defined(DEBUG_MMU)

    printf("MMU fault: addr=0x%08x w=%d u=%d eip=%08x\n", 

           addr, is_write, is_user, env->eip);

#endif



    if (env->user_mode_only) {

        /* user mode only emulation */

        error_code = 0;

        goto do_fault;

    }



    if (!(env->cr[0] & CR0_PG_MASK)) {

        pte = addr;

        virt_addr = addr & TARGET_PAGE_MASK;

        prot = PROT_READ | PROT_WRITE;

        page_size = 4096;

        goto do_mapping;

    }



    /* page directory entry */

    pde_ptr = phys_ram_base + 

        (((env->cr[3] & ~0xfff) + ((addr >> 20) & ~3)) & a20_mask);

    pde = ldl_raw(pde_ptr);

    if (!(pde & PG_PRESENT_MASK)) {

        error_code = 0;

        goto do_fault;

    }

    /* if PSE bit is set, then we use a 4MB page */

    if ((pde & PG_PSE_MASK) && (env->cr[4] & CR4_PSE_MASK)) {

        if (is_user) {

            if (!(pde & PG_USER_MASK))

                goto do_fault_protect;

            if (is_write && !(pde & PG_RW_MASK))

                goto do_fault_protect;

        } else {

            if ((env->cr[0] & CR0_WP_MASK) && (pde & PG_USER_MASK) &&

                is_write && !(pde & PG_RW_MASK)) 

                goto do_fault_protect;

        }

        is_dirty = is_write && !(pde & PG_DIRTY_MASK);

        if (!(pde & PG_ACCESSED_MASK) || is_dirty) {

            pde |= PG_ACCESSED_MASK;

            if (is_dirty)

                pde |= PG_DIRTY_MASK;

            stl_raw(pde_ptr, pde);

        }

        

        pte = pde & ~0x003ff000; /* align to 4MB */

        ptep = pte;

        page_size = 4096 * 1024;

        virt_addr = addr & ~0x003fffff;

    } else {

        if (!(pde & PG_ACCESSED_MASK)) {

            pde |= PG_ACCESSED_MASK;

            stl_raw(pde_ptr, pde);

        }



        /* page directory entry */

        pte_ptr = phys_ram_base + 

            (((pde & ~0xfff) + ((addr >> 10) & 0xffc)) & a20_mask);

        pte = ldl_raw(pte_ptr);

        if (!(pte & PG_PRESENT_MASK)) {

            error_code = 0;

            goto do_fault;

        }

        /* combine pde and pte user and rw protections */

        ptep = pte & pde;

        if (is_user) {

            if (!(ptep & PG_USER_MASK))

                goto do_fault_protect;

            if (is_write && !(ptep & PG_RW_MASK))

                goto do_fault_protect;

        } else {

            if ((env->cr[0] & CR0_WP_MASK) && (ptep & PG_USER_MASK) &&

                is_write && !(ptep & PG_RW_MASK)) 

                goto do_fault_protect;

        }

        is_dirty = is_write && !(pte & PG_DIRTY_MASK);

        if (!(pte & PG_ACCESSED_MASK) || is_dirty) {

            pte |= PG_ACCESSED_MASK;

            if (is_dirty)

                pte |= PG_DIRTY_MASK;

            stl_raw(pte_ptr, pte);

        }

        page_size = 4096;

        virt_addr = addr & ~0xfff;

    }



    /* the page can be put in the TLB */

    prot = PROT_READ;

    if (pte & PG_DIRTY_MASK) {

        /* only set write access if already dirty... otherwise wait

           for dirty access */

        if (is_user) {

            if (ptep & PG_RW_MASK)

                prot |= PROT_WRITE;

        } else {

            if (!(env->cr[0] & CR0_WP_MASK) || !(ptep & PG_USER_MASK) ||

                (ptep & PG_RW_MASK))

                prot |= PROT_WRITE;

        }

    }



 do_mapping:

    pte = pte & a20_mask;



    /* Even if 4MB pages, we map only one 4KB page in the cache to

       avoid filling it too fast */

    page_offset = (addr & TARGET_PAGE_MASK) & (page_size - 1);

    paddr = (pte & TARGET_PAGE_MASK) + page_offset;

    vaddr = virt_addr + page_offset;

    

    ret = tlb_set_page(env, vaddr, paddr, prot, is_user, is_softmmu);

    return ret;

 do_fault_protect:

    error_code = PG_ERROR_P_MASK;

 do_fault:

    env->cr[2] = addr;

    env->error_code = (is_write << PG_ERROR_W_BIT) | error_code;

    if (is_user)

        env->error_code |= PG_ERROR_U_MASK;

    return 1;

}
