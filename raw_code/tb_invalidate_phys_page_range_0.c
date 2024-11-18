void tb_invalidate_phys_page_range(tb_page_addr_t start, tb_page_addr_t end,

                                   int is_cpu_write_access)

{

    TranslationBlock *tb, *tb_next;

#if defined(TARGET_HAS_PRECISE_SMC)

    CPUState *cpu = current_cpu;

    CPUArchState *env = NULL;

#endif

    tb_page_addr_t tb_start, tb_end;

    PageDesc *p;

    int n;

#ifdef TARGET_HAS_PRECISE_SMC

    int current_tb_not_found = is_cpu_write_access;

    TranslationBlock *current_tb = NULL;

    int current_tb_modified = 0;

    target_ulong current_pc = 0;

    target_ulong current_cs_base = 0;

    uint32_t current_flags = 0;

#endif /* TARGET_HAS_PRECISE_SMC */



    assert_memory_lock();

    assert_tb_locked();



    p = page_find(start >> TARGET_PAGE_BITS);

    if (!p) {

        return;

    }

#if defined(TARGET_HAS_PRECISE_SMC)

    if (cpu != NULL) {

        env = cpu->env_ptr;

    }

#endif



    /* we remove all the TBs in the range [start, end[ */

    /* XXX: see if in some cases it could be faster to invalidate all

       the code */

    tb = p->first_tb;

    while (tb != NULL) {

        n = (uintptr_t)tb & 3;

        tb = (TranslationBlock *)((uintptr_t)tb & ~3);

        tb_next = tb->page_next[n];

        /* NOTE: this is subtle as a TB may span two physical pages */

        if (n == 0) {

            /* NOTE: tb_end may be after the end of the page, but

               it is not a problem */

            tb_start = tb->page_addr[0] + (tb->pc & ~TARGET_PAGE_MASK);

            tb_end = tb_start + tb->size;

        } else {

            tb_start = tb->page_addr[1];

            tb_end = tb_start + ((tb->pc + tb->size) & ~TARGET_PAGE_MASK);

        }

        if (!(tb_end <= start || tb_start >= end)) {

#ifdef TARGET_HAS_PRECISE_SMC

            if (current_tb_not_found) {

                current_tb_not_found = 0;

                current_tb = NULL;

                if (cpu->mem_io_pc) {

                    /* now we have a real cpu fault */

                    current_tb = tb_find_pc(cpu->mem_io_pc);

                }

            }

            if (current_tb == tb &&

                (current_tb->cflags & CF_COUNT_MASK) != 1) {

                /* If we are modifying the current TB, we must stop

                its execution. We could be more precise by checking

                that the modification is after the current PC, but it

                would require a specialized function to partially

                restore the CPU state */



                current_tb_modified = 1;

                cpu_restore_state_from_tb(cpu, current_tb, cpu->mem_io_pc);

                cpu_get_tb_cpu_state(env, &current_pc, &current_cs_base,

                                     &current_flags);

            }

#endif /* TARGET_HAS_PRECISE_SMC */

            tb_phys_invalidate(tb, -1);

        }

        tb = tb_next;

    }

#if !defined(CONFIG_USER_ONLY)

    /* if no code remaining, no need to continue to use slow writes */

    if (!p->first_tb) {

        invalidate_page_bitmap(p);

        tlb_unprotect_code(start);

    }

#endif

#ifdef TARGET_HAS_PRECISE_SMC

    if (current_tb_modified) {

        /* we generate a block containing just the instruction

           modifying the memory. It will ensure that it cannot modify

           itself */

        tb_gen_code(cpu, current_pc, current_cs_base, current_flags,

                    1 | curr_cflags());

        cpu_loop_exit_noexc(cpu);

    }

#endif

}
