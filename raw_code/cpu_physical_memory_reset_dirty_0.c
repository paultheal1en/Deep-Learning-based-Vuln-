void cpu_physical_memory_reset_dirty(ram_addr_t start, ram_addr_t end,

                                     int dirty_flags)

{

    CPUState *env;

    unsigned long length, start1;

    int i;



    start &= TARGET_PAGE_MASK;

    end = TARGET_PAGE_ALIGN(end);



    length = end - start;

    if (length == 0)

        return;

    cpu_physical_memory_mask_dirty_range(start, length, dirty_flags);



    /* we modify the TLB cache so that the dirty bit will be set again

       when accessing the range */

    start1 = (unsigned long)qemu_safe_ram_ptr(start);

    /* Chek that we don't span multiple blocks - this breaks the

       address comparisons below.  */

    if ((unsigned long)qemu_safe_ram_ptr(end - 1) - start1

            != (end - 1) - start) {

        abort();

    }



    for(env = first_cpu; env != NULL; env = env->next_cpu) {

        int mmu_idx;

        for (mmu_idx = 0; mmu_idx < NB_MMU_MODES; mmu_idx++) {

            for(i = 0; i < CPU_TLB_SIZE; i++)

                tlb_reset_dirty_range(&env->tlb_table[mmu_idx][i],

                                      start1, length);

        }

    }

}
