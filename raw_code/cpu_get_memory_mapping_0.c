int cpu_get_memory_mapping(MemoryMappingList *list, CPUArchState *env)

{

    if (!(env->cr[0] & CR0_PG_MASK)) {

        /* paging is disabled */

        return 0;

    }



    if (env->cr[4] & CR4_PAE_MASK) {

#ifdef TARGET_X86_64

        if (env->hflags & HF_LMA_MASK) {

            target_phys_addr_t pml4e_addr;



            pml4e_addr = (env->cr[3] & ~0xfff) & env->a20_mask;

            walk_pml4e(list, pml4e_addr, env->a20_mask);

        } else

#endif

        {

            target_phys_addr_t pdpe_addr;



            pdpe_addr = (env->cr[3] & ~0x1f) & env->a20_mask;

            walk_pdpe2(list, pdpe_addr, env->a20_mask);

        }

    } else {

        target_phys_addr_t pde_addr;

        bool pse;



        pde_addr = (env->cr[3] & ~0xfff) & env->a20_mask;

        pse = !!(env->cr[4] & CR4_PSE_MASK);

        walk_pde2(list, pde_addr, env->a20_mask, pse);

    }



    return 0;

}
