int xtensa_get_physical_addr(CPUXtensaState *env, bool update_tlb,

        uint32_t vaddr, int is_write, int mmu_idx,

        uint32_t *paddr, uint32_t *page_size, unsigned *access)

{

    if (xtensa_option_enabled(env->config, XTENSA_OPTION_MMU)) {

        return get_physical_addr_mmu(env, update_tlb,

                vaddr, is_write, mmu_idx, paddr, page_size, access, true);

    } else if (xtensa_option_bits_enabled(env->config,

                XTENSA_OPTION_BIT(XTENSA_OPTION_REGION_PROTECTION) |

                XTENSA_OPTION_BIT(XTENSA_OPTION_REGION_TRANSLATION))) {

        return get_physical_addr_region(env, vaddr, is_write, mmu_idx,

                paddr, page_size, access);

    } else {

        *paddr = vaddr;

        *page_size = TARGET_PAGE_SIZE;

        *access = PAGE_READ | PAGE_WRITE | PAGE_EXEC | PAGE_CACHE_BYPASS;

        return 0;

    }

}
