static int get_physical_addr_mmu(CPUXtensaState *env, bool update_tlb,

        uint32_t vaddr, int is_write, int mmu_idx,

        uint32_t *paddr, uint32_t *page_size, unsigned *access,

        bool may_lookup_pt)

{

    bool dtlb = is_write != 2;

    uint32_t wi;

    uint32_t ei;

    uint8_t ring;

    uint32_t vpn;

    uint32_t pte;

    const xtensa_tlb_entry *entry = NULL;

    xtensa_tlb_entry tmp_entry;

    int ret = xtensa_tlb_lookup(env, vaddr, dtlb, &wi, &ei, &ring);



    if ((ret == INST_TLB_MISS_CAUSE || ret == LOAD_STORE_TLB_MISS_CAUSE) &&

            may_lookup_pt && get_pte(env, vaddr, &pte) == 0) {

        ring = (pte >> 4) & 0x3;

        wi = 0;

        split_tlb_entry_spec_way(env, vaddr, dtlb, &vpn, wi, &ei);



        if (update_tlb) {

            wi = ++env->autorefill_idx & 0x3;

            xtensa_tlb_set_entry(env, dtlb, wi, ei, vpn, pte);

            env->sregs[EXCVADDR] = vaddr;

            qemu_log("%s: autorefill(%08x): %08x -> %08x\n",

                    __func__, vaddr, vpn, pte);

        } else {

            xtensa_tlb_set_entry_mmu(env, &tmp_entry, dtlb, wi, ei, vpn, pte);

            entry = &tmp_entry;

        }

        ret = 0;

    }

    if (ret != 0) {

        return ret;

    }



    if (entry == NULL) {

        entry = xtensa_tlb_get_entry(env, dtlb, wi, ei);

    }



    if (ring < mmu_idx) {

        return dtlb ?

            LOAD_STORE_PRIVILEGE_CAUSE :

            INST_FETCH_PRIVILEGE_CAUSE;

    }



    *access = mmu_attr_to_access(entry->attr);

    if (!is_access_granted(*access, is_write)) {

        return dtlb ?

            (is_write ?

             STORE_PROHIBITED_CAUSE :

             LOAD_PROHIBITED_CAUSE) :

            INST_FETCH_PROHIBITED_CAUSE;

    }



    *paddr = entry->paddr | (vaddr & ~xtensa_tlb_get_addr_mask(env, dtlb, wi));

    *page_size = ~xtensa_tlb_get_addr_mask(env, dtlb, wi) + 1;



    return 0;

}
