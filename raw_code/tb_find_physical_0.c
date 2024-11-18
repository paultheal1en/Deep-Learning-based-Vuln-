static TranslationBlock *tb_find_physical(CPUState *cpu,

                                          target_ulong pc,

                                          target_ulong cs_base,

                                          uint32_t flags)

{

    CPUArchState *env = (CPUArchState *)cpu->env_ptr;

    TranslationBlock *tb, **tb_hash_head, **ptb1;

    unsigned int h;

    tb_page_addr_t phys_pc, phys_page1;



    tcg_ctx.tb_ctx.tb_invalidated_flag = 0;



    /* find translated block using physical mappings */

    phys_pc = get_page_addr_code(env, pc);

    phys_page1 = phys_pc & TARGET_PAGE_MASK;

    h = tb_phys_hash_func(phys_pc);



    /* Start at head of the hash entry */

    ptb1 = tb_hash_head = &tcg_ctx.tb_ctx.tb_phys_hash[h];

    tb = *ptb1;



    while (tb) {

        if (tb->pc == pc &&

            tb->page_addr[0] == phys_page1 &&

            tb->cs_base == cs_base &&

            tb->flags == flags) {



            if (tb->page_addr[1] == -1) {

                /* done, we have a match */

                break;

            } else {

                /* check next page if needed */

                target_ulong virt_page2 = (pc & TARGET_PAGE_MASK) +

                                          TARGET_PAGE_SIZE;

                tb_page_addr_t phys_page2 = get_page_addr_code(env, virt_page2);



                if (tb->page_addr[1] == phys_page2) {

                    break;

                }

            }

        }



        ptb1 = &tb->phys_hash_next;

        tb = *ptb1;

    }



    if (tb) {

        /* Move the TB to the head of the list */

        *ptb1 = tb->phys_hash_next;

        tb->phys_hash_next = *tb_hash_head;

        *tb_hash_head = tb;

    }

    return tb;

}
