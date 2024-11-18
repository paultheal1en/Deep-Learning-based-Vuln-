static TranslationBlock *tb_find_slow(target_ulong pc,

                                      target_ulong cs_base,

                                      uint64_t flags)

{

    TranslationBlock *tb, **ptb1;

    unsigned int h;

    tb_page_addr_t phys_pc, phys_page1, phys_page2;

    target_ulong virt_page2;



    tb_invalidated_flag = 0;



    /* find translated block using physical mappings */

    phys_pc = get_page_addr_code(env, pc);

    phys_page1 = phys_pc & TARGET_PAGE_MASK;

    phys_page2 = -1;

    h = tb_phys_hash_func(phys_pc);

    ptb1 = &tb_phys_hash[h];

    for(;;) {

        tb = *ptb1;

        if (!tb)

            goto not_found;

        if (tb->pc == pc &&

            tb->page_addr[0] == phys_page1 &&

            tb->cs_base == cs_base &&

            tb->flags == flags) {

            /* check next page if needed */

            if (tb->page_addr[1] != -1) {

                virt_page2 = (pc & TARGET_PAGE_MASK) +

                    TARGET_PAGE_SIZE;

                phys_page2 = get_page_addr_code(env, virt_page2);

                if (tb->page_addr[1] == phys_page2)

                    goto found;

            } else {

                goto found;

            }

        }

        ptb1 = &tb->phys_hash_next;

    }

 not_found:

   /* if no translated code available, then translate it now */

    tb = tb_gen_code(env, pc, cs_base, flags, 0);



 found:

    /* Move the last found TB to the head of the list */

    if (likely(*ptb1)) {

        *ptb1 = tb->phys_hash_next;

        tb->phys_hash_next = tb_phys_hash[h];

        tb_phys_hash[h] = tb;

    }

    /* we add the TB in the virtual pc hash table */

    env->tb_jmp_cache[tb_jmp_cache_hash_func(pc)] = tb;

    return tb;

}
