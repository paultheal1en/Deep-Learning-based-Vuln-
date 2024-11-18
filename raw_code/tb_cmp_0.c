static bool tb_cmp(const void *p, const void *d)

{

    const TranslationBlock *tb = p;

    const struct tb_desc *desc = d;



    if (tb->pc == desc->pc &&

        tb->page_addr[0] == desc->phys_page1 &&

        tb->cs_base == desc->cs_base &&

        tb->flags == desc->flags &&

        tb->trace_vcpu_dstate == desc->trace_vcpu_dstate &&

        !atomic_read(&tb->invalid)) {

        /* check next page if needed */

        if (tb->page_addr[1] == -1) {

            return true;

        } else {

            tb_page_addr_t phys_page2;

            target_ulong virt_page2;



            virt_page2 = (desc->pc & TARGET_PAGE_MASK) + TARGET_PAGE_SIZE;

            phys_page2 = get_page_addr_code(desc->env, virt_page2);

            if (tb->page_addr[1] == phys_page2) {

                return true;

            }

        }

    }

    return false;

}
