void cpu_register_physical_memory_offset(target_phys_addr_t start_addr,

                                         ram_addr_t size,

                                         ram_addr_t phys_offset,

                                         ram_addr_t region_offset)

{

    target_phys_addr_t addr, end_addr;

    PhysPageDesc *p;

    CPUState *env;

    ram_addr_t orig_size = size;

    void *subpage;



#ifdef CONFIG_KQEMU

    /* XXX: should not depend on cpu context */

    env = first_cpu;

    if (env->kqemu_enabled) {

        kqemu_set_phys_mem(start_addr, size, phys_offset);

    }

#endif

    if (kvm_enabled())

        kvm_set_phys_mem(start_addr, size, phys_offset);



    if (phys_offset == IO_MEM_UNASSIGNED) {

        region_offset = start_addr;

    }

    region_offset &= TARGET_PAGE_MASK;

    size = (size + TARGET_PAGE_SIZE - 1) & TARGET_PAGE_MASK;

    end_addr = start_addr + (target_phys_addr_t)size;

    for(addr = start_addr; addr != end_addr; addr += TARGET_PAGE_SIZE) {

        p = phys_page_find(addr >> TARGET_PAGE_BITS);

        if (p && p->phys_offset != IO_MEM_UNASSIGNED) {

            ram_addr_t orig_memory = p->phys_offset;

            target_phys_addr_t start_addr2, end_addr2;

            int need_subpage = 0;



            CHECK_SUBPAGE(addr, start_addr, start_addr2, end_addr, end_addr2,

                          need_subpage);

            if (need_subpage || phys_offset & IO_MEM_SUBWIDTH) {

                if (!(orig_memory & IO_MEM_SUBPAGE)) {

                    subpage = subpage_init((addr & TARGET_PAGE_MASK),

                                           &p->phys_offset, orig_memory,

                                           p->region_offset);

                } else {

                    subpage = io_mem_opaque[(orig_memory & ~TARGET_PAGE_MASK)

                                            >> IO_MEM_SHIFT];

                }

                subpage_register(subpage, start_addr2, end_addr2, phys_offset,

                                 region_offset);

                p->region_offset = 0;

            } else {

                p->phys_offset = phys_offset;

                if ((phys_offset & ~TARGET_PAGE_MASK) <= IO_MEM_ROM ||

                    (phys_offset & IO_MEM_ROMD))

                    phys_offset += TARGET_PAGE_SIZE;

            }

        } else {

            p = phys_page_find_alloc(addr >> TARGET_PAGE_BITS, 1);

            p->phys_offset = phys_offset;

            p->region_offset = region_offset;

            if ((phys_offset & ~TARGET_PAGE_MASK) <= IO_MEM_ROM ||

                (phys_offset & IO_MEM_ROMD)) {

                phys_offset += TARGET_PAGE_SIZE;

            } else {

                target_phys_addr_t start_addr2, end_addr2;

                int need_subpage = 0;



                CHECK_SUBPAGE(addr, start_addr, start_addr2, end_addr,

                              end_addr2, need_subpage);



                if (need_subpage || phys_offset & IO_MEM_SUBWIDTH) {

                    subpage = subpage_init((addr & TARGET_PAGE_MASK),

                                           &p->phys_offset, IO_MEM_UNASSIGNED,

                                           addr & TARGET_PAGE_MASK);

                    subpage_register(subpage, start_addr2, end_addr2,

                                     phys_offset, region_offset);

                    p->region_offset = 0;

                }

            }

        }

        region_offset += TARGET_PAGE_SIZE;

    }



    /* since each CPU stores ram addresses in its TLB cache, we must

       reset the modified entries */

    /* XXX: slow ! */

    for(env = first_cpu; env != NULL; env = env->next_cpu) {

        tlb_flush(env, 1);

    }

}
