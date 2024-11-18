static void mem_info(Monitor *mon)

{

    CPUState *env;

    int l1, l2, prot, last_prot;

    uint32_t pgd, pde, pte, start, end;



    env = mon_get_cpu();

    if (!env)

        return;



    if (!(env->cr[0] & CR0_PG_MASK)) {

        monitor_printf(mon, "PG disabled\n");

        return;

    }

    pgd = env->cr[3] & ~0xfff;

    last_prot = 0;

    start = -1;

    for(l1 = 0; l1 < 1024; l1++) {

        cpu_physical_memory_read(pgd + l1 * 4, (uint8_t *)&pde, 4);

        pde = le32_to_cpu(pde);

        end = l1 << 22;

        if (pde & PG_PRESENT_MASK) {

            if ((pde & PG_PSE_MASK) && (env->cr[4] & CR4_PSE_MASK)) {

                prot = pde & (PG_USER_MASK | PG_RW_MASK | PG_PRESENT_MASK);

                mem_print(mon, &start, &last_prot, end, prot);

            } else {

                for(l2 = 0; l2 < 1024; l2++) {

                    cpu_physical_memory_read((pde & ~0xfff) + l2 * 4,

                                             (uint8_t *)&pte, 4);

                    pte = le32_to_cpu(pte);

                    end = (l1 << 22) + (l2 << 12);

                    if (pte & PG_PRESENT_MASK) {

                        prot = pte & (PG_USER_MASK | PG_RW_MASK | PG_PRESENT_MASK);

                    } else {

                        prot = 0;

                    }

                    mem_print(mon, &start, &last_prot, end, prot);

                }

            }

        } else {

            prot = 0;

            mem_print(mon, &start, &last_prot, end, prot);

        }

    }

}
