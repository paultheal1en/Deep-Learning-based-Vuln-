static void tlb_info_32(Monitor *mon, CPUState *env)

{

    int l1, l2;

    uint32_t pgd, pde, pte;



    pgd = env->cr[3] & ~0xfff;

    for(l1 = 0; l1 < 1024; l1++) {

        cpu_physical_memory_read(pgd + l1 * 4, &pde, 4);

        pde = le32_to_cpu(pde);

        if (pde & PG_PRESENT_MASK) {

            if ((pde & PG_PSE_MASK) && (env->cr[4] & CR4_PSE_MASK)) {

                /* 4M pages */

                print_pte(mon, (l1 << 22), pde, ~((1 << 21) - 1));

            } else {

                for(l2 = 0; l2 < 1024; l2++) {

                    cpu_physical_memory_read((pde & ~0xfff) + l2 * 4, &pte, 4);

                    pte = le32_to_cpu(pte);

                    if (pte & PG_PRESENT_MASK) {

                        print_pte(mon, (l1 << 22) + (l2 << 12),

                                  pte & ~PG_PSE_MASK,

                                  ~0xfff);

                    }

                }

            }

        }

    }

}
