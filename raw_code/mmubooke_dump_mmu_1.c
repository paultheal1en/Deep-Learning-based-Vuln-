static void mmubooke_dump_mmu(FILE *f, fprintf_function cpu_fprintf,

                                 CPUPPCState *env)

{

    ppcemb_tlb_t *entry;

    int i;



    if (kvm_enabled() && !env->kvm_sw_tlb) {

        cpu_fprintf(f, "Cannot access KVM TLB\n");

        return;

    }



    cpu_fprintf(f, "\nTLB:\n");

    cpu_fprintf(f, "Effective          Physical           Size PID   Prot     "

                "Attr\n");



    entry = &env->tlb.tlbe[0];

    for (i = 0; i < env->nb_tlb; i++, entry++) {

        hwaddr ea, pa;

        target_ulong mask;

        uint64_t size = (uint64_t)entry->size;

        char size_buf[20];



        /* Check valid flag */

        if (!(entry->prot & PAGE_VALID)) {

            continue;

        }



        mask = ~(entry->size - 1);

        ea = entry->EPN & mask;

        pa = entry->RPN & mask;

#if (TARGET_PHYS_ADDR_SPACE_BITS >= 36)

        /* Extend the physical address to 36 bits */

        pa |= (hwaddr)(entry->RPN & 0xF) << 32;

#endif

        size /= 1024;

        if (size >= 1024) {

            snprintf(size_buf, sizeof(size_buf), "%3" PRId64 "M", size / 1024);

        } else {

            snprintf(size_buf, sizeof(size_buf), "%3" PRId64 "k", size);

        }

        cpu_fprintf(f, "0x%016" PRIx64 " 0x%016" PRIx64 " %s %-5u %08x %08x\n",

                    (uint64_t)ea, (uint64_t)pa, size_buf, (uint32_t)entry->PID,

                    entry->prot, entry->attr);

    }



}
