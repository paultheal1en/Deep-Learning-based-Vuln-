int kvm_on_sigbus(int code, void *addr)

{

#if defined(KVM_CAP_MCE)

    if ((first_cpu->mcg_cap & MCG_SER_P) && addr && code == BUS_MCEERR_AO) {

        uint64_t status;

        void *vaddr;

        ram_addr_t ram_addr;

        target_phys_addr_t paddr;



        /* Hope we are lucky for AO MCE */

        vaddr = addr;

        if (qemu_ram_addr_from_host(vaddr, &ram_addr) ||

            !kvm_physical_memory_addr_from_ram(first_cpu->kvm_state, ram_addr, &paddr)) {

            fprintf(stderr, "Hardware memory error for memory used by "

                    "QEMU itself instead of guest system!: %p\n", addr);

            return 0;

        }

        status = MCI_STATUS_VAL | MCI_STATUS_UC | MCI_STATUS_EN

            | MCI_STATUS_MISCV | MCI_STATUS_ADDRV | MCI_STATUS_S

            | 0xc0;

        kvm_inject_x86_mce(first_cpu, 9, status,

                           MCG_STATUS_MCIP | MCG_STATUS_RIPV, paddr,

                           (MCM_ADDR_PHYS << 6) | 0xc, 1);

        kvm_mce_broadcast_rest(first_cpu);

    } else

#endif

    {

        if (code == BUS_MCEERR_AO) {

            return 0;

        } else if (code == BUS_MCEERR_AR) {

            hardware_memory_error();

        } else {

            return 1;

        }

    }

    return 0;

}
