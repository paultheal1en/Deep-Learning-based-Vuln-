int kvm_arch_on_sigbus_vcpu(CPUState *env, int code, void *addr)

{

#ifdef KVM_CAP_MCE

    ram_addr_t ram_addr;

    target_phys_addr_t paddr;



    if ((env->mcg_cap & MCG_SER_P) && addr

        && (code == BUS_MCEERR_AR || code == BUS_MCEERR_AO)) {

        if (qemu_ram_addr_from_host(addr, &ram_addr) ||

            !kvm_physical_memory_addr_from_ram(env->kvm_state, ram_addr,

                                               &paddr)) {

            fprintf(stderr, "Hardware memory error for memory used by "

                    "QEMU itself instead of guest system!\n");

            /* Hope we are lucky for AO MCE */

            if (code == BUS_MCEERR_AO) {

                return 0;

            } else {

                hardware_memory_error();

            }

        }


        kvm_mce_inject(env, paddr, code);

    } else

#endif /* KVM_CAP_MCE */

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