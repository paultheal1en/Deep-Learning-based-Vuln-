void kvm_inject_x86_mce(CPUState *cenv, int bank, uint64_t status,

                        uint64_t mcg_status, uint64_t addr, uint64_t misc,

                        int abort_on_error)

{

#ifdef KVM_CAP_MCE

    struct kvm_x86_mce mce = {

        .bank = bank,

        .status = status,

        .mcg_status = mcg_status,

        .addr = addr,

        .misc = misc,

    };

    struct kvm_x86_mce_data data = {

            .env = cenv,

            .mce = &mce,

    };



    if (!cenv->mcg_cap) {

        fprintf(stderr, "MCE support is not enabled!\n");

        return;

    }



    run_on_cpu(cenv, kvm_do_inject_x86_mce, &data);

#else

    if (abort_on_error)

        abort();

#endif

}
