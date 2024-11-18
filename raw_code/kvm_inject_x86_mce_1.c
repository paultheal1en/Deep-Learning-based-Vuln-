void kvm_inject_x86_mce(CPUState *cenv, int bank, uint64_t status,

                        uint64_t mcg_status, uint64_t addr, uint64_t misc)

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



    run_on_cpu(cenv, kvm_do_inject_x86_mce, &data);

#endif

}
