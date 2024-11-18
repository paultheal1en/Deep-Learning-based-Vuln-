void kvm_inject_x86_mce(CPUState *cenv, int bank, uint64_t status,

                        uint64_t mcg_status, uint64_t addr, uint64_t misc,

                        int flag)

{

#ifdef KVM_CAP_MCE

    struct kvm_x86_mce mce = {

        .bank = bank,

        .status = status,

        .mcg_status = mcg_status,

        .addr = addr,

        .misc = misc,

    };



    if (flag & MCE_BROADCAST) {

        kvm_mce_broadcast_rest(cenv);

    }



    kvm_inject_x86_mce_on(cenv, &mce, flag);

#else /* !KVM_CAP_MCE*/

    if (flag & ABORT_ON_ERROR) {

        abort();

    }

#endif /* !KVM_CAP_MCE*/

}
