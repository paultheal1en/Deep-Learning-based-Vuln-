static int kvm_put_msrs(CPUState *env, int level)

{

    struct {

        struct kvm_msrs info;

        struct kvm_msr_entry entries[100];

    } msr_data;

    struct kvm_msr_entry *msrs = msr_data.entries;

    int n = 0;



    kvm_msr_entry_set(&msrs[n++], MSR_IA32_SYSENTER_CS, env->sysenter_cs);

    kvm_msr_entry_set(&msrs[n++], MSR_IA32_SYSENTER_ESP, env->sysenter_esp);

    kvm_msr_entry_set(&msrs[n++], MSR_IA32_SYSENTER_EIP, env->sysenter_eip);

    if (kvm_has_msr_star(env))

	kvm_msr_entry_set(&msrs[n++], MSR_STAR, env->star);

    if (kvm_has_msr_hsave_pa(env))

        kvm_msr_entry_set(&msrs[n++], MSR_VM_HSAVE_PA, env->vm_hsave);

#ifdef TARGET_X86_64

    if (lm_capable_kernel) {

        kvm_msr_entry_set(&msrs[n++], MSR_CSTAR, env->cstar);

        kvm_msr_entry_set(&msrs[n++], MSR_KERNELGSBASE, env->kernelgsbase);

        kvm_msr_entry_set(&msrs[n++], MSR_FMASK, env->fmask);

        kvm_msr_entry_set(&msrs[n++], MSR_LSTAR, env->lstar);

    }

#endif

    if (level == KVM_PUT_FULL_STATE) {

        /*

         * KVM is yet unable to synchronize TSC values of multiple VCPUs on

         * writeback. Until this is fixed, we only write the offset to SMP

         * guests after migration, desynchronizing the VCPUs, but avoiding

         * huge jump-backs that would occur without any writeback at all.

         */

        if (smp_cpus == 1 || env->tsc != 0) {

            kvm_msr_entry_set(&msrs[n++], MSR_IA32_TSC, env->tsc);

        }

        kvm_msr_entry_set(&msrs[n++], MSR_KVM_SYSTEM_TIME,

                          env->system_time_msr);

        kvm_msr_entry_set(&msrs[n++], MSR_KVM_WALL_CLOCK, env->wall_clock_msr);

#ifdef KVM_CAP_ASYNC_PF

        kvm_msr_entry_set(&msrs[n++], MSR_KVM_ASYNC_PF_EN, env->async_pf_en_msr);

#endif

    }

#ifdef KVM_CAP_MCE

    if (env->mcg_cap) {

        int i;

        if (level == KVM_PUT_RESET_STATE)

            kvm_msr_entry_set(&msrs[n++], MSR_MCG_STATUS, env->mcg_status);

        else if (level == KVM_PUT_FULL_STATE) {

            kvm_msr_entry_set(&msrs[n++], MSR_MCG_STATUS, env->mcg_status);

            kvm_msr_entry_set(&msrs[n++], MSR_MCG_CTL, env->mcg_ctl);

            for (i = 0; i < (env->mcg_cap & 0xff) * 4; i++)

                kvm_msr_entry_set(&msrs[n++], MSR_MC0_CTL + i, env->mce_banks[i]);

        }

    }

#endif



    msr_data.info.nmsrs = n;



    return kvm_vcpu_ioctl(env, KVM_SET_MSRS, &msr_data);



}
