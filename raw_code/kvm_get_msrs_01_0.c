static int kvm_get_msrs(CPUState *env)

{

    struct {

        struct kvm_msrs info;

        struct kvm_msr_entry entries[100];

    } msr_data;

    struct kvm_msr_entry *msrs = msr_data.entries;

    int ret, i, n;



    n = 0;

    msrs[n++].index = MSR_IA32_SYSENTER_CS;

    msrs[n++].index = MSR_IA32_SYSENTER_ESP;

    msrs[n++].index = MSR_IA32_SYSENTER_EIP;

    if (kvm_has_msr_star(env)) {

        msrs[n++].index = MSR_STAR;

    }

    if (kvm_has_msr_hsave_pa(env)) {

        msrs[n++].index = MSR_VM_HSAVE_PA;

    }

    msrs[n++].index = MSR_IA32_TSC;

#ifdef TARGET_X86_64

    if (lm_capable_kernel) {

        msrs[n++].index = MSR_CSTAR;

        msrs[n++].index = MSR_KERNELGSBASE;

        msrs[n++].index = MSR_FMASK;

        msrs[n++].index = MSR_LSTAR;

    }

#endif

    msrs[n++].index = MSR_KVM_SYSTEM_TIME;

    msrs[n++].index = MSR_KVM_WALL_CLOCK;

#ifdef KVM_CAP_ASYNC_PF

    msrs[n++].index = MSR_KVM_ASYNC_PF_EN;

#endif



#ifdef KVM_CAP_MCE

    if (env->mcg_cap) {

        msrs[n++].index = MSR_MCG_STATUS;

        msrs[n++].index = MSR_MCG_CTL;

        for (i = 0; i < (env->mcg_cap & 0xff) * 4; i++) {

            msrs[n++].index = MSR_MC0_CTL + i;

        }

    }

#endif



    msr_data.info.nmsrs = n;

    ret = kvm_vcpu_ioctl(env, KVM_GET_MSRS, &msr_data);

    if (ret < 0) {

        return ret;

    }



    for (i = 0; i < ret; i++) {

        switch (msrs[i].index) {

        case MSR_IA32_SYSENTER_CS:

            env->sysenter_cs = msrs[i].data;

            break;

        case MSR_IA32_SYSENTER_ESP:

            env->sysenter_esp = msrs[i].data;

            break;

        case MSR_IA32_SYSENTER_EIP:

            env->sysenter_eip = msrs[i].data;

            break;

        case MSR_STAR:

            env->star = msrs[i].data;

            break;

#ifdef TARGET_X86_64

        case MSR_CSTAR:

            env->cstar = msrs[i].data;

            break;

        case MSR_KERNELGSBASE:

            env->kernelgsbase = msrs[i].data;

            break;

        case MSR_FMASK:

            env->fmask = msrs[i].data;

            break;

        case MSR_LSTAR:

            env->lstar = msrs[i].data;

            break;

#endif

        case MSR_IA32_TSC:

            env->tsc = msrs[i].data;

            break;

        case MSR_VM_HSAVE_PA:

            env->vm_hsave = msrs[i].data;

            break;

        case MSR_KVM_SYSTEM_TIME:

            env->system_time_msr = msrs[i].data;

            break;

        case MSR_KVM_WALL_CLOCK:

            env->wall_clock_msr = msrs[i].data;

            break;

#ifdef KVM_CAP_MCE

        case MSR_MCG_STATUS:

            env->mcg_status = msrs[i].data;

            break;

        case MSR_MCG_CTL:

            env->mcg_ctl = msrs[i].data;

            break;

#endif

        default:

#ifdef KVM_CAP_MCE

            if (msrs[i].index >= MSR_MC0_CTL &&

                msrs[i].index < MSR_MC0_CTL + (env->mcg_cap & 0xff) * 4) {

                env->mce_banks[msrs[i].index - MSR_MC0_CTL] = msrs[i].data;

            }

#endif

            break;

#ifdef KVM_CAP_ASYNC_PF

        case MSR_KVM_ASYNC_PF_EN:

            env->async_pf_en_msr = msrs[i].data;

            break;

#endif

        }

    }



    return 0;

}
