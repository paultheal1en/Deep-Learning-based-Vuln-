static int kvm_get_msrs(X86CPU *cpu)

{

    CPUX86State *env = &cpu->env;

    struct kvm_msr_entry *msrs = cpu->kvm_msr_buf->entries;

    int ret, i;

    uint64_t mtrr_top_bits;



    kvm_msr_buf_reset(cpu);



    kvm_msr_entry_add(cpu, MSR_IA32_SYSENTER_CS, 0);

    kvm_msr_entry_add(cpu, MSR_IA32_SYSENTER_ESP, 0);

    kvm_msr_entry_add(cpu, MSR_IA32_SYSENTER_EIP, 0);

    kvm_msr_entry_add(cpu, MSR_PAT, 0);

    if (has_msr_star) {

        kvm_msr_entry_add(cpu, MSR_STAR, 0);

    }

    if (has_msr_hsave_pa) {

        kvm_msr_entry_add(cpu, MSR_VM_HSAVE_PA, 0);

    }

    if (has_msr_tsc_aux) {

        kvm_msr_entry_add(cpu, MSR_TSC_AUX, 0);

    }

    if (has_msr_tsc_adjust) {

        kvm_msr_entry_add(cpu, MSR_TSC_ADJUST, 0);

    }

    if (has_msr_tsc_deadline) {

        kvm_msr_entry_add(cpu, MSR_IA32_TSCDEADLINE, 0);

    }

    if (has_msr_misc_enable) {

        kvm_msr_entry_add(cpu, MSR_IA32_MISC_ENABLE, 0);

    }

    if (has_msr_smbase) {

        kvm_msr_entry_add(cpu, MSR_IA32_SMBASE, 0);

    }

    if (has_msr_feature_control) {

        kvm_msr_entry_add(cpu, MSR_IA32_FEATURE_CONTROL, 0);

    }

    if (has_msr_bndcfgs) {

        kvm_msr_entry_add(cpu, MSR_IA32_BNDCFGS, 0);

    }

    if (has_msr_xss) {

        kvm_msr_entry_add(cpu, MSR_IA32_XSS, 0);

    }





    if (!env->tsc_valid) {

        kvm_msr_entry_add(cpu, MSR_IA32_TSC, 0);

        env->tsc_valid = !runstate_is_running();

    }



#ifdef TARGET_X86_64

    if (lm_capable_kernel) {

        kvm_msr_entry_add(cpu, MSR_CSTAR, 0);

        kvm_msr_entry_add(cpu, MSR_KERNELGSBASE, 0);

        kvm_msr_entry_add(cpu, MSR_FMASK, 0);

        kvm_msr_entry_add(cpu, MSR_LSTAR, 0);

    }

#endif

    kvm_msr_entry_add(cpu, MSR_KVM_SYSTEM_TIME, 0);

    kvm_msr_entry_add(cpu, MSR_KVM_WALL_CLOCK, 0);

    if (has_msr_async_pf_en) {

        kvm_msr_entry_add(cpu, MSR_KVM_ASYNC_PF_EN, 0);

    }

    if (has_msr_pv_eoi_en) {

        kvm_msr_entry_add(cpu, MSR_KVM_PV_EOI_EN, 0);

    }

    if (has_msr_kvm_steal_time) {

        kvm_msr_entry_add(cpu, MSR_KVM_STEAL_TIME, 0);

    }

    if (has_msr_architectural_pmu) {

        kvm_msr_entry_add(cpu, MSR_CORE_PERF_FIXED_CTR_CTRL, 0);

        kvm_msr_entry_add(cpu, MSR_CORE_PERF_GLOBAL_CTRL, 0);

        kvm_msr_entry_add(cpu, MSR_CORE_PERF_GLOBAL_STATUS, 0);

        kvm_msr_entry_add(cpu, MSR_CORE_PERF_GLOBAL_OVF_CTRL, 0);

        for (i = 0; i < MAX_FIXED_COUNTERS; i++) {

            kvm_msr_entry_add(cpu, MSR_CORE_PERF_FIXED_CTR0 + i, 0);

        }

        for (i = 0; i < num_architectural_pmu_counters; i++) {

            kvm_msr_entry_add(cpu, MSR_P6_PERFCTR0 + i, 0);

            kvm_msr_entry_add(cpu, MSR_P6_EVNTSEL0 + i, 0);

        }

    }



    if (env->mcg_cap) {

        kvm_msr_entry_add(cpu, MSR_MCG_STATUS, 0);

        kvm_msr_entry_add(cpu, MSR_MCG_CTL, 0);

        if (has_msr_mcg_ext_ctl) {

            kvm_msr_entry_add(cpu, MSR_MCG_EXT_CTL, 0);

        }

        for (i = 0; i < (env->mcg_cap & 0xff) * 4; i++) {

            kvm_msr_entry_add(cpu, MSR_MC0_CTL + i, 0);

        }

    }



    if (has_msr_hv_hypercall) {

        kvm_msr_entry_add(cpu, HV_X64_MSR_HYPERCALL, 0);

        kvm_msr_entry_add(cpu, HV_X64_MSR_GUEST_OS_ID, 0);

    }

    if (has_msr_hv_vapic) {

        kvm_msr_entry_add(cpu, HV_X64_MSR_APIC_ASSIST_PAGE, 0);

    }

    if (has_msr_hv_tsc) {

        kvm_msr_entry_add(cpu, HV_X64_MSR_REFERENCE_TSC, 0);

    }

    if (has_msr_hv_crash) {

        int j;



        for (j = 0; j < HV_X64_MSR_CRASH_PARAMS; j++) {

            kvm_msr_entry_add(cpu, HV_X64_MSR_CRASH_P0 + j, 0);

        }

    }

    if (has_msr_hv_runtime) {

        kvm_msr_entry_add(cpu, HV_X64_MSR_VP_RUNTIME, 0);

    }

    if (cpu->hyperv_synic) {

        uint32_t msr;



        kvm_msr_entry_add(cpu, HV_X64_MSR_SCONTROL, 0);

        kvm_msr_entry_add(cpu, HV_X64_MSR_SVERSION, 0);

        kvm_msr_entry_add(cpu, HV_X64_MSR_SIEFP, 0);

        kvm_msr_entry_add(cpu, HV_X64_MSR_SIMP, 0);

        for (msr = HV_X64_MSR_SINT0; msr <= HV_X64_MSR_SINT15; msr++) {

            kvm_msr_entry_add(cpu, msr, 0);

        }

    }

    if (has_msr_hv_stimer) {

        uint32_t msr;



        for (msr = HV_X64_MSR_STIMER0_CONFIG; msr <= HV_X64_MSR_STIMER3_COUNT;

             msr++) {

            kvm_msr_entry_add(cpu, msr, 0);

        }

    }

    if (has_msr_mtrr) {

        kvm_msr_entry_add(cpu, MSR_MTRRdefType, 0);

        kvm_msr_entry_add(cpu, MSR_MTRRfix64K_00000, 0);

        kvm_msr_entry_add(cpu, MSR_MTRRfix16K_80000, 0);

        kvm_msr_entry_add(cpu, MSR_MTRRfix16K_A0000, 0);

        kvm_msr_entry_add(cpu, MSR_MTRRfix4K_C0000, 0);

        kvm_msr_entry_add(cpu, MSR_MTRRfix4K_C8000, 0);

        kvm_msr_entry_add(cpu, MSR_MTRRfix4K_D0000, 0);

        kvm_msr_entry_add(cpu, MSR_MTRRfix4K_D8000, 0);

        kvm_msr_entry_add(cpu, MSR_MTRRfix4K_E0000, 0);

        kvm_msr_entry_add(cpu, MSR_MTRRfix4K_E8000, 0);

        kvm_msr_entry_add(cpu, MSR_MTRRfix4K_F0000, 0);

        kvm_msr_entry_add(cpu, MSR_MTRRfix4K_F8000, 0);

        for (i = 0; i < MSR_MTRRcap_VCNT; i++) {

            kvm_msr_entry_add(cpu, MSR_MTRRphysBase(i), 0);

            kvm_msr_entry_add(cpu, MSR_MTRRphysMask(i), 0);

        }

    }



    ret = kvm_vcpu_ioctl(CPU(cpu), KVM_GET_MSRS, cpu->kvm_msr_buf);

    if (ret < 0) {

        return ret;

    }



    assert(ret == cpu->kvm_msr_buf->nmsrs);

    /*

     * MTRR masks: Each mask consists of 5 parts

     * a  10..0: must be zero

     * b  11   : valid bit

     * c n-1.12: actual mask bits

     * d  51..n: reserved must be zero

     * e  63.52: reserved must be zero

     *

     * 'n' is the number of physical bits supported by the CPU and is

     * apparently always <= 52.   We know our 'n' but don't know what

     * the destinations 'n' is; it might be smaller, in which case

     * it masks (c) on loading. It might be larger, in which case

     * we fill 'd' so that d..c is consistent irrespetive of the 'n'

     * we're migrating to.

     */



    if (cpu->fill_mtrr_mask) {

        QEMU_BUILD_BUG_ON(TARGET_PHYS_ADDR_SPACE_BITS > 52);

        assert(cpu->phys_bits <= TARGET_PHYS_ADDR_SPACE_BITS);

        mtrr_top_bits = MAKE_64BIT_MASK(cpu->phys_bits, 52 - cpu->phys_bits);

    } else {

        mtrr_top_bits = 0;

    }



    for (i = 0; i < ret; i++) {

        uint32_t index = msrs[i].index;

        switch (index) {

        case MSR_IA32_SYSENTER_CS:

            env->sysenter_cs = msrs[i].data;

            break;

        case MSR_IA32_SYSENTER_ESP:

            env->sysenter_esp = msrs[i].data;

            break;

        case MSR_IA32_SYSENTER_EIP:

            env->sysenter_eip = msrs[i].data;

            break;

        case MSR_PAT:

            env->pat = msrs[i].data;

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

        case MSR_TSC_AUX:

            env->tsc_aux = msrs[i].data;

            break;

        case MSR_TSC_ADJUST:

            env->tsc_adjust = msrs[i].data;

            break;

        case MSR_IA32_TSCDEADLINE:

            env->tsc_deadline = msrs[i].data;

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

        case MSR_MCG_STATUS:

            env->mcg_status = msrs[i].data;

            break;

        case MSR_MCG_CTL:

            env->mcg_ctl = msrs[i].data;

            break;

        case MSR_MCG_EXT_CTL:

            env->mcg_ext_ctl = msrs[i].data;

            break;

        case MSR_IA32_MISC_ENABLE:

            env->msr_ia32_misc_enable = msrs[i].data;

            break;

        case MSR_IA32_SMBASE:

            env->smbase = msrs[i].data;

            break;

        case MSR_IA32_FEATURE_CONTROL:

            env->msr_ia32_feature_control = msrs[i].data;

            break;

        case MSR_IA32_BNDCFGS:

            env->msr_bndcfgs = msrs[i].data;

            break;

        case MSR_IA32_XSS:

            env->xss = msrs[i].data;

            break;

        default:

            if (msrs[i].index >= MSR_MC0_CTL &&

                msrs[i].index < MSR_MC0_CTL + (env->mcg_cap & 0xff) * 4) {

                env->mce_banks[msrs[i].index - MSR_MC0_CTL] = msrs[i].data;

            }

            break;

        case MSR_KVM_ASYNC_PF_EN:

            env->async_pf_en_msr = msrs[i].data;

            break;

        case MSR_KVM_PV_EOI_EN:

            env->pv_eoi_en_msr = msrs[i].data;

            break;

        case MSR_KVM_STEAL_TIME:

            env->steal_time_msr = msrs[i].data;

            break;

        case MSR_CORE_PERF_FIXED_CTR_CTRL:

            env->msr_fixed_ctr_ctrl = msrs[i].data;

            break;

        case MSR_CORE_PERF_GLOBAL_CTRL:

            env->msr_global_ctrl = msrs[i].data;

            break;

        case MSR_CORE_PERF_GLOBAL_STATUS:

            env->msr_global_status = msrs[i].data;

            break;

        case MSR_CORE_PERF_GLOBAL_OVF_CTRL:

            env->msr_global_ovf_ctrl = msrs[i].data;

            break;

        case MSR_CORE_PERF_FIXED_CTR0 ... MSR_CORE_PERF_FIXED_CTR0 + MAX_FIXED_COUNTERS - 1:

            env->msr_fixed_counters[index - MSR_CORE_PERF_FIXED_CTR0] = msrs[i].data;

            break;

        case MSR_P6_PERFCTR0 ... MSR_P6_PERFCTR0 + MAX_GP_COUNTERS - 1:

            env->msr_gp_counters[index - MSR_P6_PERFCTR0] = msrs[i].data;

            break;

        case MSR_P6_EVNTSEL0 ... MSR_P6_EVNTSEL0 + MAX_GP_COUNTERS - 1:

            env->msr_gp_evtsel[index - MSR_P6_EVNTSEL0] = msrs[i].data;

            break;

        case HV_X64_MSR_HYPERCALL:

            env->msr_hv_hypercall = msrs[i].data;

            break;

        case HV_X64_MSR_GUEST_OS_ID:

            env->msr_hv_guest_os_id = msrs[i].data;

            break;

        case HV_X64_MSR_APIC_ASSIST_PAGE:

            env->msr_hv_vapic = msrs[i].data;

            break;

        case HV_X64_MSR_REFERENCE_TSC:

            env->msr_hv_tsc = msrs[i].data;

            break;

        case HV_X64_MSR_CRASH_P0 ... HV_X64_MSR_CRASH_P4:

            env->msr_hv_crash_params[index - HV_X64_MSR_CRASH_P0] = msrs[i].data;

            break;

        case HV_X64_MSR_VP_RUNTIME:

            env->msr_hv_runtime = msrs[i].data;

            break;

        case HV_X64_MSR_SCONTROL:

            env->msr_hv_synic_control = msrs[i].data;

            break;

        case HV_X64_MSR_SVERSION:

            env->msr_hv_synic_version = msrs[i].data;

            break;

        case HV_X64_MSR_SIEFP:

            env->msr_hv_synic_evt_page = msrs[i].data;

            break;

        case HV_X64_MSR_SIMP:

            env->msr_hv_synic_msg_page = msrs[i].data;

            break;

        case HV_X64_MSR_SINT0 ... HV_X64_MSR_SINT15:

            env->msr_hv_synic_sint[index - HV_X64_MSR_SINT0] = msrs[i].data;

            break;

        case HV_X64_MSR_STIMER0_CONFIG:

        case HV_X64_MSR_STIMER1_CONFIG:

        case HV_X64_MSR_STIMER2_CONFIG:

        case HV_X64_MSR_STIMER3_CONFIG:

            env->msr_hv_stimer_config[(index - HV_X64_MSR_STIMER0_CONFIG)/2] =

                                msrs[i].data;

            break;

        case HV_X64_MSR_STIMER0_COUNT:

        case HV_X64_MSR_STIMER1_COUNT:

        case HV_X64_MSR_STIMER2_COUNT:

        case HV_X64_MSR_STIMER3_COUNT:

            env->msr_hv_stimer_count[(index - HV_X64_MSR_STIMER0_COUNT)/2] =

                                msrs[i].data;

            break;

        case MSR_MTRRdefType:

            env->mtrr_deftype = msrs[i].data;

            break;

        case MSR_MTRRfix64K_00000:

            env->mtrr_fixed[0] = msrs[i].data;

            break;

        case MSR_MTRRfix16K_80000:

            env->mtrr_fixed[1] = msrs[i].data;

            break;

        case MSR_MTRRfix16K_A0000:

            env->mtrr_fixed[2] = msrs[i].data;

            break;

        case MSR_MTRRfix4K_C0000:

            env->mtrr_fixed[3] = msrs[i].data;

            break;

        case MSR_MTRRfix4K_C8000:

            env->mtrr_fixed[4] = msrs[i].data;

            break;

        case MSR_MTRRfix4K_D0000:

            env->mtrr_fixed[5] = msrs[i].data;

            break;

        case MSR_MTRRfix4K_D8000:

            env->mtrr_fixed[6] = msrs[i].data;

            break;

        case MSR_MTRRfix4K_E0000:

            env->mtrr_fixed[7] = msrs[i].data;

            break;

        case MSR_MTRRfix4K_E8000:

            env->mtrr_fixed[8] = msrs[i].data;

            break;

        case MSR_MTRRfix4K_F0000:

            env->mtrr_fixed[9] = msrs[i].data;

            break;

        case MSR_MTRRfix4K_F8000:

            env->mtrr_fixed[10] = msrs[i].data;

            break;

        case MSR_MTRRphysBase(0) ... MSR_MTRRphysMask(MSR_MTRRcap_VCNT - 1):

            if (index & 1) {

                env->mtrr_var[MSR_MTRRphysIndex(index)].mask = msrs[i].data |

                                                               mtrr_top_bits;

            } else {

                env->mtrr_var[MSR_MTRRphysIndex(index)].base = msrs[i].data;

            }

            break;

        }

    }



    return 0;

}
