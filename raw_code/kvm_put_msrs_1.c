static int kvm_put_msrs(X86CPU *cpu, int level)

{

    CPUX86State *env = &cpu->env;

    int i;

    int ret;



    kvm_msr_buf_reset(cpu);



    kvm_msr_entry_add(cpu, MSR_IA32_SYSENTER_CS, env->sysenter_cs);

    kvm_msr_entry_add(cpu, MSR_IA32_SYSENTER_ESP, env->sysenter_esp);

    kvm_msr_entry_add(cpu, MSR_IA32_SYSENTER_EIP, env->sysenter_eip);

    kvm_msr_entry_add(cpu, MSR_PAT, env->pat);

    if (has_msr_star) {

        kvm_msr_entry_add(cpu, MSR_STAR, env->star);

    }

    if (has_msr_hsave_pa) {

        kvm_msr_entry_add(cpu, MSR_VM_HSAVE_PA, env->vm_hsave);

    }

    if (has_msr_tsc_aux) {

        kvm_msr_entry_add(cpu, MSR_TSC_AUX, env->tsc_aux);

    }

    if (has_msr_tsc_adjust) {

        kvm_msr_entry_add(cpu, MSR_TSC_ADJUST, env->tsc_adjust);

    }

    if (has_msr_misc_enable) {

        kvm_msr_entry_add(cpu, MSR_IA32_MISC_ENABLE,

                          env->msr_ia32_misc_enable);

    }

    if (has_msr_smbase) {

        kvm_msr_entry_add(cpu, MSR_IA32_SMBASE, env->smbase);

    }

    if (has_msr_bndcfgs) {

        kvm_msr_entry_add(cpu, MSR_IA32_BNDCFGS, env->msr_bndcfgs);

    }

    if (has_msr_xss) {

        kvm_msr_entry_add(cpu, MSR_IA32_XSS, env->xss);

    }

#ifdef TARGET_X86_64

    if (lm_capable_kernel) {

        kvm_msr_entry_add(cpu, MSR_CSTAR, env->cstar);

        kvm_msr_entry_add(cpu, MSR_KERNELGSBASE, env->kernelgsbase);

        kvm_msr_entry_add(cpu, MSR_FMASK, env->fmask);

        kvm_msr_entry_add(cpu, MSR_LSTAR, env->lstar);

    }

#endif

    /*

     * The following MSRs have side effects on the guest or are too heavy

     * for normal writeback. Limit them to reset or full state updates.

     */

    if (level >= KVM_PUT_RESET_STATE) {

        kvm_msr_entry_add(cpu, MSR_IA32_TSC, env->tsc);

        kvm_msr_entry_add(cpu, MSR_KVM_SYSTEM_TIME, env->system_time_msr);

        kvm_msr_entry_add(cpu, MSR_KVM_WALL_CLOCK, env->wall_clock_msr);

        if (env->features[FEAT_KVM] & (1 << KVM_FEATURE_ASYNC_PF)) {

            kvm_msr_entry_add(cpu, MSR_KVM_ASYNC_PF_EN, env->async_pf_en_msr);

        }

        if (env->features[FEAT_KVM] & (1 << KVM_FEATURE_PV_EOI)) {

            kvm_msr_entry_add(cpu, MSR_KVM_PV_EOI_EN, env->pv_eoi_en_msr);

        }

        if (env->features[FEAT_KVM] & (1 << KVM_FEATURE_STEAL_TIME)) {

            kvm_msr_entry_add(cpu, MSR_KVM_STEAL_TIME, env->steal_time_msr);

        }

        if (has_msr_architectural_pmu) {

            /* Stop the counter.  */

            kvm_msr_entry_add(cpu, MSR_CORE_PERF_FIXED_CTR_CTRL, 0);

            kvm_msr_entry_add(cpu, MSR_CORE_PERF_GLOBAL_CTRL, 0);



            /* Set the counter values.  */

            for (i = 0; i < MAX_FIXED_COUNTERS; i++) {

                kvm_msr_entry_add(cpu, MSR_CORE_PERF_FIXED_CTR0 + i,

                                  env->msr_fixed_counters[i]);

            }

            for (i = 0; i < num_architectural_pmu_counters; i++) {

                kvm_msr_entry_add(cpu, MSR_P6_PERFCTR0 + i,

                                  env->msr_gp_counters[i]);

                kvm_msr_entry_add(cpu, MSR_P6_EVNTSEL0 + i,

                                  env->msr_gp_evtsel[i]);

            }

            kvm_msr_entry_add(cpu, MSR_CORE_PERF_GLOBAL_STATUS,

                              env->msr_global_status);

            kvm_msr_entry_add(cpu, MSR_CORE_PERF_GLOBAL_OVF_CTRL,

                              env->msr_global_ovf_ctrl);



            /* Now start the PMU.  */

            kvm_msr_entry_add(cpu, MSR_CORE_PERF_FIXED_CTR_CTRL,

                              env->msr_fixed_ctr_ctrl);

            kvm_msr_entry_add(cpu, MSR_CORE_PERF_GLOBAL_CTRL,

                              env->msr_global_ctrl);

        }

        /*

         * Hyper-V partition-wide MSRs: to avoid clearing them on cpu hot-add,

         * only sync them to KVM on the first cpu

         */

        if (current_cpu == first_cpu) {

            if (has_msr_hv_hypercall) {

                kvm_msr_entry_add(cpu, HV_X64_MSR_GUEST_OS_ID,

                                  env->msr_hv_guest_os_id);

                kvm_msr_entry_add(cpu, HV_X64_MSR_HYPERCALL,

                                  env->msr_hv_hypercall);

            }

            if (cpu->hyperv_time) {

                kvm_msr_entry_add(cpu, HV_X64_MSR_REFERENCE_TSC,

                                  env->msr_hv_tsc);

            }

        }

        if (cpu->hyperv_vapic) {

            kvm_msr_entry_add(cpu, HV_X64_MSR_APIC_ASSIST_PAGE,

                              env->msr_hv_vapic);

        }

        if (has_msr_hv_crash) {

            int j;



            for (j = 0; j < HV_CRASH_PARAMS; j++)

                kvm_msr_entry_add(cpu, HV_X64_MSR_CRASH_P0 + j,

                                  env->msr_hv_crash_params[j]);



            kvm_msr_entry_add(cpu, HV_X64_MSR_CRASH_CTL, HV_CRASH_CTL_NOTIFY);

        }

        if (has_msr_hv_runtime) {

            kvm_msr_entry_add(cpu, HV_X64_MSR_VP_RUNTIME, env->msr_hv_runtime);

        }

        if (cpu->hyperv_synic) {

            int j;



            kvm_msr_entry_add(cpu, HV_X64_MSR_SVERSION, HV_SYNIC_VERSION);



            kvm_msr_entry_add(cpu, HV_X64_MSR_SCONTROL,

                              env->msr_hv_synic_control);

            kvm_msr_entry_add(cpu, HV_X64_MSR_SIEFP,

                              env->msr_hv_synic_evt_page);

            kvm_msr_entry_add(cpu, HV_X64_MSR_SIMP,

                              env->msr_hv_synic_msg_page);



            for (j = 0; j < ARRAY_SIZE(env->msr_hv_synic_sint); j++) {

                kvm_msr_entry_add(cpu, HV_X64_MSR_SINT0 + j,

                                  env->msr_hv_synic_sint[j]);

            }

        }

        if (has_msr_hv_stimer) {

            int j;



            for (j = 0; j < ARRAY_SIZE(env->msr_hv_stimer_config); j++) {

                kvm_msr_entry_add(cpu, HV_X64_MSR_STIMER0_CONFIG + j * 2,

                                env->msr_hv_stimer_config[j]);

            }



            for (j = 0; j < ARRAY_SIZE(env->msr_hv_stimer_count); j++) {

                kvm_msr_entry_add(cpu, HV_X64_MSR_STIMER0_COUNT + j * 2,

                                env->msr_hv_stimer_count[j]);

            }

        }

        if (env->features[FEAT_1_EDX] & CPUID_MTRR) {

            uint64_t phys_mask = MAKE_64BIT_MASK(0, cpu->phys_bits);



            kvm_msr_entry_add(cpu, MSR_MTRRdefType, env->mtrr_deftype);

            kvm_msr_entry_add(cpu, MSR_MTRRfix64K_00000, env->mtrr_fixed[0]);

            kvm_msr_entry_add(cpu, MSR_MTRRfix16K_80000, env->mtrr_fixed[1]);

            kvm_msr_entry_add(cpu, MSR_MTRRfix16K_A0000, env->mtrr_fixed[2]);

            kvm_msr_entry_add(cpu, MSR_MTRRfix4K_C0000, env->mtrr_fixed[3]);

            kvm_msr_entry_add(cpu, MSR_MTRRfix4K_C8000, env->mtrr_fixed[4]);

            kvm_msr_entry_add(cpu, MSR_MTRRfix4K_D0000, env->mtrr_fixed[5]);

            kvm_msr_entry_add(cpu, MSR_MTRRfix4K_D8000, env->mtrr_fixed[6]);

            kvm_msr_entry_add(cpu, MSR_MTRRfix4K_E0000, env->mtrr_fixed[7]);

            kvm_msr_entry_add(cpu, MSR_MTRRfix4K_E8000, env->mtrr_fixed[8]);

            kvm_msr_entry_add(cpu, MSR_MTRRfix4K_F0000, env->mtrr_fixed[9]);

            kvm_msr_entry_add(cpu, MSR_MTRRfix4K_F8000, env->mtrr_fixed[10]);

            for (i = 0; i < MSR_MTRRcap_VCNT; i++) {

                /* The CPU GPs if we write to a bit above the physical limit of

                 * the host CPU (and KVM emulates that)

                 */

                uint64_t mask = env->mtrr_var[i].mask;

                mask &= phys_mask;



                kvm_msr_entry_add(cpu, MSR_MTRRphysBase(i),

                                  env->mtrr_var[i].base);

                kvm_msr_entry_add(cpu, MSR_MTRRphysMask(i), mask);

            }

        }



        /* Note: MSR_IA32_FEATURE_CONTROL is written separately, see

         *       kvm_put_msr_feature_control. */

    }

    if (env->mcg_cap) {

        int i;



        kvm_msr_entry_add(cpu, MSR_MCG_STATUS, env->mcg_status);

        kvm_msr_entry_add(cpu, MSR_MCG_CTL, env->mcg_ctl);

        if (has_msr_mcg_ext_ctl) {

            kvm_msr_entry_add(cpu, MSR_MCG_EXT_CTL, env->mcg_ext_ctl);

        }

        for (i = 0; i < (env->mcg_cap & 0xff) * 4; i++) {

            kvm_msr_entry_add(cpu, MSR_MC0_CTL + i, env->mce_banks[i]);

        }

    }



    ret = kvm_vcpu_ioctl(CPU(cpu), KVM_SET_MSRS, cpu->kvm_msr_buf);

    if (ret < 0) {

        return ret;

    }



    if (ret < cpu->kvm_msr_buf->nmsrs) {

        struct kvm_msr_entry *e = &cpu->kvm_msr_buf->entries[ret];

        error_report("error: failed to set MSR 0x%" PRIx32 " to 0x%" PRIx64,

                     (uint32_t)e->index, (uint64_t)e->data);

    }



    assert(ret == cpu->kvm_msr_buf->nmsrs);

    return 0;

}
