static void x86_cpu_initfn(Object *obj)
{
    CPUState *cs = CPU(obj);
    X86CPU *cpu = X86_CPU(obj);
    X86CPUClass *xcc = X86_CPU_GET_CLASS(obj);
    CPUX86State *env = &cpu->env;
    FeatureWord w;
    cs->env_ptr = env;
    object_property_add(obj, "family", "int",
                        x86_cpuid_version_get_family,
                        x86_cpuid_version_set_family, NULL, NULL, NULL);
    object_property_add(obj, "model", "int",
                        x86_cpuid_version_get_model,
                        x86_cpuid_version_set_model, NULL, NULL, NULL);
    object_property_add(obj, "stepping", "int",
                        x86_cpuid_version_get_stepping,
                        x86_cpuid_version_set_stepping, NULL, NULL, NULL);
    object_property_add_str(obj, "vendor",
                            x86_cpuid_get_vendor,
                            x86_cpuid_set_vendor, NULL);
    object_property_add_str(obj, "model-id",
                            x86_cpuid_get_model_id,
                            x86_cpuid_set_model_id, NULL);
    object_property_add(obj, "tsc-frequency", "int",
                        x86_cpuid_get_tsc_freq,
                        x86_cpuid_set_tsc_freq, NULL, NULL, NULL);
    object_property_add(obj, "feature-words", "X86CPUFeatureWordInfo",
                        x86_cpu_get_feature_words,
                        NULL, NULL, (void *)env->features, NULL);
    object_property_add(obj, "filtered-features", "X86CPUFeatureWordInfo",
                        x86_cpu_get_feature_words,
                        NULL, NULL, (void *)cpu->filtered_features, NULL);
    cpu->hyperv_spinlock_attempts = HYPERV_SPINLOCK_NEVER_RETRY;
    for (w = 0; w < FEATURE_WORDS; w++) {
        int bitnr;
        for (bitnr = 0; bitnr < 32; bitnr++) {
            x86_cpu_register_feature_bit_props(cpu, w, bitnr);
        }
    }
    object_property_add_alias(obj, "sse3", obj, "pni", &error_abort);
    object_property_add_alias(obj, "pclmuldq", obj, "pclmulqdq", &error_abort);
    object_property_add_alias(obj, "sse4-1", obj, "sse4.1", &error_abort);
    object_property_add_alias(obj, "sse4-2", obj, "sse4.2", &error_abort);
    object_property_add_alias(obj, "xd", obj, "nx", &error_abort);
    object_property_add_alias(obj, "ffxsr", obj, "fxsr-opt", &error_abort);
    object_property_add_alias(obj, "i64", obj, "lm", &error_abort);
    object_property_add_alias(obj, "ds_cpl", obj, "ds-cpl", &error_abort);
    object_property_add_alias(obj, "tsc_adjust", obj, "tsc-adjust", &error_abort);
    object_property_add_alias(obj, "fxsr_opt", obj, "fxsr-opt", &error_abort);
    object_property_add_alias(obj, "lahf_lm", obj, "lahf-lm", &error_abort);
    object_property_add_alias(obj, "cmp_legacy", obj, "cmp-legacy", &error_abort);
    object_property_add_alias(obj, "nodeid_msr", obj, "nodeid-msr", &error_abort);
    object_property_add_alias(obj, "perfctr_core", obj, "perfctr-core", &error_abort);
    object_property_add_alias(obj, "perfctr_nb", obj, "perfctr-nb", &error_abort);
    object_property_add_alias(obj, "kvm_nopiodelay", obj, "kvm-nopiodelay", &error_abort);
    object_property_add_alias(obj, "kvm_mmu", obj, "kvm-mmu", &error_abort);
    object_property_add_alias(obj, "kvm_asyncpf", obj, "kvm-asyncpf", &error_abort);
    object_property_add_alias(obj, "kvm_steal_time", obj, "kvm-steal-time", &error_abort);
    object_property_add_alias(obj, "kvm_pv_eoi", obj, "kvm-pv-eoi", &error_abort);
    object_property_add_alias(obj, "kvm_pv_unhalt", obj, "kvm-pv-unhalt", &error_abort);
    object_property_add_alias(obj, "svm_lock", obj, "svm-lock", &error_abort);
    object_property_add_alias(obj, "nrip_save", obj, "nrip-save", &error_abort);
    object_property_add_alias(obj, "tsc_scale", obj, "tsc-scale", &error_abort);
    object_property_add_alias(obj, "vmcb_clean", obj, "vmcb-clean", &error_abort);
    object_property_add_alias(obj, "pause_filter", obj, "pause-filter", &error_abort);
    object_property_add_alias(obj, "sse4_1", obj, "sse4.1", &error_abort);
    object_property_add_alias(obj, "sse4_2", obj, "sse4.2", &error_abort);
    x86_cpu_load_def(cpu, xcc->cpu_def, &error_abort);
}