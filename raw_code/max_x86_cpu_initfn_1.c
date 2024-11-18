static void max_x86_cpu_initfn(Object *obj)

{

    X86CPU *cpu = X86_CPU(obj);

    CPUX86State *env = &cpu->env;

    KVMState *s = kvm_state;



    /* We can't fill the features array here because we don't know yet if

     * "migratable" is true or false.

     */

    cpu->max_features = true;



    if (kvm_enabled()) {

        X86CPUDefinition host_cpudef = { };

        uint32_t eax = 0, ebx = 0, ecx = 0, edx = 0;



        host_vendor_fms(host_cpudef.vendor, &host_cpudef.family,

                        &host_cpudef.model, &host_cpudef.stepping);



        cpu_x86_fill_model_id(host_cpudef.model_id);



        x86_cpu_load_def(cpu, &host_cpudef, &error_abort);



        env->cpuid_min_level =

            kvm_arch_get_supported_cpuid(s, 0x0, 0, R_EAX);

        env->cpuid_min_xlevel =

            kvm_arch_get_supported_cpuid(s, 0x80000000, 0, R_EAX);

        env->cpuid_min_xlevel2 =

            kvm_arch_get_supported_cpuid(s, 0xC0000000, 0, R_EAX);



        if (lmce_supported()) {

            object_property_set_bool(OBJECT(cpu), true, "lmce", &error_abort);

        }

    } else {

        object_property_set_str(OBJECT(cpu), CPUID_VENDOR_AMD,

                                "vendor", &error_abort);

        object_property_set_int(OBJECT(cpu), 6, "family", &error_abort);

        object_property_set_int(OBJECT(cpu), 6, "model", &error_abort);

        object_property_set_int(OBJECT(cpu), 3, "stepping", &error_abort);

        object_property_set_str(OBJECT(cpu),

                                "QEMU TCG CPU version " QEMU_HW_VERSION,

                                "model-id", &error_abort);

    }



    object_property_set_bool(OBJECT(cpu), true, "pmu", &error_abort);

}
