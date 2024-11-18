static int kvm_has_msr_hsave_pa(CPUState *env)

{

    kvm_supported_msrs(env);

    return has_msr_hsave_pa;

}
