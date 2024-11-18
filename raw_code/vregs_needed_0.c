static bool vregs_needed(void *opaque)

{

#ifdef CONFIG_KVM

    if (kvm_enabled()) {

        return kvm_check_extension(kvm_state, KVM_CAP_S390_VECTOR_REGISTERS);

    }

#endif

    return 0;

}
