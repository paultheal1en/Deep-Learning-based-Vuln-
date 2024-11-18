void kvm_s390_crw_mchk(S390CPU *cpu)

{

    kvm_s390_interrupt_internal(cpu, KVM_S390_MCHK, 1 << 28,

                                0x00400f1d40330000, 1);

}
