static void enter_pgmcheck(S390CPU *cpu, uint16_t code)

{

    kvm_s390_interrupt(cpu, KVM_S390_PROGRAM_INT, code);

}
