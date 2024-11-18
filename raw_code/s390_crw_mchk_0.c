void s390_crw_mchk(S390CPU *cpu)

{

    if (kvm_enabled()) {

        kvm_s390_crw_mchk(cpu);

    } else {

        cpu_inject_crw_mchk(cpu);

    }

}
