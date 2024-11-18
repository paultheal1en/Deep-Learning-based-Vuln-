void kvm_arm_reset_vcpu(ARMCPU *cpu)

{

    /* Re-init VCPU so that all registers are set to

     * their respective reset values.

     */

    kvm_arm_vcpu_init(CPU(cpu));

    write_kvmstate_to_list(cpu);

}
