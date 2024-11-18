static int kvmppc_get_pvinfo(CPUPPCState *env, struct kvm_ppc_pvinfo *pvinfo)

 {

     PowerPCCPU *cpu = ppc_env_get_cpu(env);

     CPUState *cs = CPU(cpu);



    if (kvm_check_extension(cs->kvm_state, KVM_CAP_PPC_GET_PVINFO) &&

        !kvm_vm_ioctl(cs->kvm_state, KVM_PPC_GET_PVINFO, pvinfo)) {

        return 0;

    }



    return 1;

}
