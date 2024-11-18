int kvm_arch_init_vcpu(CPUState *cenv)

{

    int ret = 0;

    struct kvm_sregs sregs;



    sregs.pvr = cenv->spr[SPR_PVR];

    ret = kvm_vcpu_ioctl(cenv, KVM_SET_SREGS, &sregs);



    idle_timer = qemu_new_timer_ns(vm_clock, kvm_kick_env, cenv);



    return ret;

}
