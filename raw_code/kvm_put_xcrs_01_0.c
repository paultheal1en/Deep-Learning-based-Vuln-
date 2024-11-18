static int kvm_put_xcrs(X86CPU *cpu)

{

    CPUX86State *env = &cpu->env;

    struct kvm_xcrs xcrs = {};



    if (!kvm_has_xcrs()) {

        return 0;

    }



    xcrs.nr_xcrs = 1;

    xcrs.flags = 0;

    xcrs.xcrs[0].xcr = 0;

    xcrs.xcrs[0].value = env->xcr0;

    return kvm_vcpu_ioctl(CPU(cpu), KVM_SET_XCRS, &xcrs);

}
