static int kvm_max_vcpus(KVMState *s)

{

    int ret;



    /* Find number of supported CPUs using the recommended

     * procedure from the kernel API documentation to cope with

     * older kernels that may be missing capabilities.

     */

    ret = kvm_check_extension(s, KVM_CAP_MAX_VCPUS);

    if (ret) {

        return ret;

    }

    ret = kvm_check_extension(s, KVM_CAP_NR_VCPUS);

    if (ret) {

        return ret;

    }



    return 4;

}
