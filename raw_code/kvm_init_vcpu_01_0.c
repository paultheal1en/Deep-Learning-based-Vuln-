int kvm_init_vcpu(CPUState *cpu)

{

    KVMState *s = kvm_state;

    long mmap_size;

    int ret;



    DPRINTF("kvm_init_vcpu\n");



    ret = kvm_vm_ioctl(s, KVM_CREATE_VCPU, (void *)kvm_arch_vcpu_id(cpu));

    if (ret < 0) {

        DPRINTF("kvm_create_vcpu failed\n");

        goto err;

    }



    cpu->kvm_fd = ret;

    cpu->kvm_state = s;

    cpu->kvm_vcpu_dirty = true;



    mmap_size = kvm_ioctl(s, KVM_GET_VCPU_MMAP_SIZE, 0);

    if (mmap_size < 0) {

        ret = mmap_size;

        DPRINTF("KVM_GET_VCPU_MMAP_SIZE failed\n");

        goto err;

    }



    cpu->kvm_run = mmap(NULL, mmap_size, PROT_READ | PROT_WRITE, MAP_SHARED,

                        cpu->kvm_fd, 0);

    if (cpu->kvm_run == MAP_FAILED) {

        ret = -errno;

        DPRINTF("mmap'ing vcpu state failed\n");

        goto err;

    }



    if (s->coalesced_mmio && !s->coalesced_mmio_ring) {

        s->coalesced_mmio_ring =

            (void *)cpu->kvm_run + s->coalesced_mmio * PAGE_SIZE;

    }



    ret = kvm_arch_init_vcpu(cpu);

err:

    return ret;

}
