int kvm_init(int smp_cpus)

{

    static const char upgrade_note[] =

        "Please upgrade to at least kernel 2.6.29 or recent kvm-kmod\n"

        "(see http://sourceforge.net/projects/kvm).\n";

    KVMState *s;

    int ret;

    int i;



    if (smp_cpus > 1) {

        fprintf(stderr, "No SMP KVM support, use '-smp 1'\n");

        return -EINVAL;

    }



    s = qemu_mallocz(sizeof(KVMState));



#ifdef KVM_CAP_SET_GUEST_DEBUG

    TAILQ_INIT(&s->kvm_sw_breakpoints);

#endif

    for (i = 0; i < ARRAY_SIZE(s->slots); i++)

        s->slots[i].slot = i;



    s->vmfd = -1;

    s->fd = open("/dev/kvm", O_RDWR);

    if (s->fd == -1) {

        fprintf(stderr, "Could not access KVM kernel module: %m\n");

        ret = -errno;

        goto err;

    }



    ret = kvm_ioctl(s, KVM_GET_API_VERSION, 0);

    if (ret < KVM_API_VERSION) {

        if (ret > 0)

            ret = -EINVAL;

        fprintf(stderr, "kvm version too old\n");

        goto err;

    }



    if (ret > KVM_API_VERSION) {

        ret = -EINVAL;

        fprintf(stderr, "kvm version not supported\n");

        goto err;

    }



    s->vmfd = kvm_ioctl(s, KVM_CREATE_VM, 0);

    if (s->vmfd < 0)

        goto err;



    /* initially, KVM allocated its own memory and we had to jump through

     * hooks to make phys_ram_base point to this.  Modern versions of KVM

     * just use a user allocated buffer so we can use regular pages

     * unmodified.  Make sure we have a sufficiently modern version of KVM.

     */

    if (!kvm_check_extension(s, KVM_CAP_USER_MEMORY)) {

        ret = -EINVAL;

        fprintf(stderr, "kvm does not support KVM_CAP_USER_MEMORY\n%s",

                upgrade_note);

        goto err;

    }



    /* There was a nasty bug in < kvm-80 that prevents memory slots from being

     * destroyed properly.  Since we rely on this capability, refuse to work

     * with any kernel without this capability. */

    if (!kvm_check_extension(s, KVM_CAP_DESTROY_MEMORY_REGION_WORKS)) {

        ret = -EINVAL;



        fprintf(stderr,

                "KVM kernel module broken (DESTROY_MEMORY_REGION).\n%s",

                upgrade_note);

        goto err;

    }



#ifdef KVM_CAP_COALESCED_MMIO

    s->coalesced_mmio = kvm_check_extension(s, KVM_CAP_COALESCED_MMIO);

#else

    s->coalesced_mmio = 0;

#endif



    s->broken_set_mem_region = 1;

#ifdef KVM_CAP_JOIN_MEMORY_REGIONS_WORKS

    ret = kvm_ioctl(s, KVM_CHECK_EXTENSION, KVM_CAP_JOIN_MEMORY_REGIONS_WORKS);

    if (ret > 0) {

        s->broken_set_mem_region = 0;

    }

#endif



    ret = kvm_arch_init(s, smp_cpus);

    if (ret < 0)

        goto err;



    kvm_state = s;



    return 0;



err:

    if (s) {

        if (s->vmfd != -1)

            close(s->vmfd);

        if (s->fd != -1)

            close(s->fd);

    }

    qemu_free(s);



    return ret;

}
