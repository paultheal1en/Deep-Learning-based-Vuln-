int kvm_arch_init(KVMState *s, int smp_cpus)

{

    int ret;



    struct utsname utsname;



    uname(&utsname);

    lm_capable_kernel = strcmp(utsname.machine, "x86_64") == 0;



    /* create vm86 tss.  KVM uses vm86 mode to emulate 16-bit code

     * directly.  In order to use vm86 mode, a TSS is needed.  Since this

     * must be part of guest physical memory, we need to allocate it.  Older

     * versions of KVM just assumed that it would be at the end of physical

     * memory but that doesn't work with more than 4GB of memory.  We simply

     * refuse to work with those older versions of KVM. */

    ret = kvm_check_extension(s, KVM_CAP_SET_TSS_ADDR);

    if (ret <= 0) {

        fprintf(stderr, "kvm does not support KVM_CAP_SET_TSS_ADDR\n");

        return ret;

    }



    /* this address is 3 pages before the bios, and the bios should present

     * as unavaible memory.  FIXME, need to ensure the e820 map deals with

     * this?

     */

    /*

     * Tell fw_cfg to notify the BIOS to reserve the range.

     */

    if (e820_add_entry(0xfffbc000, 0x4000, E820_RESERVED) < 0) {

        perror("e820_add_entry() table is full");

        exit(1);

    }

    ret = kvm_vm_ioctl(s, KVM_SET_TSS_ADDR, 0xfffbd000);

    if (ret < 0) {

        return ret;

    }



    return kvm_init_identity_map_page(s);

}
