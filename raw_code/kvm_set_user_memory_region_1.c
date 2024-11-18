static int kvm_set_user_memory_region(KVMState *s, KVMSlot *slot)

{

    struct kvm_userspace_memory_region mem;



    mem.slot = slot->slot;

    mem.guest_phys_addr = slot->start_addr;

    mem.userspace_addr = (unsigned long)slot->ram;

    mem.flags = slot->flags;

    if (s->migration_log) {

        mem.flags |= KVM_MEM_LOG_DIRTY_PAGES;

    }

    if (mem.flags & KVM_MEM_READONLY) {

        /* Set the slot size to 0 before setting the slot to the desired

         * value. This is needed based on KVM commit 75d61fbc. */

        mem.memory_size = 0;

        kvm_vm_ioctl(s, KVM_SET_USER_MEMORY_REGION, &mem);

    }

    mem.memory_size = slot->memory_size;

    return kvm_vm_ioctl(s, KVM_SET_USER_MEMORY_REGION, &mem);

}
