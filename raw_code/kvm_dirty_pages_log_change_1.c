static int kvm_dirty_pages_log_change(target_phys_addr_t phys_addr, target_phys_addr_t end_addr,

                                      unsigned flags,

                                      unsigned mask)

{

    KVMState *s = kvm_state;

    KVMSlot *mem = kvm_lookup_slot(s, phys_addr);

    if (mem == NULL)  {

            dprintf("invalid parameters %llx-%llx\n", phys_addr, end_addr);

            return -EINVAL;

    }



    flags = (mem->flags & ~mask) | flags;

    /* Nothing changed, no need to issue ioctl */

    if (flags == mem->flags)

            return 0;



    mem->flags = flags;



    return kvm_set_user_memory_region(s, mem);

}
