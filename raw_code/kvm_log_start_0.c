int kvm_log_start(target_phys_addr_t phys_addr, ram_addr_t size)

{

        return kvm_dirty_pages_log_change(phys_addr, size,

                                          KVM_MEM_LOG_DIRTY_PAGES,

                                          KVM_MEM_LOG_DIRTY_PAGES);

}
