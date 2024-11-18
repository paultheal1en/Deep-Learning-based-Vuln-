int kvm_log_start(target_phys_addr_t phys_addr, target_phys_addr_t end_addr)

{

        return kvm_dirty_pages_log_change(phys_addr, end_addr,

                                          KVM_MEM_LOG_DIRTY_PAGES,

                                          KVM_MEM_LOG_DIRTY_PAGES);

}
