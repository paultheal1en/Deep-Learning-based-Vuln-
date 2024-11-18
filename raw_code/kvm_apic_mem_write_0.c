static void kvm_apic_mem_write(void *opaque, target_phys_addr_t addr,

                               uint64_t data, unsigned size)

{

    MSIMessage msg = { .address = addr, .data = data };

    int ret;



    ret = kvm_irqchip_send_msi(kvm_state, msg);

    if (ret < 0) {

        fprintf(stderr, "KVM: injection failed, MSI lost (%s)\n",

                strerror(-ret));

    }

}
