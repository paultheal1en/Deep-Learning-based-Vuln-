void kvm_arch_init_irq_routing(KVMState *s)

{

    if (!kvm_check_extension(s, KVM_CAP_IRQ_ROUTING)) {

        /* If kernel can't do irq routing, interrupt source

         * override 0->2 cannot be set up as required by HPET.

         * So we have to disable it.

         */

        no_hpet = 1;

    }

    /* We know at this point that we're using the in-kernel

     * irqchip, so we can use irqfds, and on x86 we know

     * we can use msi via irqfd and GSI routing.

     */

    kvm_irqfds_allowed = true;

    kvm_msi_via_irqfd_allowed = true;

    kvm_gsi_routing_allowed = true;

}
