void kvm_irqchip_add_irq_route(KVMState *s, int irq, int irqchip, int pin)

{

    struct kvm_irq_routing_entry e;



    assert(pin < s->gsi_count);



    e.gsi = irq;

    e.type = KVM_IRQ_ROUTING_IRQCHIP;

    e.flags = 0;

    e.u.irqchip.irqchip = irqchip;

    e.u.irqchip.pin = pin;

    kvm_add_routing_entry(s, &e);

}
